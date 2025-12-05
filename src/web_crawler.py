"""Web crawling and content extraction module for Vietnamese news sources."""

import time
import logging
import hashlib
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass, field, asdict
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import trafilatura

from src.config import SearchConfig

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Selenium not available. Dynamic content crawling will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class WebContent:
    """Represents extracted web content."""
    
    url: str
    title: str
    main_text: str
    author: Optional[str] = None
    publish_date: Optional[str] = None  # ISO format string
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_success: bool = True
    extraction_method: str = "trafilatura"
    
    def __post_init__(self):
        """Validate web content data."""
        if not self.url:
            raise ValueError("URL cannot be empty")
        
        if not self.title and self.extraction_success:
            logger.warning(f"Title is empty for {self.url}")
        
        if not self.main_text and self.extraction_success:
            logger.warning(f"Main text is empty for {self.url}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebContent":
        """Create from dictionary."""
        return cls(**data)


class StaticHTMLCrawler:
    """Crawler for static HTML content using BeautifulSoup."""
    
    # Approved Vietnamese news sources
    APPROVED_SOURCES = [
        "vnexpress.net",
        "vtv.vn",
        "vov.vn",
        "tuoitre.vn",
        "thanhnien.vn",
        "baochinhphu.vn"
    ]
    
    # User agents for rotation
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the static HTML crawler.
        
        Args:
            config: Search configuration object
        """
        self.config = config or SearchConfig()
        self.session = requests.Session()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.request_timestamps: Dict[str, List[float]] = {}
        self.user_agent_index = 0
        
        # Set default timeout
        self.timeout = 30
        
        # Rate limiting: minimum seconds between requests to same domain
        self.min_request_interval = 1.0
    
    def _get_user_agent(self) -> str:
        """Get next user agent from rotation."""
        user_agent = self.USER_AGENTS[self.user_agent_index]
        self.user_agent_index = (self.user_agent_index + 1) % len(self.USER_AGENTS)
        return user_agent
    
    def _is_approved_source(self, url: str) -> bool:
        """
        Check if URL is from an approved source.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is from approved source
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return any(approved in domain for approved in self.APPROVED_SOURCES)
    
    def _get_robots_parser(self, url: str) -> Optional[RobotFileParser]:
        """
        Get robots.txt parser for domain.
        
        Args:
            url: URL to get robots.txt for
            
        Returns:
            RobotFileParser object or None if unavailable
        """
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check cache
        if base_url in self.robots_cache:
            return self.robots_cache[base_url]
        
        # Fetch and parse robots.txt
        robots_url = urljoin(base_url, '/robots.txt')
        parser = RobotFileParser()
        parser.set_url(robots_url)
        
        try:
            parser.read()
            self.robots_cache[base_url] = parser
            logger.debug(f"Loaded robots.txt from {robots_url}")
            return parser
        except Exception as e:
            logger.warning(f"Could not load robots.txt from {robots_url}: {e}")
            # Cache None to avoid repeated failures
            self.robots_cache[base_url] = None
            return None
    
    def _can_fetch(self, url: str, user_agent: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            user_agent: User agent string
            
        Returns:
            True if URL can be fetched
        """
        parser = self._get_robots_parser(url)
        
        if parser is None:
            # If robots.txt unavailable, allow fetching
            return True
        
        return parser.can_fetch(user_agent, url)
    
    def _apply_rate_limit(self, domain: str) -> None:
        """
        Apply rate limiting for domain.
        
        Args:
            domain: Domain to rate limit
        """
        current_time = time.time()
        
        # Initialize timestamp list for domain if needed
        if domain not in self.request_timestamps:
            self.request_timestamps[domain] = []
        
        # Clean old timestamps (older than 60 seconds)
        self.request_timestamps[domain] = [
            ts for ts in self.request_timestamps[domain]
            if current_time - ts < 60
        ]
        
        # Check if we need to wait
        if self.request_timestamps[domain]:
            last_request = self.request_timestamps[domain][-1]
            time_since_last = current_time - last_request
            
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
                time.sleep(wait_time)
        
        # Record this request
        self.request_timestamps[domain].append(time.time())
    
    def fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string, or None if fetch failed
        """
        # Validate approved source
        if not self._is_approved_source(url):
            logger.error(f"URL not from approved source: {url}")
            return None
        
        # Get user agent
        user_agent = self._get_user_agent()
        
        # Check robots.txt
        if not self._can_fetch(url, user_agent):
            logger.warning(f"Robots.txt disallows fetching: {url}")
            return None
        
        # Apply rate limiting
        parsed = urlparse(url)
        domain = parsed.netloc
        self._apply_rate_limit(domain)
        
        # Fetch content
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Try to decode with proper encoding
            if response.encoding:
                html = response.text
            else:
                # Detect encoding
                html = response.content.decode('utf-8', errors='replace')
            
            logger.info(f"Successfully fetched {url}")
            return html
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def extract_content(self, html: str, url: str) -> Optional[WebContent]:
        """
        Extract main content from HTML using trafilatura.
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            WebContent object or None if extraction failed
        """
        try:
            # Extract with trafilatura
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
                url=url
            )
            
            if not extracted:
                logger.warning(f"Trafilatura extraction returned empty content for {url}")
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html, default_url=url)
            
            # Build WebContent object
            title = metadata.title if metadata and metadata.title else ""
            author = metadata.author if metadata and metadata.author else None
            publish_date = metadata.date if metadata and metadata.date else None
            
            # Validate extraction quality
            if len(extracted) < 100:
                logger.warning(f"Extracted text too short ({len(extracted)} chars) for {url}")
                return WebContent(
                    url=url,
                    title=title,
                    main_text=extracted,
                    author=author,
                    publish_date=publish_date,
                    extraction_success=False,
                    extraction_method="trafilatura"
                )
            
            if not title:
                logger.warning(f"No title extracted for {url}")
            
            content = WebContent(
                url=url,
                title=title,
                main_text=extracted,
                author=author,
                publish_date=publish_date,
                metadata={
                    'sitename': metadata.sitename if metadata and metadata.sitename else None,
                    'description': metadata.description if metadata and metadata.description else None,
                    'categories': metadata.categories if metadata and metadata.categories else None,
                    'tags': metadata.tags if metadata and metadata.tags else None,
                },
                extraction_success=True,
                extraction_method="trafilatura"
            )
            
            logger.info(f"Successfully extracted content from {url}")
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def crawl(self, url: str) -> Optional[WebContent]:
        """
        Crawl URL and extract content.
        
        Args:
            url: URL to crawl
            
        Returns:
            WebContent object or None if crawl failed
        """
        # Fetch HTML
        html = self.fetch_html(url)
        if html is None:
            return None
        
        # Extract content
        content = self.extract_content(html, url)
        return content



class DynamicContentCrawler:
    """Crawler for dynamic content using Selenium with headless Chrome."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the dynamic content crawler.
        
        Args:
            config: Search configuration object
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not installed. Install with: pip install selenium")
        
        self.config = config or SearchConfig()
        self.static_crawler = StaticHTMLCrawler(config)
        self.driver = None
        
        # Selenium settings
        self.page_load_timeout = 30
        self.implicit_wait = 10
        self.explicit_wait = 15
    
    def _setup_driver(self) -> webdriver.Chrome:
        """
        Setup headless Chrome driver.
        
        Returns:
            Chrome WebDriver instance
        """
        chrome_options = Options()
        
        # Headless mode
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        
        # Performance optimizations
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--blink-settings=imagesEnabled=false')
        
        # User agent
        user_agent = self.static_crawler._get_user_agent()
        chrome_options.add_argument(f'user-agent={user_agent}')
        
        # Language
        chrome_options.add_argument('--lang=vi-VN')
        
        # Create driver
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.page_load_timeout)
            driver.implicitly_wait(self.implicit_wait)
            
            logger.info("Chrome driver initialized successfully")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            raise
    
    def _wait_for_content(self, driver: webdriver.Chrome) -> None:
        """
        Wait for dynamic content to load.
        
        Args:
            driver: Chrome WebDriver instance
        """
        try:
            # Wait for body to be present
            WebDriverWait(driver, self.explicit_wait).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for common article containers
            article_selectors = [
                "article",
                ".article-content",
                ".article-body",
                ".content-detail",
                ".detail-content",
                "#article-content",
            ]
            
            for selector in article_selectors:
                try:
                    WebDriverWait(driver, 2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.debug(f"Found article content with selector: {selector}")
                    break
                except:
                    continue
            
            # Additional wait for JavaScript execution
            time.sleep(2)
            
        except Exception as e:
            logger.warning(f"Timeout waiting for content: {e}")
    
    def fetch_dynamic_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML content with JavaScript rendering.
        
        Args:
            url: URL to fetch
            
        Returns:
            Rendered HTML content as string, or None if fetch failed
        """
        # Validate approved source
        if not self.static_crawler._is_approved_source(url):
            logger.error(f"URL not from approved source: {url}")
            return None
        
        # Apply rate limiting
        parsed = urlparse(url)
        domain = parsed.netloc
        self.static_crawler._apply_rate_limit(domain)
        
        driver = None
        try:
            # Setup driver
            driver = self._setup_driver()
            
            # Load page
            logger.info(f"Loading dynamic content from {url}")
            driver.get(url)
            
            # Wait for content
            self._wait_for_content(driver)
            
            # Get rendered HTML
            html = driver.page_source
            
            logger.info(f"Successfully fetched dynamic content from {url}")
            return html
            
        except Exception as e:
            logger.error(f"Error fetching dynamic content from {url}: {e}")
            return None
            
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def crawl(self, url: str, fallback_to_static: bool = True) -> Optional[WebContent]:
        """
        Crawl URL with dynamic content support.
        
        Args:
            url: URL to crawl
            fallback_to_static: If True, fallback to static parsing if Selenium fails
            
        Returns:
            WebContent object or None if crawl failed
        """
        # Try dynamic crawling
        html = self.fetch_dynamic_html(url)
        
        if html is None:
            if fallback_to_static:
                logger.info(f"Falling back to static crawling for {url}")
                return self.static_crawler.crawl(url)
            else:
                return None
        
        # Extract content
        content = self.static_crawler.extract_content(html, url)
        
        if content:
            content.extraction_method = "selenium+trafilatura"
        
        return content


class WebCrawler:
    """Unified web crawler with both static and dynamic support."""
    
    def __init__(self, config: Optional[SearchConfig] = None, use_selenium: bool = False):
        """
        Initialize the web crawler.
        
        Args:
            config: Search configuration object
            use_selenium: If True, use Selenium for JavaScript rendering
        """
        self.config = config or SearchConfig()
        self.static_crawler = StaticHTMLCrawler(config)
        
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        if use_selenium and not SELENIUM_AVAILABLE:
            logger.warning("Selenium requested but not available. Using static crawler only.")
        
        self.dynamic_crawler = None
        if self.use_selenium:
            try:
                self.dynamic_crawler = DynamicContentCrawler(config)
            except Exception as e:
                logger.error(f"Failed to initialize dynamic crawler: {e}")
                self.use_selenium = False
    
    def crawl(self, url: str, force_dynamic: bool = False) -> Optional[WebContent]:
        """
        Crawl URL and extract content.
        
        Args:
            url: URL to crawl
            force_dynamic: If True, force use of Selenium even if not default
            
        Returns:
            WebContent object or None if crawl failed
        """
        # Determine which crawler to use
        use_dynamic = force_dynamic or self.use_selenium
        
        if use_dynamic and self.dynamic_crawler:
            return self.dynamic_crawler.crawl(url, fallback_to_static=True)
        else:
            return self.static_crawler.crawl(url)
    
    def is_approved_source(self, url: str) -> bool:
        """
        Check if URL is from an approved source.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is from approved source
        """
        return self.static_crawler._is_approved_source(url)
