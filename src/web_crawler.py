"""Web crawling and content extraction module prioritizing Firecrawl."""

import logging
import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from bs4 import BeautifulSoup
from src.config import SearchConfig
import urllib3

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    extraction_method: str = "firecrawl"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class FirecrawlCrawler:
    """Crawler using Firecrawl API for reliable content extraction with retry."""
    
    # Class-level rate limiting to track across all instances
    _last_request_time: float = 0.0
    _min_request_interval: float = 4.0  # 4 seconds between requests = max 15 req/min
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key
        self._client = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits by waiting if needed."""
        current_time = time.time()
        time_since_last = current_time - FirecrawlCrawler._last_request_time
        
        if time_since_last < FirecrawlCrawler._min_request_interval:
            wait_time = FirecrawlCrawler._min_request_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next Firecrawl request")
            time.sleep(wait_time)
        
        FirecrawlCrawler._last_request_time = time.time()
        
    @property
    def client(self):
        """Lazy-load Firecrawl client."""
        if self._client is None:
            try:
                from firecrawl import FirecrawlApp
                # Note: Check newer SDK usage, sometimes it is FirecrawlApp or Firecrawl
                # attempting standard import based on recent docs or fallback
                try:
                    self._client = FirecrawlApp(api_key=self.api_key)
                except ImportError:
                    from firecrawl import Firecrawl
                    self._client = Firecrawl(api_key=self.api_key)
                
                logger.info("Firecrawl client initialized")
            except ImportError:
                logger.error("firecrawl-py not installed. Run: pip install firecrawl-py")
                raise
        return self._client
    
    def scrape(self, url: str) -> Optional[WebContent]:
        """Scrape URL with retry mechanism."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Wait for rate limiting before making request
                self._wait_for_rate_limit()
                
                logger.info(f"Scraping with Firecrawl (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                # Using scrape method with direct kwargs
                result = self.client.scrape(url, formats=["markdown"])
                
                if not result:
                    logger.warning(f"Firecrawl returned empty for {url}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                
                # Handle potential different return structures depending on SDK version
                if hasattr(result, 'markdown'):
                    # Object-based return (e.g. Document object from new SDK)
                    markdown = result.markdown
                    metadata_obj = getattr(result, 'metadata', None)
                    # Convert metadata object to dict
                    if metadata_obj is None:
                        metadata = {}
                    elif isinstance(metadata_obj, dict):
                        metadata = metadata_obj
                    else:
                        # Convert object attributes to dict
                        metadata = {
                            'title': getattr(metadata_obj, 'title', ''),
                            'author': getattr(metadata_obj, 'author', None),
                            'publishedTime': getattr(metadata_obj, 'publishedTime', None),
                            'description': getattr(metadata_obj, 'description', None),
                            'sourceURL': getattr(metadata_obj, 'sourceURL', None),
                        }
                elif isinstance(result, dict):
                    # Dictionary-based return
                    markdown = result.get('markdown', '')
                    metadata = result.get('metadata', {}) or {}
                else:
                    logger.warning(f"Unexpected result type from Firecrawl: {type(result)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                
                if not markdown or len(markdown) < 50:
                    logger.warning(f"Firecrawl content too short for {url}: {len(markdown) if markdown else 0} chars")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None

                logger.info(f"Successfully scraped {url}: {len(markdown)} chars")
                return WebContent(
                    url=url,
                    title=metadata.get('title', '') or '',
                    main_text=markdown,
                    author=metadata.get('author'),
                    publish_date=metadata.get('publishedTime'),
                    metadata=metadata,
                    extraction_success=True,
                    extraction_method="firecrawl"
                )
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check for rate limit error and wait longer
                if 'rate limit' in error_str or '429' in error_str:
                    wait_time = 35  # Wait for rate limit reset (typically 30s + buffer)
                    logger.warning(f"Rate limit hit for {url}, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Firecrawl attempt {attempt + 1} failed for {url}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
        
        logger.error(f"Firecrawl failed for {url} after {self.max_retries} attempts: {last_error}")
        return None

class SimpleStaticCrawler:
    """Fallback crawler using requests and BeautifulSoup with retry."""
    
    def __init__(self, max_retries: int = 2, timeout: int = 15):
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
    
    def crawl(self, url: str) -> Optional[WebContent]:
        """Crawl URL with retry mechanism."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Static crawling (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                # Use session for better connection handling
                session = requests.Session()
                session.headers.update(self.headers)
                
                response = session.get(
                    url, 
                    timeout=self.timeout, 
                    verify=False,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Handle encoding properly
                response.encoding = response.apparent_encoding or 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove scripts and styles
                for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'aside', 'form']):
                    element.decompose()
                
                # Try to find main content area
                main_content = None
                content_selectors = [
                    'article', 'main', '.content', '.post-content', '.entry-content',
                    '#content', '.article-content', '.news-content', '.post-body'
                ]
                
                for selector in content_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                # Fallback to body if no main content found
                if not main_content:
                    main_content = soup.body or soup
                
                # Extract text with proper line breaks
                text = main_content.get_text(separator='\n\n', strip=True)
                
                # Clean up extra whitespace
                import re
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r' {2,}', ' ', text)
                
                title = soup.title.string.strip() if soup.title and soup.title.string else ""
                
                # Extract author if available
                author = None
                author_meta = soup.find('meta', {'name': 'author'})
                if author_meta:
                    author = author_meta.get('content')
                
                # Extract publish date if available
                publish_date = None
                date_meta = soup.find('meta', {'property': 'article:published_time'})
                if date_meta:
                    publish_date = date_meta.get('content')
                
                if len(text) < 100:
                    logger.warning(f"Content too short for {url}: {len(text)} chars")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return None
                
                logger.info(f"Successfully crawled {url}: {len(text)} chars")
                return WebContent(
                    url=url,
                    title=title,
                    main_text=text,
                    author=author,
                    publish_date=publish_date,
                    extraction_success=True,
                    extraction_method="static_fallback"
                )
                
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(f"Static crawl timeout for {url}")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Static crawl request error for {url}: {e}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Static crawl error for {url}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))
        
        logger.error(f"Static crawl failed for {url} after {self.max_retries} attempts: {last_error}")
        return None

class WebCrawler:
    """Main crawler facade prioritizing Firecrawl with improved fallback."""
    
    def __init__(self, config: Optional[SearchConfig] = None, use_selenium: bool = False):
        self.config = config or SearchConfig()
        
        # Firecrawl setup
        self.firecrawl = None
        key = getattr(self.config, 'firecrawl_api_key', None)
        if not key:
            # Fallback to env var or config defaults if not in passed config object
            import os
            key = os.getenv('FIRECRAWL_API_KEY', '')
            
        if key:
            try:
                self.firecrawl = FirecrawlCrawler(key, max_retries=3, retry_delay=2.0)
            except Exception as e:
                logger.warning(f"Could not init Firecrawl: {e}")
        else:
            logger.warning("No Firecrawl API Key found.")
            
        self.static = SimpleStaticCrawler(max_retries=2, timeout=15)
        
        # Track crawl statistics
        self.stats = {
            'total_attempts': 0,
            'firecrawl_success': 0,
            'static_success': 0,
            'failures': 0
        }
        
    def crawl(self, url: str, force_dynamic: bool = False) -> Optional[WebContent]:
        """Crawl URL with Firecrawl first, then fallback to static."""
        self.stats['total_attempts'] += 1
        
        # Always try Firecrawl first
        if self.firecrawl:
            content = self.firecrawl.scrape(url)
            if content and content.main_text and len(content.main_text) >= 50:
                self.stats['firecrawl_success'] += 1
                return content
            logger.info(f"Firecrawl failed or returned insufficient content for {url}, trying fallback")
        
        # Fallback to static crawler
        content = self.static.crawl(url)
        if content and content.main_text and len(content.main_text) >= 50:
            self.stats['static_success'] += 1
            return content
        
        self.stats['failures'] += 1
        logger.error(f"All crawl methods failed for {url}")
        return None

    def extract_content(self, url: str, force_dynamic: bool = False) -> Optional[Dict[str, Any]]:
        """Extract content and return as dictionary."""
        content = self.crawl(url)
        if content:
            return content.to_dict()
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get crawl statistics."""
        return self.stats.copy()
        
    def is_approved_source(self, url: str) -> bool:
        """Check if URL is from an approved source."""
        # List of priority domains for Vietnamese fact-checking
        priority_domains = [
            '.gov.vn', 'chinhphu.vn', 'gso.gov.vn', 'quochoi.vn',
            'thuvienphapluat.vn', 'vnexpress.net', 'tuoitre.vn',
            'thanhnien.vn', 'vtv.vn', 'vov.vn', 'nhandan.vn',
            'baochinhphu.vn', 'moh.gov.vn', 'mof.gov.vn'
        ]
        
        url_lower = url.lower()
        return any(domain in url_lower for domain in priority_domains)
