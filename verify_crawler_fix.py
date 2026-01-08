import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.web_crawler import WebCrawler

logging.basicConfig(level=logging.INFO)

def test_crawler():
    print("Initializing WebCrawler...")
    # Use a dummy key if not set, just to test method dispatch
    if not os.getenv("FIRECRAWL_API_KEY"):
         os.environ["FIRECRAWL_API_KEY"] = "fc-TEST_KEY"
         
    crawler = WebCrawler()
    
    # We want to test that it CALLS .scrape() without crashing on AttributeError
    # It might fail with 401 Unauthorized if key is bad, but that proves the method exists and was called.
    url = "https://example.com"
    print(f"Attempting to crawl {url} with Firecrawl...")
    
    try:
        content = crawler.crawl(url)
        print("Crawl call finished (success or handled failure).")
        if content:
            print(f"Result: {content.url} - {len(content.main_text)} chars")
        else:
            print("Result: None (expected if key is invalid or mock)")
            
    except AttributeError as e:
        print(f"FAILED with AttributeError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Finished with expected error (likely auth): {e}")

if __name__ == "__main__":
    test_crawler()
