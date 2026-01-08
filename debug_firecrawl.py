from firecrawl import FirecrawlApp
import firecrawl

print(f"Firecrawl Version: {getattr(firecrawl, '__version__', 'Unknown')}")

app = FirecrawlApp(api_key="TEST")

# Test 1: scrape_url
try:
    print("Testing app.scrape_url...")
    # Just checking existence, not calling with real network to avoid auth error if possible, 
    # but simplest is to check attribute
    if hasattr(app, 'scrape_url'):
        print("  - Exists: YES")
    else:
        print("  - Exists: NO")
except Exception as e:
    print(f"  - Error checking: {e}")

# Test 2: scrape
try:
    print("Testing app.scrape...")
    if hasattr(app, 'scrape'):
        print("  - Exists: YES")
    else:
        print("  - Exists: NO")
except Exception as e:
    print(f"  - Error checking: {e}")

# Test 3: v1.scrape_url
try:
    print("Testing app.v1.scrape_url...")
    if hasattr(app, 'v1') and hasattr(app.v1, 'scrape_url'):
         print("  - Exists: YES")
    else:
         print("  - Exists: NO")
except Exception as e:
    print(f"  - Error checking: {e}")
    
# Test 4: print all methods again carefully
print("\nAll methods:")
print(dir(app))
