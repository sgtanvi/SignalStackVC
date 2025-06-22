import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from urllib.parse import urlparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# User agent list to rotate and avoid being blocked
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

def get_random_user_agent():
    """Return a random user agent from the list."""
    return random.choice(USER_AGENTS)

def scrape_website(url, timeout=10, max_retries=2):
    """
    Scrape content from a website URL.
    
    Args:
        url (str): The URL to scrape
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        dict: Dictionary containing the scraped content and metadata
    """
    if not url or not url.startswith(('http://', 'https://')):
        return {"error": "Invalid URL", "content": ""}
    
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    domain = urlparse(url).netloc
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Scraping {url} (Attempt {attempt+1}/{max_retries})")
            response = requests.get(url, headers=headers, timeout=timeout)
            
            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "No title found"
                
                # Extract main content (prioritize article content, then main, then body)
                content = ""
                
                # Try to find article content first
                article = soup.find('article')
                if article:
                    content = article.get_text(separator=' ', strip=True)
                
                # If no article, try main content
                if not content:
                    main = soup.find('main')
                    if main:
                        content = main.get_text(separator=' ', strip=True)
                
                # If still no content, try common content divs
                if not content:
                    for div_id in ['content', 'main-content', 'mainContent', 'bodyContent']:
                        content_div = soup.find('div', id=div_id)
                        if content_div:
                            content = content_div.get_text(separator=' ', strip=True)
                            break
                
                # If still no content, get all paragraph text
                if not content:
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                
                # If still no content, get body text
                if not content:
                    body = soup.find('body')
                    if body:
                        content = body.get_text(separator=' ', strip=True)
                
                # Clean up content
                content = ' '.join(content.split())
                
                # Extract meta description
                meta_desc = ""
                meta_tag = soup.find('meta', attrs={'name': 'description'})
                if meta_tag:
                    meta_desc = meta_tag.get('content', '')
                
                return {
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "meta_description": meta_desc,
                    "content": content[:5000],  # Limit content to first 5000 chars
                    "content_length": len(content),
                    "status": "success"
                }
            else:
                logger.warning(f"Failed to scrape {url}: HTTP {response.status_code}")
                time.sleep(1 + attempt)  # Incremental backoff
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            time.sleep(1 + attempt)  # Incremental backoff
    
    return {
        "url": url,
        "domain": domain,
        "title": "",
        "meta_description": "",
        "content": "",
        "content_length": 0,
        "status": "error",
        "error": "Failed after max retries"
    }

def scrape_multiple_urls(urls, delay_range=(1, 3)):
    """
    Scrape multiple URLs with a random delay between requests.
    
    Args:
        urls (list): List of URLs to scrape
        delay_range (tuple): Range for random delay between requests in seconds
        
    Returns:
        list: List of dictionaries containing scraped content
    """
    results = []
    
    for i, url in enumerate(urls):
        logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
        
        # Scrape the website
        result = scrape_website(url)
        results.append(result)
        
        # Add a random delay between requests to avoid being blocked
        if i < len(urls) - 1:  # No need to delay after the last request
            delay = random.uniform(delay_range[0], delay_range[1])
            logger.info(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)
    
    return results

def save_scraped_content(results, startup_name, output_dir="data/scraped"):
    """
    Save scraped content to files.
    
    Args:
        results (list): List of dictionaries containing scraped content
        startup_name (str): Name of the startup
        output_dir (str): Directory to save the files
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean startup name for filename
    safe_name = startup_name.lower().replace(" ", "_")
    
    # Save as JSON
    import json
    output_path = f"{output_dir}/{safe_name}_scraped_content.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved scraped content to {output_path}")
    return output_path

def test():
    """Test the scraper with a few example URLs."""
    test_urls = [
        "https://www.crunchbase.com/organization/stripe",
        "https://techcrunch.com/2021/03/14/stripe-closes-600m-round-at-a-95b-valuation/",
        "https://stripe.com"
    ]
    
    print(f"Testing scraper with {len(test_urls)} URLs...")
    results = scrape_multiple_urls(test_urls)
    
    for result in results:
        print(f"\nURL: {result['url']}")
        print(f"Status: {result['status']}")
        print(f"Title: {result['title']}")
        print(f"Content length: {result['content_length']} characters")
        print(f"Content preview: {result['content'][:150]}...")
    
    # Save results
    output_path = save_scraped_content(results, "Stripe", "data/test_scraped")
    print(f"\nSaved test results to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test()
    else:
        print("Run with --test to test the scraper")