import asyncio
import os
import json
import csv
from openai import OpenAI
from dotenv import load_dotenv
from serpapi import GoogleSearch
import sys, os
from backend.process.relavencescorer import get_embedding, cosine_similarity
import numpy as np
import datetime

load_dotenv()
client      = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
MAX_RESULTS = 30 # Maximum number of results to keep per startup

# Define category descriptions for embedding comparison
CATEGORY_DESCRIPTIONS = {
    "funding": "Information about startup funding, investments, venture capital, angel investors, seed rounds, series A/B/C funding, accelerators, incubators, and financial backing.",
    "traction": "Information about product adoption, user growth, customer acquisition, market penetration, revenue growth, user engagement metrics, and product-market fit indicators.",
    "tech": "Information about technology stack, programming languages, frameworks, infrastructure, architecture, repositories, technical documentation, and engineering team.",
    "team": "Information about founders, co-founders, leadership team, executives, employees, hiring, team background, experience, and organizational structure.",
    "company_website": "Official company website, product pages, about us, company mission, vision, and official company information.",
    "other": "Miscellaneous information that doesn't fit into the above categories."
}

# Pre-compute embeddings for each category
category_embeddings = {category: get_embedding(description) for category, description in CATEGORY_DESCRIPTIONS.items()}

def generate_startup_search_queries(startup_name: str, n: int = 4, mode: str = "hybrid") -> list[str]:
    """
    Generate optimized search queries to gather comprehensive data about a startup.
    
    Args:
        startup_name: Name of the startup to research
        n: Number of queries to generate
        mode: Query generation mode - "targeted" (rule-based only), 
              "creative" (LLM-based only), or "hybrid" (both)
    
    Returns:
        List of search queries
    """
    # 1. Create targeted queries that will find high-quality sources
    targeted_queries = [
        f"\"{startup_name}\" site:ycombinator.com OR site:crunchbase.com OR site:linkedin.com OR site:angel.co",
        f"\"{startup_name}\" funding OR investors OR raised",
        f"\"{startup_name}\" official website",
        f"\"{startup_name}\" founders OR team OR leadership",
        f"\"{startup_name}\" product OR technology OR tech stack"
    ]
    
    # Return early if only targeted queries are requested
    if mode.lower() == "targeted":
        return targeted_queries[:n]
    
    # 2. Generate queries using LLM for more creative/comprehensive coverage
    prompt = (
        f"You are an expert at crafting search queries for gathering startup intelligence. "
        f"Generate exactly {n} search queries to find comprehensive information about the startup: '{startup_name}'.\n"
        f"Create diverse queries that will find information from these sources:\n"
        "1. Company website and product information\n"
        "2. Funding history (Crunchbase, AngelList)\n"
        "3. Product traction (Product Hunt, reviews)\n"
        "4. Technical credibility (GitHub, tech stack)\n"
        "5. Team information (LinkedIn, founder backgrounds)\n"
        "Return only the raw query strings, one per line."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You craft precise search queries for startup intelligence gathering."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3,
    )
    text = resp.choices[0].message.content or ""
    llm_queries = [q.strip() for q in text.splitlines() if q.strip()]
    
    # Return early if only creative queries are requested
    if mode.lower() == "creative":
        return llm_queries[:n]
    
    # 3. For hybrid mode, combine both sets of queries, removing duplicates
    combined_queries = []
    seen_queries = set()
    
    # Add targeted queries first (higher priority)
    for query in targeted_queries:
        if query.lower() not in seen_queries:
            combined_queries.append(query)
            seen_queries.add(query.lower())
    
    # Add LLM queries next, skipping any that are too similar to existing queries
    for query in llm_queries:
        # Skip if exact match or very similar to existing query
        if query.lower() not in seen_queries and not any(
            similar_queries(query, existing) for existing in combined_queries
        ):
            combined_queries.append(query)
            seen_queries.add(query.lower())
    
    # Return the requested number of queries
    return combined_queries[:n]

def similar_queries(query1: str, query2: str) -> bool:
    """
    Check if two queries are very similar to avoid redundancy.
    """
    # Convert to lowercase for comparison
    q1 = query1.lower()
    q2 = query2.lower()
    
    # If one is a substring of the other with high overlap
    if q1 in q2 and len(q1) > len(q2) * 0.8:
        return True
    if q2 in q1 and len(q2) > len(q1) * 0.8:
        return True
    
    # Count common words (for more sophisticated comparison)
    words1 = set(q1.split())
    words2 = set(q2.split())
    common_words = words1.intersection(words2)
    
    # If they share more than 70% of their words, consider them similar
    if min(len(words1), len(words2)) == 0:
        return False
    similarity = len(common_words) / min(len(words1), len(words2))
    return similarity > 0.7

def get_best_category_and_score(similarities: dict) -> tuple[str, float]:
    """
    Find the category with the highest similarity score.
    """
    best_category = None
    best_score = -1
    
    for category, score in similarities.items():
        if score > best_score:
            best_score = score
            best_category = category
    
    return best_category, float(best_score)

def calculate_weighted_average(current_avg: float, new_value: float, count: int) -> float:
    """
    Calculate weighted average when adding a new value to an existing average.
    """
    return (current_avg * (count - 1) + new_value) / count

def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that might contain markdown code blocks.
    """
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def categorize_with_embeddings(result: dict, startup_name: str) -> tuple[str, float]:
    """
    Categorize a search result using embeddings for semantic understanding.
    Returns a tuple of (category, confidence_score).
    """
    url = result.get("link", "").lower()
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    
    # Rule-based categorization for high-confidence matches (1.0 confidence)
    # Company website detection
    if startup_name.lower() in url and not any(domain in url for domain in ["linkedin.com", "crunchbase.com", "github.com"]):
        return "company_website", 1.0
    
    # Known funding platforms
    if any(domain in url for domain in ["crunchbase.com", "angellist.co", "angel.co", "openvc.app", "dealroom.co", "seedtable.com"]):
        return "funding", 1.0
    
    # Y Combinator is definitely funding
    if "ycombinator.com" in url or ("yc" in title and "batch" in snippet):
        return "funding", 1.0
    
    # GitHub is definitely tech
    if "github.com" in url:
        return "tech", 1.0
    
    # LinkedIn is typically about team
    if "linkedin.com" in url:
        return "team", 1.0
    
    # For everything else, use embeddings for semantic categorization
    # Combine title and snippet for better context
    content = f"{title}. {snippet}"
    content_embedding = get_embedding(content)
    
    # Calculate similarity with each category
    similarities = {}
    for category, category_embedding in category_embeddings.items():
        similarities[category] = cosine_similarity(content_embedding, category_embedding)
    
    # Get the category with highest similarity and its score
    return get_best_category_and_score(similarities)

def scrape_startup_data(startup_name: str, queries: list[str], results_per_query: int = 10, enrich_content: bool = False) -> dict:
    """
    Run each query and collect all unique search results, organized by category.
    
    Args:
        startup_name: Name of the startup to research
        queries: List of search queries to run
        results_per_query: Maximum results to keep per query
        enrich_content: Whether to scrape the actual content of each page
    """
    all_results = {
        "company_website": [],
        "funding": [],
        "traction": [],
        "tech": [],
        "team": [],
        "other": []
    }
    confidence_scores = {
        "company_website": 0.0,
        "funding": 0.0,
        "traction": 0.0,
        "tech": 0.0,
        "team": 0.0,
        "other": 0.0
    }
    category_counts = {category: 0 for category in all_results}
    seen_links = set()
    
    for query in queries:
        params = {
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "q": query,
            "num": results_per_query,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "safe": "off",
            "filter": "0"  # Turns off Google's duplicate content filter
        }
        search = GoogleSearch(params)
        data = search.get_dict()
        
        for item in data.get("organic_results", []):
            link = item.get("link")
            if link and link not in seen_links:
                seen_links.add(link)
                result = {
                    "title": item.get("title", ""),
                    "link": link,
                    "snippet": item.get("snippet", ""),
                    "source_query": query
                }
                
                # Optionally enrich with actual page content
                if enrich_content:
                    try:
                        from backend.scrap.scraper import scrape_website
                        scraped = scrape_website(link)
                        if scraped.get("status") == "success":
                            result["content"] = scraped.get("content", "")[:1000]  # First 1000 chars
                            result["meta_description"] = scraped.get("meta_description", "")
                    except Exception as e:
                        print(f"Error enriching content for {link}: {e}")
                
                # Use embedding-based categorization with confidence score
                category, confidence = categorize_with_embeddings(result, startup_name)
                result["confidence"] = round(confidence, 2)
                
                # Update category confidence scores (weighted average)
                category_counts[category] += 1
                confidence_scores[category] = calculate_weighted_average(
                    confidence_scores[category], confidence, category_counts[category]
                )
                
                all_results[category].append(result)
    
    # Limit each category to MAX_RESULTS
    for category in all_results:
        all_results[category] = all_results[category][:MAX_RESULTS]
    
    # Round confidence scores for readability
    confidence_scores = {k: round(v, 2) for k, v in confidence_scores.items() if category_counts[k] > 0}
    
    return {
        "results": all_results,
        "confidence_scores": confidence_scores,
        "category_counts": {k: v for k, v in category_counts.items() if v > 0}
    }

def generate_startup_profile(startup_name: str, data: dict, query_mode: str = "hybrid", enrich_content: bool = False) -> dict:
    """
    Use OpenAI to generate a structured startup profile from the gathered data.
    
    Args:
        startup_name: Name of the startup to research
        data: Dictionary containing categorized search results
        query_mode: Mode used for query generation
        enrich_content: Whether content enrichment was performed
    """
    # Prepare the data for the prompt
    categorized_data = data["results"]
    confidence_scores = data["confidence_scores"]
    category_counts = data["category_counts"]
    
    data_summary = []
    for category, results in categorized_data.items():
        if results:
            confidence = confidence_scores.get(category, 0)
            data_summary.append(f"{category.upper()} ({len(results)} sources, confidence: {confidence:.2f}):")
            for i, result in enumerate(results[:3]):  # Include only top 3 for the prompt
                snippet = result.get("snippet", "")
                content = result.get("content", "")  # Use enriched content if available
                text = content if content else snippet
                data_summary.append(f"- {result['title']}: {text[:150]}...")
    
    data_text = "\n".join(data_summary)
    
    prompt = (
        f"Based on the following information about the startup '{startup_name}', "
        f"create a comprehensive startup profile in JSON format.\n\n"
        f"DATA SOURCES:\n{data_text}\n\n"
        f"Generate a JSON object with these fields:\n"
        f"1. company_name: The startup name\n"
        f"2. summary: A clear 2-3 sentence description of what they do\n"
        f"3. product: Key features and value proposition\n"
        f"4. team: Any information about founders and key team members\n"
        f"5. traction: Any signals of growth or market validation\n"
        f"6. funding: Any funding information found\n"
        f"7. market: The market they operate in and potential fit\n"
        f"8. tech_stack: Any technology information found\n"
        # f"9. sources: List of the most important URLs to reference\n\n"
        f"Return only a **valid JSON object**.\n"
        f"If any field is missing or uncertain, return it as null or \"unknown\" — do not make anything up.\n"
        f"Do not include any explanations or markdown formatting, just the raw JSON."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You create structured startup profiles from research data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.4,
    )
    
    try:
        profile_text = resp.choices[0].message.content or "{}"
        # Extract JSON if it's wrapped in markdown code blocks
        profile_text = extract_json_from_text(profile_text)
        
        profile = json.loads(profile_text)
        
        # Add metadata to the profile
        profile["confidence_scores"] = confidence_scores
        profile["category_counts"] = category_counts
        profile["raw_data"] = categorized_data
        
        # Add profile metadata with query information and timestamp
        profile["profile_metadata"] = {
            "query_mode": query_mode,
            "enriched": enrich_content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return profile
    except json.JSONDecodeError:
        # Fallback when JSON parsing fails
        return {
            "company_name": startup_name,
            "summary": "Profile generation failed. Please check raw data.",
            "confidence_scores": confidence_scores,
            "category_counts": category_counts,
            "raw_data": categorized_data,
            "profile_metadata": {
                "query_mode": query_mode,
                "enriched": enrich_content,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }

def save_startup_profile(profile: dict, output_dir: str = "data/startups"):
    """
    Save the startup profile to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    startup_name = profile.get("company_name", "unknown").lower().replace(" ", "_")
    output_path = f"{output_dir}/{startup_name}_profile.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved startup profile to {output_path}")
    return output_path

def main(startup_name: str = None, query_mode: str = "hybrid", enrich_content: bool = False):
    """
    Main function to gather intelligence on a startup.
    
    Args:
        startup_name: Name of the startup to research
        query_mode: Mode for query generation - "targeted", "creative", or "hybrid"
        enrich_content: Whether to scrape the actual content of each page
    """
    # Get startup name from command line or use default
    if not startup_name:
        startup_name = input("Enter startup name: ")
    
    print(f"Gathering intelligence on: {startup_name}")
    
    # Generate search queries using specified approach
    queries = generate_startup_search_queries(startup_name, n=4, mode=query_mode)
    print(f"Generated {len(queries)} search queries ({query_mode} mode):")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")
    
    # Scrape and categorize data
    data = scrape_startup_data(startup_name, queries, enrich_content=enrich_content)
    total_results = sum(len(results) for results in data["results"].values())
    print(f"Found {total_results} unique results across {len(data['category_counts'])} categories")
    print(f"Confidence scores: {data['confidence_scores']}")
    
    # Generate structured profile - pass query_mode and enrich_content
    profile = generate_startup_profile(startup_name, data, query_mode, enrich_content)
    print(f"Generated startup profile for {profile.get('company_name', startup_name)}")
    
    # Save profile
    output_path = save_startup_profile(profile)
    print(f"Process complete. Profile saved to {output_path}")
    
    return profile

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Simple test function for query.py
        print("Testing startup intelligence gathering...")
        test_startup = None
        if not test_startup:
            test_startup = input("Enter startup name: ")
        print(f"Using test startup: {test_startup}")
        
        # Test query generation
        queries = generate_startup_search_queries(test_startup, n=2, mode="targeted")
        print(f"Generated queries: {queries}")
        
        # Test minimal data scraping (1 query, few results)
        data = scrape_startup_data(test_startup, queries[:1], results_per_query=3)
        print(f"Categories found: {list(data['category_counts'].keys())}")
        
        # Test profile generation
        profile = generate_startup_profile(test_startup, data, "targeted", False)
        print(f"Profile generated with metadata: {profile['profile_metadata']}")
        
        # Save to test location
        test_path = save_startup_profile(profile, "data/startups")
        print(f"Test complete. Profile saved to {test_path}")
    else:
        # Run the main function as usual
        main()
