import asyncio
import os
import json
import csv
from openai import OpenAI
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()
client      = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
MAX_RESULTS = 30 # Maximum number of results to keep per startup

def generate_startup_search_queries(startup_name: str, n: int = 2) -> list[str]:
    """
    Generate optimized search queries to gather comprehensive data about a startup.
    """
    # ai_usecase = payload.get("ai_usecase")
    # labels     = payload.get("input_labels", [])
    # labels_str = ", ".join(labels)

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
    queries = [q.strip() for q in text.splitlines() if q.strip()]

    # Fallback if nothing returned
    if not queries:
        queries = [
            f"{startup_name} company website",
            f"{startup_name} crunchbase funding",
            f"{startup_name} product hunt launch",
            f"{startup_name} github repository",
            f"{startup_name} linkedin founders team"
        ]

    return queries[:n]

def categorize_result(result: dict, startup_name: str) -> str:
    """
    Categorize a search result based on its content and URL.
    """
    url = result.get("link", "").lower()
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    
    if "crunchbase.com" in url or "angellist.co" in url or "funding" in title or "raised" in snippet:
        return "funding"
    elif "producthunt.com" in url or "launch" in title:
        return "traction"
    elif "github.com" in url or "tech stack" in title or "repository" in snippet:
        return "tech"
    elif "linkedin.com" in url or "founder" in title or "team" in title:
        return "team"
    elif startup_name.lower() in url:
        return "company_website"
    else:
        return "other"

def scrape_startup_data(startup_name: str, queries: list[str], results_per_query: int = 10) -> dict:
    """
    Run each query and collect all unique search results, organized by category.
    """
    all_results = {
        "company_website": [],
        "funding": [],
        "traction": [],
        "tech": [],
        "team": [],
        "other": []
    }
    seen_links = set()
    
    for query in queries:
        params = {
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "q": query,
            "num": results_per_query,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en"
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
                
                # Categorize and store the result
                category = categorize_result(result, startup_name)
                all_results[category].append(result)
    
    # Limit each category to MAX_RESULTS
    for category in all_results:
        all_results[category] = all_results[category][:MAX_RESULTS]
    
    return all_results

def generate_startup_profile(startup_name: str, categorized_data: dict) -> dict:
    """
    Use OpenAI to generate a structured startup profile from the gathered data.
    """
    # Prepare the data for the prompt
    data_summary = []
    for category, results in categorized_data.items():
        if results:
            data_summary.append(f"{category.upper()} ({len(results)} sources):")
            for i, result in enumerate(results[:3]):  # Include only top 3 for the prompt
                data_summary.append(f"- {result['title']}: {result['snippet'][:100]}...")
    
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
        f"9. sources: List of the most important URLs to reference\n\n"
        f"Return only valid JSON."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You create structured startup profiles from research data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2,
    )
    
    try:
        profile_text = resp.choices[0].message.content or "{}"
        # Extract JSON if it's wrapped in markdown code blocks
        if "```json" in profile_text:
            profile_text = profile_text.split("```json")[1].split("```")[0].strip()
        elif "```" in profile_text:
            profile_text = profile_text.split("```")[1].split("```")[0].strip()
        
        profile = json.loads(profile_text)
        # Add raw data sources
        profile["raw_data"] = categorized_data
        return profile
    except json.JSONDecodeError:
        # Fallback when JSON parsing fails
        return {
            "company_name": startup_name,
            "summary": "Profile generation failed. Please check raw data.",
            "raw_data": categorized_data
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

def main(startup_name: str = None):
    # Get startup name from command line or use default
    if not startup_name:
        startup_name = input("Enter startup name: ")
    
    print(f"Gathering intelligence on: {startup_name}")
    
    # Generate search queries
    queries = generate_startup_search_queries(startup_name)
    print(f"Generated {len(queries)} search queries")
    
    # Scrape and categorize data
    categorized_data = scrape_startup_data(startup_name, queries)
    total_results = sum(len(results) for results in categorized_data.values())
    print(f"Found {total_results} unique results across {len(categorized_data)} categories")
    
    # Generate structured profile
    profile = generate_startup_profile(startup_name, categorized_data)
    print(f"Generated startup profile for {profile.get('company_name', startup_name)}")
    
    # Save profile
    output_path = save_startup_profile(profile)
    print(f"Process complete. Profile saved to {output_path}")

if __name__ == "__main__":
    main()
