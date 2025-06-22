import os
import serpapi 
from dotenv import load_dotenv


load_dotenv()  # this reads .env in the cwd and sets os.environ

# 1. Inspect the module
print("serpapi module path:", serpapi.__file__)
print("Available names in serpapi:", dir(serpapi))

# 2. Try imports
GoogleSearch = None
try:
    from serpapi import GoogleSearch
    GoogleSearch = GoogleSearch
    print("Imported GoogleSearch from serpapi")
except ImportError as e:
    print("Cannot import GoogleSearch:", e)
    try:
        from serpapi.GoogleSearch import GoogleSearchResults as GoogleSearch
        print("Imported GoogleSearchResults as GoogleSearch")
    except Exception as e2:
        print("Cannot import GoogleSearchResults:", e2)

# 3. If import succeeded, run a basic search
if GoogleSearch:
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("Please set SERPAPI_KEY in your environment.")
    else:
        params = {"q": "coffee", "api_key": api_key, "num": 5}
        try:
            results = GoogleSearch(params).get_dict()
            print("Search results keys:", list(results.keys()))
            print("First result snippet:", results.get("organic_results", [{}])[0].get("snippet"))
        except Exception as e:
            print("Error during GoogleSearch:", e)
else:
    print("No valid search class available; check your serpapi installation.")