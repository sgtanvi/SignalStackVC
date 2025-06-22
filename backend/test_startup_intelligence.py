#!/usr/bin/env python3
"""
Simple test script for startup intelligence gathering.
Run this to test the startup profile generation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Add the parent directory to the path so we can import backend modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.scrap.query_final import main

if __name__ == "__main__":
    # Test with the original startup to confirm structure
    startup_name = "vly ai"  # Back to original test case
    print(f"Testing startup intelligence gathering for: {startup_name}")
    
    # Run the main function
    profile = main(startup_name=startup_name, query_mode="hybrid", enrich_content=False)
    
    if profile:
        print(f"\nSuccessfully generated profile for {startup_name}")
        print(f"Summary: {profile.get('summary', 'No summary available')}")
        print(f"Product: {profile.get('product', 'No product info available')}")
        print(f"Team: {profile.get('team', 'No team info available')}")
        print(f"Funding: {profile.get('funding', 'No funding info available')}")
        print(f"Sources: {len(profile.get('sources', []))} real URLs extracted")
        print(f"Confidence scores: {profile.get('confidence_scores', {})}")
        print(f"Category counts: {profile.get('category_counts', {})}")
    else:
        print(" Failed to generate profile") 