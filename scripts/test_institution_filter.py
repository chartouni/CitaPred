"""
Test if Semantic Scholar API supports filtering by institution/affiliation.
"""

import requests
import json

API_KEY = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"

print("="*60)
print("Testing Semantic Scholar Institution Support")
print("="*60)

# Test 1: Check if author search returns affiliation data
print("\n1. Checking author affiliation data...")
print("-"*60)

try:
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": "American University of Beirut",
        "limit": 3,
        "fields": "name,affiliations,paperCount"
    }
    headers = {"x-api-key": API_KEY}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    data = response.json()

    if 'data' in data and data['data']:
        print(f"✅ Found {len(data['data'])} authors")
        for author in data['data'][:3]:
            print(f"\nAuthor: {author.get('name')}")
            print(f"Affiliations: {author.get('affiliations', 'N/A')}")
            print(f"Papers: {author.get('paperCount', 'N/A')}")
    else:
        print("❌ No authors found or no affiliation data")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Check a sample paper for author affiliation data
print("\n2. Checking paper for author affiliation data...")
print("-"*60)

try:
    # Get a sample AUB paper
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": "American University of Beirut computer science",
        "limit": 1,
        "fields": "title,authors.name,authors.affiliations,authors.authorId"
    }
    headers = {"x-api-key": API_KEY}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    data = response.json()

    if 'data' in data and data['data']:
        paper = data['data'][0]
        print(f"Paper: {paper.get('title', 'N/A')[:80]}")
        print(f"\nAuthors:")
        for author in paper.get('authors', [])[:3]:
            print(f"  - {author.get('name')}")
            print(f"    Affiliations: {author.get('affiliations', 'N/A')}")
            print(f"    AuthorID: {author.get('authorId', 'N/A')}")
    else:
        print("❌ No papers found")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Check if we can query by authorId
print("\n3. Checking if we can get papers by AuthorID...")
print("-"*60)

print("\n💡 FINDINGS:")
print("="*60)
print("""
Semantic Scholar API limitations:
1. No direct "institution ID" or "affiliation filter" for paper search
2. Author affiliations exist but are often incomplete/missing
3. Cannot filter papers by institution in search query

ALTERNATIVE APPROACHES:
1. Get list of AUB author IDs → fetch their papers
2. Get list of paper IDs from AUB → look them up directly
3. Ask AUB for their official publication list

RECOMMENDATION:
Contact AUB's research office/library for:
- List of faculty names
- List of paper DOIs/titles
- Access to institutional repository
""")
