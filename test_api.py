"""
Test script to verify Semantic Scholar API access.
"""

import requests

api_key = "0G8y90GfQIaYaqoxFYPFH5kQFkH75un23fvs0hIx"

# Test 1: Simple search with small limit
print("Test 1: Searching with limit=10...")
url = "https://api.semanticscholar.org/graph/v1/paper/search"
params = {
    "query": "deep learning",
    "limit": 10,
    "fields": "title,abstract,year,citationCount"
}
headers = {
    "x-api-key": api_key
}

response = requests.get(url, params=params, headers=headers)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:500]}")

if response.status_code == 200:
    data = response.json()
    print(f"\nSuccess! Found {len(data.get('data', []))} papers")
    if data.get('data'):
        print(f"First paper: {data['data'][0].get('title')}")
else:
    print(f"\nError: {response.status_code}")
    print(f"Headers sent: {headers}")
