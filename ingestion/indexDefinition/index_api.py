import requests, os
from dotenv import load_dotenv
load_dotenv()


AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX = os.getenv("INDEX_NAME")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")


index_schema = {
    "name": AZURE_SEARCH_INDEX,
    "fields": [
        {"name": "id", "type": "Edm.String", "key": True},
        {"name": "source", "type": "Edm.String", "filterable": True},
        {"name": "chunk_index", "type": "Edm.Int32", "filterable": True},
        {"name": "entities", "type": "Edm.String", "searchable": True, "analyzer": "standard.lucene"},
        {"name": "key_phrases", "type": "Edm.String", "searchable": True, "analyzer": "standard.lucene"},
        {
            "name": "contentVector",
            "type": "Collection(Edm.Single)",
            "dimensions": 1536,
            "retrievable": True,
            "vectorSearchProfile": "my-profile"
        }
    ],
    "vectorSearch": {
        "profiles": [
            {
                "name": "my-profile",
                "algorithm": "my-hnsw"
            }
        ],
        "algorithms": [
            {
                "name": "my-hnsw",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 4,
                    "efConstruction": 400
                }
            }
        ]
    }
}

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_SEARCH_KEY
}

url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}?api-version=2024-05-01-preview"
response = requests.put(url, headers=headers, json=index_schema)

if response.status_code in [200, 201, 204]:
    print(f"✅ Index '{AZURE_SEARCH_INDEX}' created or updated successfully.")
else:
    print(f"❌ Failed to create index: {response.status_code}")
    print(response.text)
