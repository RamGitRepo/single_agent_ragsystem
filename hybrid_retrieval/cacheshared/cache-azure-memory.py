from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import openai
import time
from datetime import datetime

search = SearchClient(
    endpoint   = "https://<SEARCH-NAME>.search.windows.net",
    index_name = "rag-memory",
    credential = AzureKeyCredential("<ADMIN-KEY>")
)

def write_turn(conversation_id, role, text):
    embedding = AzureOpenAI(
        api_key       = "<OPENAI-KEY>",
        azure_endpoint= "https://<OPENAI-ENDPOINT>.openai.azure.com",
    ).embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    ).data[0].embedding
    
    search.upload_documents([{
        "id": f"{conversation_id}-{role}-{int(time.time()*1000)}",
        "conversationId": conversation_id,
        "role": role,
        "content": text,
        "contentVector": embedding,
        "timestamp": datetime.utcnow().isoformat()
    }])

def fetch_memory(conversation_id, last_user_question, k=4):
     emb = openai.embeddings.create(
        input=[last_user_question],
        model="text-embedding-ada-002"
    ).data[0].embedding
    
     results = search.search(
        search_text="",             # empty ⇒ vector-only
        vectors=[
          { "value": emb, "fields": "contentVector", "k": k }
        ],
        filter=f"conversationId eq '{conversation_id}'"
    )
     return [r["content"] for r in results]

