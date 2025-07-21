import os
import json, time 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorQuery
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
import azure.functions as func
from opentelemetry import trace
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from cacheshared import cache_redis as callCacheå
from cacheshared import cache_memory as callCacheMemory
from dotenv import load_dotenv
load_dotenv()

api_version = os.getenv("api_version")
AZURE_OPENAI_ENDPOINT = ""    # e.g., https://your-resource.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = "gpt-35-turbo"   # This is the name of your GPT deployment in Azure
EMBEDDING_ENGINE = os.getenv("EMBEDDING_ENGINE")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
INDEX_NAME = os.getenv("INDEX_NAME")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")

# Setup OpenTelemetry Tracer
provider = TracerProvider(
    resource=Resource.create({SERVICE_NAME: "functionapp-hybrid-retrieval", "cloud.role": "ram-app-role"})
)

trace.set_tracer_provider(provider)

AI_CONNECTION_STRING = ()

exporter = AzureMonitorTraceExporter.from_connection_string(AI_CONNECTION_STRING)
span_processor = BatchSpanProcessor(exporter)


trace.get_tracer_provider().add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)

# Set up logging
CHUNK_SIZE = os.getenv("CHUNK_SIZE")

BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")  # Default to 'contractdata' if not set
# Load configuration from environment variables
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")

# Instantiate clients
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))
blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=api_version,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


def get_embedding(question):
    with tracer.start_as_current_span("get_embedding") as span:
        span.set_attribute("input.question", question)
        embedding_response = openai_client.embeddings.create(
             input=question,
             model="text-embedding-ada-002"
        )
        trace.get_tracer_provider().force_flush()
        return embedding_response.data[0].embedding

def call_llm(prompt, start):
    with tracer.start_as_current_span("call_llm"):
        latency = time.time() - start * 1000
        completion = openai_client.chat.completions.create(
           model=AZURE_OPENAI_MODEL,
           messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ])
        return completion.choices[0].message.content

def main(req: func.HttpRequest) -> func.HttpResponse:
    with tracer.start_as_current_span("handle_rag_request") as span:
        try:
          start = time.time()
          body = req.get_json()
          question = body.get("Question")
          conversation_id = body.get("conversation_id")
          if not question: 
             return func.HttpResponse("Missing 'question' in request.", status_code=400)
          
          # Step 1: Get embedding
          embedding = get_embedding(question)
          
      
          with tracer.start_as_current_span("vector_search"):
          # Create a vector query
                vector_query = {
                "vector": embedding,
                "fields": "contentVector",
                "k": 5,  # Number of top results to return
                "kind": "vector"        
             }
          
          results = search_client.search(
                search_text="",
                vector_queries=[vector_query],
                filter=None
            )
          # You may need to serialize results to store them
          
  
          relevant_chunks = []    
          for result in results:
            blob_name = result.get("source")
            chunk_index = result.get("chunk_index")
            if blob_name is None or chunk_index is None:
                continue
            chunk_key = f"blob:{blob_name}:{chunk_index}"

            cached_chunk = callCacheMemory.get(chunk_key)

            if not cached_chunk:
                 blob_client = blob_service.get_blob_client(container=BLOB_CONTAINER, blob=blob_name)
                 content = blob_client.download_blob().readall()
                 chunks = json.loads(content)
                 matched_chunk = chunks[chunk_index]["chunk"]
                 callCacheMemory.set(chunk_key, matched_chunk)
            else:
                matched_chunk = cached_chunk
            
            relevant_chunks.append(matched_chunk)
            
            system_prompt = (
            "You are a factual assistant. Only use the information provided in the context below.\n"
            "If the answer is not explicitly stated in the context, reply with:\n"
            "'The answer is not available in the provided context.'\n"
            )

            # 1. Fetch recent memory for this conversation
            chat_history = callCacheMemory.fetch(conversation_id, last_n=5)

            # 2. Convert to usable format
            history_block = ""
            for msg in chat_history:
                history_block += f"{msg['role']}: {msg['text']}\n"

            
            context = "\n\n".join(relevant_chunks)
     
            # 3. Build the full prompt
            final_prompt = f"""{system_prompt}

            Memory:
            {history_block}

            Context:
            {context}

            Question: {question}
            """

            # Step 3: Send to LLM
            answer_cache_key = f"answer:{conversation_id}:{question}"
            cached_answer = callCacheMemory.get(answer_cache_key)   

            if cached_answer:
                answer = cached_answer
                print(f"Cached LLM Answer: {answer}, Conversation ID: {conversation_id}")
            else:
                answer = call_llm(final_prompt, start)
                callCacheMemory.set(answer_cache_key, answer)
                print(f"LLM Answer: {answer}, Conversation ID: {conversation_id}")
            

            # Step 4: Cache the conversation
            callCacheMemory.append(conversation_id, "user", question)
            callCacheMemory.append(conversation_id, "assistant", answer)
            
            # Evaluation
            safe_answer = answer if answer is not None else "No answer generated."
            safe_conversation_id = conversation_id if conversation_id is not None else "unknown"

            return func.HttpResponse(
                json.dumps({
                    "answer": safe_answer, "conversation_id": safe_conversation_id,}),
                    status_code=200, mimetype="application/json" )
            
        except json.JSONDecodeError:
            return func.HttpResponse("Invalid JSON in request body.", status_code=400)
        
        except TypeError as e:
            return func.HttpResponse("Invalid JSON in request body.",
                 status_code=500,)
        
        except Exception as e:
            return func.HttpResponse("Invalid JSON in request body.",
                 status_code=500,)
        
    return func.HttpResponse(
        json.dumps({"error": "Unknown internal error."}),
        status_code=500,
        mimetype="application/json"
    )
        


        
