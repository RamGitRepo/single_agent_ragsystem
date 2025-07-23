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
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
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
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler


logger = logging.getLogger(__name__)
logger.addHandler(
    AzureLogHandler()
)
logger.setLevel(logging.INFO)

logger.info("This is a log from your RAG app!")

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
    logging.info(f"Generating embedding for input: {question}")

    try:
        embedding_response = openai_client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        )
        embedding = embedding_response.data[0].embedding
        logging.info(f"Successfully generated embedding of length {len(embedding)}")
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}", exc_info=True)
        return None
    

def call_llm(prompt, start):
    try:
        # Correct latency calculation: convert to milliseconds properly
        latency_ms = (time.time() - start) * 1000
        logging.info(f"Calling LLM with prompt: {prompt}")
        

        # Call to OpenAI or Azure OpenAI
        completion = openai_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful structured career assistant, Retrieve the user's current skill summary from their CV"},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content
        logging.info(f"LLM Call Time since start: {latency_ms:.2f} ms")
        return response

    except Exception as e:
        logging.error(f"Error while calling LLM: {e}", exc_info=True)
        return "❌ Failed to get response from LLM."

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        start = time.time()
        body = req.get_json()
        question = body.get("Question")
        conversation_id = body.get("conversation_id")

        if not question:
            return func.HttpResponse("Missing 'question' in request.", status_code=400)

        logging.info(f"Received question: '{question}' | Conversation ID: {conversation_id}")

        # Step 1: Get embedding
        embedding = get_embedding(question)

        # Step 2: Query vector search
        vector_query = {
            "vector": embedding,
            "fields": "contentVector",
            "k": 5,
            "kind": "vector"
        }

        results = search_client.search(search_text="", vector_queries=[vector_query])

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
                logging.debug(f"Cached new chunk: {chunk_key}")
            else:
                matched_chunk = cached_chunk

            relevant_chunks.append(matched_chunk)

        # Build system prompt
        system_prompt = (
            "You are a factual assistant. Use only the information provided in the context below to answer the question.\n"
            "If you believe the answer is not clearly supported, explain why and reply with :\n"
            "'The answer is not available in the provided context.'"
        )

        # Step 3: Prepare chat history
        chat_history = callCacheMemory.fetch(conversation_id, last_n=5)
        history_block = ("\n".join(f"{msg['role']}: {msg['text']}" for msg in chat_history) if chat_history else "No recent interactions.")

        # Step 4: Final prompt
        context = "\n\n".join(relevant_chunks)

        final_prompt = f"""{system_prompt}

        Memory:
        {history_block}

        Context:
        {context}

        Question: {question}
        """

        # Step 5: Check cache
        answer_cache_key = f"answer:{conversation_id}:{question}"
        cached_answer = callCacheMemory.get(answer_cache_key)

        if cached_answer:
            answer = cached_answer
            latency_ms = (time.time() - start) * 1000
            logging.info(f"Cache all Time since start: {latency_ms:.2f} ms")
            logging.info(f"Used cached answer for conversation {conversation_id}")
        else:
            answer = call_llm(final_prompt, start)
            callCacheMemory.set(answer_cache_key, answer)
            logging.info(f"LLM generated answer: {answer}")

        # Step 6: Update memory
        callCacheMemory.append(conversation_id, "user", question)
        callCacheMemory.append(conversation_id, "assistant", answer)

        # Step 7: Return response
        return func.HttpResponse(
            json.dumps({
                "answer": answer or "No answer generated.",
                "conversation_id": conversation_id or "unknown",
            }),
            status_code=200,
            mimetype="application/json"
        )

    except json.JSONDecodeError:
        logging.error("Invalid JSON in request body.")
        return func.HttpResponse("Invalid JSON in request body.", status_code=400)

    except TypeError as e:
        logging.exception("Type error in request.")
        return func.HttpResponse("Type error in request.", status_code=500)

    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return func.HttpResponse(
            f"Unexpected error: {str(e)}",
            status_code=500
        )
    