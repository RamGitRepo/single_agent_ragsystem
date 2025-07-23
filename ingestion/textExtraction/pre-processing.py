import os, json, fitz, io
import logging
import azure.functions as func
from keybert import KeyBERT
from langdetect import detect
from nltk.tokenize import sent_tokenize
import spacy, re, ast
import nltk
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import AzureOpenAI
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

api_type = "azure"
api_version = "2024-12-01-preview"
api_base = ""    # e.g., https://your-resource.openai.azure.com
api_key = ""
DEPLOYMENT_NAME = "gpt-35-turbo"   # This is the name of your GPT deployment in Azure
EMBEDDING_ENGINE = "text-embedding-ada-002"
AZURE_SEARCH_ENDPOINT = ""
AZURE_SEARCH_INDEX = "rag-index"
AZURE_SEARCH_KEY = ""

container_name = "contractdata"  # Replace with your actual container name
# Setup OpenTelemetry Tracer
trace.set_tracer_provider(
    TracerProvider(
        resource=Resource.create({
            SERVICE_NAME: "functionapp-hybrid-retrieval"
        })
    )
)

AI_CONNECTION_STRING = ()

exporter = AzureMonitorTraceExporter.from_connection_string(AI_CONNECTION_STRING)


span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)


# Load NLP models globally (for performance)
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

def extract_text_from_pdf(pdf_bytes):
    TEXT_THRESHOLD = 50

    # ✅ Validate input bytes
    if not pdf_bytes or not isinstance(pdf_bytes, (bytes, bytearray)):
        logging.error("❌ Invalid or empty PDF byte stream.")
        raise ValueError("Invalid or empty PDF byte stream")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logging.error(f"❌ Failed to open PDF stream with fitz: {e}")
        raise

    # Extract text from each page
    text = "\n".join([page.get_text() for page in doc])
    text = text.replace("\n", " ").strip()

    if len(text.strip()) >= TEXT_THRESHOLD:
        logging.info(f"✅ Extracted text directly (length: {len(text)} characters)")
        return text

    else:
        logging.warning(f"ℹ️ Text length {len(text)} is below threshold. Attempting OCR.")
        try:
            images = [Image.open(io.BytesIO(page.get_pixmap().tobytes())) for page in doc]
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
            return text
        except Exception as e:
            logging.error(f"❌ OCR extraction failed: {e}")
            raise

# ---------------- context aware chunks  --------------------------------

def semantic_aware_chunks(text: str, max_tokens: int = 500) -> dict:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base
    )

    prompt = f"""
You are an intelligent assistant helping to preprocess documents.

Given the input text:
1. Split it into **coherent chunks** of about {max_tokens} tokens, preserving sentence integrity.
2. For each chunk, extract:
    - Key phrases (topic-specific nouns, concepts, etc.)
    - Named entities (persons, organizations, locations)

Return the output in this JSON-like format:

[
  {{
    "chunk": "...",
    "entities": ["Entity1", "Entity2"],
    "key_phrases": ["KeyPhrase1", "KeyPhrase2"]
  }},
  ...
]

Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048
        )

        content = response.choices[0].message.content.strip()

        # Try to parse as Python-like JSON
        # You could use `ast.literal_eval` or `json.loads` after some precleaning
        try:
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            json_str = content[json_start:json_end]
            parsed = ast.literal_eval(json_str)
            return parsed
        except Exception as parse_error:
            print("[⚠️] Failed to parse JSON structure from model response.")
            print(content)
            return {"raw_output": content, "error": str(parse_error)}

    except Exception as e:
        return {"error": str(e)}

    except Exception as e:
        print(f"[LLM Chunking Error] {e}")
        return []

# ---------------- context aware chunks  --------------------------------
#model = SentenceTransformer("all-MiniLM-L6-v2")
 #def semantic_aware_chunks(text, max_tokens=500, overlap_sentences=1, sim_threshold=0.6):
    with tracer.start_as_current_span("documentChunking") as span:
         sentences = re.split(r'(?<=[.?!])\s+', text.strip())
         if not sentences:
            return []
         chunks = []
         current_chunk = [sentences[0]]

         for i in range(1, len(sentences)):
            sim = util.cos_sim(model.encode(sentences[i - 1]), model.encode(sentences[i]))[0][0]

            if sim < sim_threshold or len(" ".join(current_chunk).split()) > max_tokens:
               chunks.append(" ".join(current_chunk))
               current_chunk = sentences[max(i - overlap_sentences, 0):i+1]
            else:
               current_chunk.append(sentences[i])      

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

def extract_entities(text):
    with tracer.start_as_current_span("extractEntities") as span:
      doc = nlp(text)
      return [(ent.text, ent.label_) for ent in doc.ents]

def extract_keywords(text, top_n=5):
    with tracer.start_as_current_span("extractEntities") as span:
        return kw_model.extract_keywords(text, top_n=top_n)

def upload_to_blob_storage(container_name, blob_name, data, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Convert Python object (e.g., list of chunks) to JSON string
    chunk_json_data = json.dumps(data, indent=2)
    
    print(f"Uploading {chunk_json_data} chunks to blob: {blob_name}")
    # Upload JSON as blob
    blob_client.upload_blob(chunk_json_data, overwrite=True, content_settings=ContentSettings(content_type="application/json"))

# embed chunks and upload to Azure Search
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version
   )

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def sanitize_id(blob_name, chunk_index):
    # Option 1: Replace `/` and `.` with `_`
    safe_id = re.sub(r"[^a-zA-Z0-9_\-=]", "_", f"{blob_name}_{chunk_index}")
    return safe_id

def flatten_to_string(nested_list):
    return ", ".join(
        item[0] for item in nested_list if isinstance(item, list) and len(item) > 0
    )   

# retrieve text from blob storage
def retrieve_text_from_blob(container_name, blob_name, connection_string):
    with tracer.start_as_current_span("retrieveBlob") as span:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Step 2: Get ContainerClient
        container_client = blob_service_client.get_container_client(container_name)
        for blob in container_client.list_blobs():
            # Step 3: Download each blob
            if not blob_name.endswith(".json"):
               continue
            
            print(f"blob name: {blob_name}")
            blob_client1 = container_client.get_blob_client(blob=blob_name)
            try:
            # Try reading and decoding the blob as UTF-8
                content_bytes = blob_client1.download_blob().readall()
                content_str = content_bytes  # may fail here
                print(f"content_str: {content_str}")
                chunks = json.loads(content_str)
            except UnicodeDecodeError:
                print(f"❌ Skipping non-UTF-8 blob: {blob.name}")
                continue
            except json.JSONDecodeError:
                print(f"❌ Skipping invalid JSON blob: {blob.name}")
                continue
        
            docs_to_upload = []

            for i, chunk_data in enumerate(chunks):
                entities = flatten_to_string(chunk_data.get("entities", []))
                key_phrases = flatten_to_string(chunk_data.get("key_phrases", []))

                doc_id = sanitize_id(blob_name, i)  
                chunk_text = chunk_data["chunk"]
                response = client.embeddings.create(
                model=EMBEDDING_ENGINE,
                input=chunk_text)
                embedding = response.data[0].embedding

                if not isinstance(embedding[0], float):
                    embedding = embedding[1:]

                if len(embedding) != 1536:
                    raise ValueError("Embedding length mismatch")

                if not all(isinstance(x, float) for x in embedding):
                    raise TypeError("Embedding must contain only floats")
            
                doc = {
                "id": doc_id,
                "source": blob_name,
                "chunk_index": i,
                "entities": entities,
                "key_phrases": key_phrases,
                "contentVector": embedding  # your vector field in index
                }

                docs_to_upload.append(doc)

                # 5. Upload batch to Azure Search
                result = search_client.upload_documents(documents=docs_to_upload)
                for doc in result:
                  if not doc.succeeded:
                    print(f"❌ Upload failed for ID: {doc.key} — {doc.error_message}")
                  else:
                     print(f"✅ Uploaded: {doc.key}")
                return result


def main(blob: func.InputStream):
    with tracer.start_as_current_span("documentCracking") as span:
       

         pdf_bytes = blob.read()
         if not pdf_bytes:
             logging.error("❌ Blob is empty or unreadable.")
             raise ValueError("Blob is empty or unreadable")
    
         text = extract_text_from_pdf(pdf_bytes)
         print(f"Extracted >>> {text}")

        #language = detect(text)
         chunk_data = semantic_aware_chunks(text)
         # Validate format
         if isinstance(chunk_data, dict) and "error" in chunk_data:
             logging.error(f"❌ Chunking failed: {chunk_data['error']}")
             return
         if not isinstance(chunk_data, list):
            logging.error(f"❌ Unexpected LLM response format: {type(chunk_data)} — {chunk_data}")
            return 
         
         result = []
         for i, item in enumerate(chunk_data):
            if not isinstance(item, dict) or "chunk" not in item:
               logging.warning(f"⚠️ Skipping invalid item at index {i}: {item}")
               continue

            result.append({
                "chunk": item["chunk"],
                "entities": item.get("entities", []),
                "key_phrases": item.get("key_phrases", [])
                })
            
         print(f"✅ Extracted {len(result)} chunks with entities and key phrases from {blob.name}")

         output_blob_name = f"processed/{Path(blob.name).stem}.json"
         print(f"Uploading processed chunks to blob storage: {output_blob_name}")

         upload_to_blob_storage(
                container_name=container_name,
                blob_name=output_blob_name,
                data=result,
                connection_string=""
            )
 
       # Upload chunks to Azure Search
         result=retrieve_text_from_blob(container_name, blob_name=output_blob_name, connection_string=""
       )
         print(f"✅ Processed and uploaded {len(result)} chunks to Azure Search from {output_blob_name}")


