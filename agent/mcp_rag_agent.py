from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sys
import os
import uvicorn
from fastapi.responses import JSONResponse
import json, re
import asyncio
import redis
from semantic_kernel.connectors.mcp import MCPSsePlugin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib
import json
# Assuming HttpTriggerRAG is a module in the same directory, import it correctly
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
from agent import agtcache_memory


tvly_api_key = os.getenv("tvly_api_key")

api_version = os.getenv("api_version")
AZURE_OPENAI_ENDPOINT = ""    # e.g., https://your-resource.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = "gpt-35-turbo"   # This is the name of your GPT deployment in Azure


# --------- plugin for refine query plugin -------------
class QueryRefinerPlugin:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("api_version"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = "gpt-35-turbo"  # Use your deployed model name

    @kernel_function(name="refine_query", description="Cleans and reframes user input using LLM")
    async def refine_query(self, query: str) -> str:
        try:
            prompt = f"""
            You are a smart assistant that refines casual or vague user queries into clear, structured intent statements.
            Rephrase the input for clarity and remove filler words.

            Original Query: "{query.strip()}"
            Refined Intent:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query refiner that restructures raw user questions into clean, intent-focused queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=60
            )

            refined_query = response.choices[0].message.content.strip()
            print(f"\n[QueryRefinerPlugin] Refined: {refined_query}")
            return refined_query

        except Exception as e:
            fallback = f"Refined intent from user: {query.strip().capitalize()}."
            print(f"[QueryRefinerPlugin] ⚠️ Fallback due to error: {e}")
            return fallback


# --------- mcp_plugin for cv skill -------------
async def load_mcp_plugin(user_input, conversation_id):
    plugin = MCPSsePlugin(
        name="get_skills",
        description="Fetch CV skills using RAG",
        url="http://localhost:8080/sse/"
    )
    #  Connect to the FastMCP server
    await plugin.connect() 
    # Load available tools      
    await plugin.load_tools()    
    return plugin  


# --------- plugin for web search -------------
class SearchSkillPlugin:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    @kernel_function(name="search_resources")
    async def search(self, query: str) -> str:
        try:
            print(f"Agent search_resources: {query}")
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",   # you can also use "basic"
                "include_answer": True,
                "include_images": False
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(self.endpoint, headers=headers, json=payload, timeout=20.0)

            if response.status_code != 200:
                return f"[❌ Tavily Error] {response.status_code} - {response.text}"

            data = response.json()
            answer = data.get("answer", "")
            citations = "\n".join([item.get("url", "") for item in data.get("results", [])[:5]])
            return f"{answer}\n\nTop Sources:\n{citations}"

        except Exception as e:
            return f"[🔥 Tavily Plugin Error] {str(e)}"

# --------------- Semantic Kernel Setup ----   
 
def get_azure_kernel():
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            deployment_name="gpt-35-turbo",  # Use your actual deployment name,
            api_key="",
            endpoint="",
            service_id="azure-chat"
        )
    )
    return kernel

# --------- run agent -------------
async def run_sk_agent(user_input, conversation_id):

     # --- Cache Check ---
    answer_cache_key = f"answer:{conversation_id}:{user_input}"
    cached_answer = agtcache_memory.get(answer_cache_key)
    if cached_answer:
        print(f"Output Cached Answer: {cached_answer}")
        return cached_answer    

    kernel = get_azure_kernel()
    refine_plugin = QueryRefinerPlugin()
    # 🔥 Load FastMCP plugin over SSE
    mcp_plugin = await load_mcp_plugin(user_input, conversation_id)
    search_plugin = SearchSkillPlugin(api_key=tvly_api_key)
    
    agent = ChatCompletionAgent(
        service=kernel.get_service("azure-chat"),
        name="azure-agent",
        instructions=(
            "You are a concise and structured career assistant.\n"
            "Step 1: Call refine_query(user_input) to clean and normalize the user's query.\n"
            "Step 2: Call search_resources(refined_query) to compare the user's current skill and suggest additional skills required or any skills lacks. also recommend any skillset improvement based on user retrieved message from get_skills.\n"
            "Step 3: for this refined_query and with the retrieved message from get_skills generate the optimal response.\n\n"
            "Do not change or paraphrasthe CV summary. Do not include user names or roles unless explicitly mentioned in the output. Do not add any introductions like 'Based on the CV'."
        ),
        plugins=[refine_plugin, mcp_plugin, search_plugin]   
    )

    combined_input = f"[user_input:{user_input}]\n\n[conversation_id:{conversation_id}]"
    response = await agent.get_response(combined_input)
    agent_response = str(response.content).strip()  # Ensures it's string, not custom object

    agtcache_memory.append(conversation_id, "user", user_input)
    agtcache_memory.append(conversation_id, "assistant", agent_response)
    agtcache_memory.set(answer_cache_key, agent_response)

    print(f"Output agent_response: {agent_response}")

    return agent_response

# ---- FastAPI App ----
app = FastAPI()

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    Question: str
    conversation_id: str = None

@app.post("/api/agtchat")
async def chat(req: ChatRequest):
    answer = await run_sk_agent(req.Question, req.conversation_id)
    return JSONResponse(content={"answer": answer})

if __name__ == "__main__":
    uvicorn.run("mcp_rag_agent:app", host="0.0.0.0", port=7081, reload=True)