from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sys
import os
import uvicorn
from fastapi.responses import JSONResponse
import json
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Assuming HttpTriggerRAG is a module in the same directory, import it correctly
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
from dotenv import load_dotenv
load_dotenv()

tvly_api_key = os.getenv("tvly_api_key")
# Importing the custom function to get OpenAI response

# --------- plugin for web search -------------
class SearchSkillPlugin:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "xxxx"

    @kernel_function(name="search_resources")
    async def search(self, query: str) -> str:
        try:
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

# --------- plugin for cv skill -------------
 
class cvSkillsPlugin:
    def __init__(self):
        self.local_function_url = "http://localhost"

    @kernel_function()
    async def get_skills(self, user_input: str, conversation_id : str) -> str:
        try:
           print(f"Agent get_skills: {user_input}")
           payload = {"Question": user_input, "conversation_id": conversation_id}
           async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self.local_function_url,
                    json=payload,
                    timeout=30.0
                )
           if response.status_code != 200:
                return f"[❌] Local function call failed: {response.status_code} - {response.text}"

           data = response.json()
           answer = data.get("answer", "Sorry, no answer returned.")
           conversation_id = data.get("conversation_id", "unknown") 
           print(f"Output Answer: {answer} topic: {conversation_id}")
           return answer
        except Exception as e:
            return f"[🔥] Error calling local function app: {str(e)}"
        
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
    kernel = get_azure_kernel()
    cv_plugin = cvSkillsPlugin()
    search_plugin = SearchSkillPlugin(api_key=tvly_api_key)

    #ßskills = await cv_plugin.get_skills_with_context(user_input, conversation_id)

    agent = ChatCompletionAgent(
        service=kernel.get_service("azure-chat"),
        name="azure-agent",
        instructions=(
            "You are a concise and structured career assistant.\n"
            "Step 1: Call 'get_skills(user_input, conversation_id)' with the full user message as `user_input to retrieve the user's current skill summary from their CV. "
            "Return this exactly as received, as the **first part** of your final response.\n"
            "Step 2: Call 'search_resources(skills)' to retrieve skill improvement suggestions.\n"
            "Step 3: Combine both pieces into one response with this structure:\n\n"
            "✅ CV Summary:\n<output from get_skills>\n\n"
            "✅ Skills to Improve:\n<bullet list from search_resources>\n\n"
            "Do not change or paraphrase the CV summary. Do not include user names or roles unless explicitly mentioned in the output. Do not add any introductions like 'Based on the CV'."
        ),
        plugins=[cv_plugin, search_plugin]
    )
    combined_input = f"{user_input}\n\n[conversation_id: {conversation_id}]"
    response = await agent.get_response(combined_input)
    print(f"Agent response: {response}")
    return response.content

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
    return answer

if __name__ == "__main__":
    uvicorn.run("rag_agent:app", host="0.0.0.0", port=7080, reload=True)
