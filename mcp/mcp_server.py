from fastmcp import FastMCP
from fastapi.responses import JSONResponse
from fastapi import Request
import httpx
import uvicorn

mcp = FastMCP(name= "get_skills", host="0.0.0.0", port=8080)


# Register tool
@mcp.tool()
async def get_skills(user_input: str, conversation_id: str) -> str:
    try:
        print(f"get FastMCP: {user_input}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:7071/api/rag-query",
                json={"Question": user_input, "conversation_id": conversation_id}
            )
        response.raise_for_status()
        print(f"FastMCP Output: {response.json()}")
        return response.json().get("answer", "No answer returned.")
        
    except Exception as e:
        return f"[RAG Tool Error] {e}"

# Run server
if __name__ == "__main__":
    mcp.run(transport='sse')
