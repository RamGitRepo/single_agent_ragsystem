# Run it as a standalone test
import asyncio, os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from rag_agent import SearchSkillPlugin
load_dotenv()



tvly_api_key = os.getenv("tvly_api_key")

async def test_tavily():
    plugin = SearchSkillPlugin(api_key=tvly_api_key)
    result = await plugin.search("AI career skills for a data engineer")
    print(result)

asyncio.run(test_tavily())
