import textwrap
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util

api_type = "azure"
api_version = "2024-12-01-preview"
api_base = ""    # e.g., https://your-resource.openai.azure.com
api_key = ""
DEPLOYMENT_NAME = "gpt-35-turbo"   # This is the name of your GPT deployment in Azure
EMBEDDING_ENGINE = "text-embedding-ada-002"


grader_prompt = textwrap.dedent("""
You are a strict evaluator.
Given:
<answer>{answer}</answer>
<context>{context}</context>

Is the answer fully supported by the context?  
Respond ONLY "yes" or "no".
""")


def grounded(answer, context):
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        api_version=api_version
      )
    g = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": grader_prompt.format(
            answer=answer, context="\n".join(context))}],
            temperature=0.7)   
       
    answer = g.choices[0].message.content.strip()

    return answer

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_response(generated, reference):
    emb1 = model.encode(generated, convert_to_tensor=True)
    emb2 = model.encode(reference, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()