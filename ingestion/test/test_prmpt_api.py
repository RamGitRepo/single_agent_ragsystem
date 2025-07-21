import pytest
import requests
import json
from sentence_transformers import SentenceTransformer, util

# Load model for evaluation
model = SentenceTransformer("all-MiniLM-L6-v2")

# API base URL

# Cosine similarity evaluator
def get_similarity(predicted, expected):
    emb1 = model.encode(predicted, convert_to_tensor=True)
    emb2 = model.encode(expected, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

# Load test cases from JSON
def load_test_cases():
    with open("data/prompt-variant.json", "r") as f:
        return json.load(f)

# Parameterize tests dynamically
@pytest.mark.parametrize("case", load_test_cases())
def test_prmpt_api(case):
    prompt = case["prompt"]
    expected = case["expected"]

    # Make the API request
    
    response = requests.post("http://localhost:7071/api/chat", json={"Question": prompt, "conversation_id": 1})
    assert response.status_code == 200, f"API failed: {response.text}"

    generated = response.json().get("response", "")
    similarity = get_similarity(generated, expected)
    

    # Debug info
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print(f"Similarity: {similarity:.2f}")

    # Validate similarity score (e.g., 0.8 threshold)
    assert similarity > 0.8, f"Low similarity score: {similarity:.2f}"
