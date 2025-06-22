# process/relevance_scorer.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
import json
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

from process.keyword_score import score_keywords

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Get an embedding vector for the given text using OpenAI embeddings.
    """
    resp = openai.embeddings.create(
        model=model,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_scores(texts: list[str], keywords: list[str], context: str):
    """
    For each text:
      - Compute keyword match score
      - Compute semantic similarity to the context
      - Combine into a final relevance score
    """
    # Pre-compute embedding for context
    context_embedding = get_embedding(context)

    results = []
    for i, text in enumerate(texts):
        try:
            # 1) Keyword-based score (0–1)
            kw_score = score_keywords(text, keywords)

            # 2) Semantic score (cosine similarity)
            snippet = text[:2000]
            doc_embedding = get_embedding(snippet)
            sem_score = cosine_similarity(context_embedding, doc_embedding)

            # 3) Combine scores
            final_score = 0.6 * kw_score + 0.4 * sem_score

            results.append({
                "text_index": i,
                "keyword_score":   round(kw_score, 3),
                "semantic_score":  round(sem_score, 3),
                "final_score":     round(final_score, 3)
            })
        except Exception as e:
            results.append({
                "text_index": i,
                "error": str(e)
            })

    return results

def main():
    # Example usage with sample data
    sample_texts = [
        "This is a sample text about AI and machine learning.",
        "Another text about funding and venture capital.",
        "A third text about technology and software development."
    ]
    
    keywords = ["ai", "funding", "technology"]
    context = "Startup intelligence and venture capital"

    # Compute relevance scores
    results = compute_scores(sample_texts, keywords, context)

    # Print to console
    print("\nRelevance Scores:")
    for r in results:
        if "final_score" in r:
            print(f"Text {r['text_index']} → kw:{r['keyword_score']}  sem:{r['semantic_score']}  final:{r['final_score']}")
        else:
            print(f"Text {r['text_index']} → ERROR: {r['error']}")

if __name__ == "__main__":
    main()
