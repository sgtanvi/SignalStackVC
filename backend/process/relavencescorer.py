# process/relevance_scorer.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
import json
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# from process.pdf_text_extractor import extract_text_with_fallback_url
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

def compute_scores(urls: list[str], keywords: list[str], context: str):
    """
    For each PDF URL:
      - Extract text via Mistral OCR fallback
      - Compute keyword match score
      - Compute semantic similarity to the context
      - Combine into a final relevance score
    """
    # Pre-compute embedding for context
    context_embedding = get_embedding(context)

    results = []
    for url in urls:
        try:
            # 1) Download & extract text
            text = extract_text_with_fallback_url(url)

            # 2) Keyword-based score (0–1)
            kw_score = score_keywords(text, keywords)

            # 3) Semantic score (cosine similarity)
            snippet = text[:2000]
            doc_embedding = get_embedding(snippet)
            sem_score = cosine_similarity(context_embedding, doc_embedding)

            # 4) Combine scores
            final_score = 0.6 * kw_score + 0.4 * sem_score

            results.append({
                "url": url,
                "keyword_score":   round(kw_score, 3),
                "semantic_score":  round(sem_score, 3),
                "final_score":     round(final_score, 3)
            })
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e)
            })

    return results

def main():
    # 1) Load URLs from CSV (limit to first 10 for speed)
    urls = []
    with open("data/pdf_links.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= 10:  # Only process first 10 PDFs for speed
                break
            urls.append(row["url"])

    # 2) Load user payload
    with open("data/user_payload.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    keywords = [k.strip().lower() for k in payload.get("input_labels", [])]
    context  = payload.get("ai_usecase", "")

    # 3) Compute relevance scores
    results = compute_scores(urls, keywords, context)

    # 4) Print to console
    print("\nRelevance Scores:")
    for r in results:
        if "final_score" in r:
            print(f"{r['url']} → kw:{r['keyword_score']}  sem:{r['semantic_score']}  final:{r['final_score']}")
        else:
            print(f"{r['url']} → ERROR: {r['error']}")

    # 5) Save to CSV
    os.makedirs("data", exist_ok=True)
    out_path = "data/pdf_scores.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["url", "keyword_score", "semantic_score", "final_score", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "url":            r.get("url", ""),
                "keyword_score":  r.get("keyword_score", ""),
                "semantic_score": r.get("semantic_score", ""),
                "final_score":    r.get("final_score", ""),
                "error":          r.get("error", "")
            })
    print(f"\nSaved detailed scores to {out_path}")

if __name__ == "__main__":
    main()
