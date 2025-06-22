def score_keywords(text: str, keywords: list[str]) -> float:
    """
    Compute the fraction of keywords present in the text.
    Returns a score between 0.0 and 1.0.
    """
    text_l = text.lower()
    total = len(keywords)
    if total == 0:
        return 0.0
    hits = sum(1 for k in keywords if k.lower() in text_l)
    return hits / total
