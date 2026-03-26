"""
SciCommons Recommendation System — views.py
Simple TF-IDF based paper recommender for GSoC proposal.
"""

import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import UserSearch


PAPERS = [
    {
        "id": "arxiv:2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez"],
        "abstract": "Large pre-trained language models store factual knowledge in parameters. We present RAG, combining parametric and non-parametric memory for language generation by retrieving relevant documents.",
        "keywords": ["NLP", "retrieval", "language models"],
        "year": 2021,
        "citations": 3812,
        "venue": "NeurIPS",
        "url": "https://arxiv.org/abs/2005.11401",
    },
    {
        "id": "arxiv:2302.13971",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": ["Hugo Touvron", "Thibaut Lavril"],
        "abstract": "We introduce LLaMA, foundation language models from 7B to 65B parameters trained on public data, competitive with GPT-3 on most benchmarks.",
        "keywords": ["large language models", "open source", "foundation models"],
        "year": 2023,
        "citations": 6247,
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2302.13971",
    },
    {
        "id": "arxiv:2305.10601",
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "authors": ["Shunyu Yao", "Dian Yu"],
        "abstract": "Tree of Thoughts lets LLMs perform deliberate decision making by considering multiple reasoning paths and self-evaluating choices to decide the next course of action.",
        "keywords": ["reasoning", "language models", "problem solving"],
        "year": 2023,
        "citations": 2189,
        "venue": "NeurIPS",
        "url": "https://arxiv.org/abs/2305.10601",
    },
    {
        "id": "arxiv:2201.11903",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Jason Wei", "Xuezhi Wang"],
        "abstract": "Generating a chain of thought — intermediate reasoning steps — significantly improves LLM performance on arithmetic, commonsense, and symbolic reasoning tasks.",
        "keywords": ["chain of thought", "reasoning", "prompting"],
        "year": 2022,
        "citations": 7654,
        "venue": "NeurIPS",
        "url": "https://arxiv.org/abs/2201.11903",
    },
    {
        "id": "arxiv:2303.08774",
        "title": "GPT-4 Technical Report",
        "authors": ["OpenAI"],
        "abstract": "GPT-4 is a large-scale multimodal model accepting image and text inputs, exhibiting human-level performance on professional and academic benchmarks.",
        "keywords": ["GPT-4", "multimodal", "large language models"],
        "year": 2023,
        "citations": 9872,
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2303.08774",
    },
]

def _build_corpus_text(paper):
    return f"{paper['title']} {paper['abstract']} {' '.join(paper['keywords'])}".lower()


_corpus_texts = [_build_corpus_text(p) for p in PAPERS]
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
_corpus_matrix = _vectorizer.fit_transform(_corpus_texts)


def get_recommendations(query, top_k=5):
    # TF-IDF similarity
    query_vec = _vectorizer.transform([query.lower()])
    tfidf_scores = cosine_similarity(query_vec, _corpus_matrix).flatten()

    # Simple recency boost (normalized 0-1)
    years = np.array([p["year"] for p in PAPERS], dtype=float)
    recency = (years - years.min()) / (years.max() - years.min() + 1e-9)

    # Citation boost (log-normalized)
    citations = np.array([p["citations"] for p in PAPERS], dtype=float)
    citation_boost = np.log1p(citations) / np.log1p(citations.max())

    # Combined score: 75% relevance, 15% recency, 10% citations
    combined = 0.75 * tfidf_scores + 0.15 * recency + 0.10 * citation_boost

    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        paper = dict(PAPERS[idx])
        paper["rank"] = rank
        paper["relevance_score"] = round(float(tfidf_scores[idx]), 4)
        paper["combined_score"] = round(float(combined[idx]), 4)
        results.append(paper)

    return {
        "recommendations": results,
        "query_embeddings": query_vec.toarray()[0],
        "summary_embeddings_shape": _corpus_matrix.shape,
    }


@csrf_exempt
def server(request):
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    user_query = request.GET.get("query", "").strip()
    if not user_query:
        return JsonResponse({"error": "No query provided"}, status=400)

    top_k = int(request.GET.get("top_k", 5))

    try:
        saved_query = UserSearch.objects.create(query_text=user_query)
        result = get_recommendations(user_query, top_k=top_k)

        return JsonResponse({
            "status": "success",
            "saved_id": saved_query.id,
            "paper": result["recommendations"][0] if result["recommendations"] else {},
            "recommendations": result["recommendations"],
            "query_embeddings": result["query_embeddings"].tolist(),
            "summary_embeddings_shape": list(result["summary_embeddings_shape"]),
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)