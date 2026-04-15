"""
Retriever Híbrido — Agente FleetPro
BM25 (lexical) + ChromaDB (semântico) → cross-encoder reranker.
Retorna docs ordenados por relevância com score mínimo.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma

# Score mínimo para considerar um doc relevante
SCORE_MINIMO = 0.3
K_RETRIEVAL = 10   # docs finais após reranking
K_CANDIDATOS = 20  # candidatos antes do reranking


def hybrid_search(
    query: str,
    vectorstore: "Chroma",
    k: int = K_RETRIEVAL,
    score_minimo: float = SCORE_MINIMO,
) -> list[dict]:
    """
    Busca híbrida: semântica (ChromaDB) + lexical (BM25).
    Retorna lista de {'content': str, 'source': str, 'score': float}.
    Fallback para busca semântica simples se BM25/reranker não disponíveis.
    """
    # Busca semântica via ChromaDB (sempre disponível)
    try:
        resultados_semanticos = vectorstore.similarity_search_with_score(query, k=K_CANDIDATOS)
    except Exception:
        resultados_semanticos = []
        docs_simples = vectorstore.similarity_search(query, k=k)
        return [
            {"content": d.page_content, "source": d.metadata.get("source", ""), "score": 0.5}
            for d in docs_simples
        ]

    # Normaliza resultados semânticos (ChromaDB retorna distância, menor = melhor)
    docs_pool = []
    for doc, dist in resultados_semanticos:
        score_sem = max(0.0, 1.0 - dist)  # converte distância → similaridade
        docs_pool.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "score_sem": score_sem,
            "score_bm25": 0.0,
        })

    # Tenta enriquecer com BM25
    try:
        docs_pool = _enriquecer_bm25(query, docs_pool)
    except Exception:
        pass  # BM25 é opcional

    # Score combinado (60% semântico + 40% BM25)
    for d in docs_pool:
        d["score"] = round(0.6 * d["score_sem"] + 0.4 * d["score_bm25"], 4)

    # Tenta reranking com cross-encoder
    try:
        docs_pool = _rerankar(query, docs_pool)
    except Exception:
        docs_pool.sort(key=lambda x: x["score"], reverse=True)

    # Filtra por score mínimo e limita a k
    resultado = [d for d in docs_pool if d["score"] >= score_minimo][:k]

    # Se nada passou do threshold, retorna os top-k sem filtro
    if not resultado:
        resultado = docs_pool[:k]

    return resultado


def _enriquecer_bm25(query: str, docs: list[dict]) -> list[dict]:
    """Calcula score BM25 para cada doc e adiciona ao pool."""
    from rank_bm25 import BM25Okapi

    corpus = [d["content"].lower().split() for d in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())

    max_score = max(scores) if max(scores) > 0 else 1.0
    for doc, score in zip(docs, scores):
        doc["score_bm25"] = round(float(score) / max_score, 4)

    return docs


def _rerankar(query: str, docs: list[dict]) -> list[dict]:
    """Cross-encoder reranker para reordenar docs por relevância real."""
    from sentence_transformers import CrossEncoder

    model = _get_reranker()
    pares = [(query, d["content"][:512]) for d in docs]
    scores = model.predict(pares)

    # Normaliza scores do cross-encoder para [0, 1]
    import math
    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for doc, score in zip(docs, scores):
        doc["score"] = round(_sigmoid(float(score)), 4)

    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs


_reranker_cache: object = None


def _get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        from sentence_transformers import CrossEncoder
        _reranker_cache = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _reranker_cache
