"""
Intent Classifier — Agente FleetPro
Detecta a intenção do usuário usando embeddings semânticos (sentence-transformers)
comparados a exemplos anotados, com fallback para regex/keywords existentes.
"""
from __future__ import annotations
import re
from typing import TypedDict

# Exemplos de referência por intent (few-shot via cosine similarity)
_EXEMPLOS: dict[str, list[str]] = {
    "PRODUCT_LOOKUP": [
        "preciso de filtro para puma 150",
        "tem corrente para colheitadeira",
        "qual o pn do rolamento",
        "busca o código 1890DSH",
        "quero peça para new holland t7",
        "tem vedação para motor case",
        "qual equivalente ao pn original 47390059",
        "filtro de ar para trator new holland",
    ],
    "SALES_ARGUMENT": [
        "como vender o fleetpro",
        "quais os benefícios da corrente fleetpro",
        "argumentos para convencer o cliente",
        "como rebater objeção de preço",
        "vantagens do fleetpro sobre a original",
        "por que o cliente deve comprar fleetpro",
        "como justificar a troca pela fleetpro",
    ],
    "COMPATIBILITY": [
        "esse pn serve para puma 150",
        "funciona na colheitadeira 2188",
        "é compatível com new holland",
        "qual equipamento usa esse filtro",
        "serve para case ih",
        "esse rolamento é equivalente ao original",
        "posso usar no axial flow",
    ],
    "TECHNICAL_INFO": [
        "qual a especificação do filtro",
        "dimensões do rolamento",
        "como funciona o teejet",
        "descrição técnica da corrente",
        "material da lâmina vv",
        "vida útil do filtro fleetpro",
        "ficha técnica do produto",
    ],
    "TROUBLESHOOTING": [
        "trator com vibração no motor",
        "problema com aquecimento",
        "corrente quebrando rápido",
        "filtro entupindo",
        "desgaste prematuro da peça",
        "barulho estranho no equipamento",
        "máquina parando durante a safra",
    ],
    "PRICE_COMPARISON": [
        "quanto custa comparado ao original",
        "fleetpro é mais barato",
        "economia com fleetpro",
        "diferença de preço entre fleetpro e original",
        "vale a pena trocar pelo fleetpro",
        "relação custo benefício",
        "preço competitivo",
    ],
}


class IntentResult(TypedDict):
    intent: str
    confidence: float
    method: str  # "embedding" | "regex" | "fallback"


def classify_intent(query: str) -> IntentResult:
    """
    Classifica a intenção da query.
    Tenta embedding semântico primeiro, cai para regex se não disponível.
    """
    query_lower = query.lower().strip()

    # Tenta embedding semântico
    try:
        return _classify_embedding(query_lower)
    except Exception:
        pass

    # Fallback: regex/keywords
    return _classify_regex(query_lower)


def _classify_embedding(query: str) -> IntentResult:
    from sentence_transformers import SentenceTransformer, util
    import torch

    model = _get_model()
    q_emb = model.encode(query, convert_to_tensor=True)

    best_intent = "PRODUCT_LOOKUP"
    best_score = -1.0

    for intent, exemplos in _EXEMPLOS.items():
        embs = model.encode(exemplos, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, embs)[0]
        score = float(scores.max())
        if score > best_score:
            best_score = score
            best_intent = intent

    return IntentResult(intent=best_intent, confidence=round(best_score, 3), method="embedding")


def _classify_regex(query: str) -> IntentResult:
    """Fallback com keywords quando sentence-transformers não está disponível."""
    patterns = {
        "SALES_ARGUMENT": r"argument|benefit|vantag|vend|convenc|rebat|objeç|como (vender|apresentar)",
        "COMPATIBILITY":  r"serve para|compat|funciona|equivalente|serve no|é para",
        "TECHNICAL_INFO": r"especif|dimens|ficha|material|como funciona|descriç",
        "TROUBLESHOOTING": r"problem|vibra|aquec|quebr|entup|desgast|barulh|parando",
        "PRICE_COMPARISON": r"preç|cust|barato|econom|valor|caro|diferença de preço",
    }
    for intent, pattern in patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            return IntentResult(intent=intent, confidence=0.7, method="regex")

    return IntentResult(intent="PRODUCT_LOOKUP", confidence=0.5, method="fallback")


# Cache do modelo para não recarregar a cada chamada
_model_cache: object = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _model_cache
