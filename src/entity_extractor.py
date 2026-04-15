"""
Entity Extractor — Agente FleetPro
Extrai entidades estruturadas da query: PNs, equipamentos, categorias, fornecedores.
Usa fuzzy matching para lidar com variações (PUMA150 → PUMA 150, silobag → silo bolsa).
"""
from __future__ import annotations
import re
from typing import TypedDict

try:
    from rapidfuzz import process, fuzz
    _FUZZY_OK = True
except ImportError:
    _FUZZY_OK = False


class Entities(TypedDict):
    pns: list[str]
    equipamentos: list[str]
    categorias: list[str]
    fornecedores: list[str]
    sintomas: list[str]


# Sinônimos conhecidos: termo digitado → termo canônico
_SINONIMOS: dict[str, str] = {
    "silobag":         "silo bolsa",
    "silobolsa":       "silo bolsa",
    "silo bag":        "silo bolsa",
    "bolsa silo":      "silo bolsa",
    "teejet":          "teejet",
    "corrente":        "correntes",
    "correntes":       "correntes",
    "filtro":          "filtros",
    "rolamento":       "rolamentos",
    "vedacao":         "vedação",
    "vedaçao":         "vedação",
    "lubrificante":    "lubrificantes",
    "cilindro":        "cilindros",
    "plastico":        "plásticos",
    "plasticos":       "plásticos",
    "pino":            "pinos & buchas",
    "bucha":           "pinos & buchas",
    "correia":         "correias",
    "usinado":         "usinados",
}

_FORNECEDORES = ["vv", "teejet", "ipesa", "rexnord", "petronas", "rex"]

_SINTOMAS = [
    "vibração", "vibracao", "aquecimento", "barulho", "desgaste",
    "entupimento", "quebra", "vazamento", "falha", "problema",
    "parada", "travamento", "ruido", "ruído",
]


def extract_entities(query: str) -> Entities:
    q = query.lower().strip()

    # Normaliza sinônimos antes de processar
    q_norm = _normalizar_sinonimos(q)

    return Entities(
        pns=_extrair_pns(q),
        equipamentos=_extrair_equipamentos(q_norm),
        categorias=_extrair_categorias(q_norm),
        fornecedores=_extrair_fornecedores(q_norm),
        sintomas=_extrair_sintomas(q_norm),
    )


def _normalizar_sinonimos(text: str) -> str:
    """Substitui variações conhecidas pelo termo canônico."""
    for variante, canonico in _SINONIMOS.items():
        text = re.sub(r'\b' + re.escape(variante) + r'\b', canonico, text, flags=re.IGNORECASE)
    return text


def _extrair_pns(text: str) -> list[str]:
    """Extrai candidatos a PN: sequências alfanuméricas com letras e números."""
    candidatos = re.findall(r'\b[A-Z0-9]{4,}(?:[A-Z0-9\-]{0,10})\b', text.upper())
    # Filtra stopwords comuns que não são PNs
    stopwords = {"CASE", "NEW", "HOLLAND", "TRATOR", "FILTRO", "PARA", "COM", "FLEETPRO"}
    return [c for c in candidatos if c not in stopwords]


def _extrair_equipamentos(text: str) -> list[str]:
    """Detecta modelos de equipamentos mencionados."""
    padroes = [
        r'puma\s*\d+',
        r't[78]\.\d+',
        r'axial.?flow',
        r'2\d{3}\b',           # modelos tipo 2188, 7088
        r'case\s*ih',
        r'new\s*holland',
        r'colheitadeira',
        r'trator',
        r'plantadeira',
        r'pulverizador',
    ]
    encontrados = []
    for p in padroes:
        matches = re.findall(p, text, re.IGNORECASE)
        encontrados.extend([m.strip() for m in matches])
    return list(dict.fromkeys(encontrados))  # dedup mantendo ordem


def _extrair_categorias(text: str) -> list[str]:
    """Detecta categorias de produto."""
    categorias = [
        "silo bolsa", "filtros", "correntes", "rolamentos", "vedação",
        "lubrificantes", "cilindros", "plásticos", "pinos & buchas",
        "correias", "usinados", "lâminas", "laminas", "pulverização",
        "undercarriage", "bucket",
    ]
    encontradas = []
    for cat in categorias:
        if cat in text:
            encontradas.append(cat)

    # Fuzzy match se nada encontrado e rapidfuzz disponível
    if not encontradas and _FUZZY_OK:
        resultado = process.extractOne(text, categorias, scorer=fuzz.partial_ratio, score_cutoff=70)
        if resultado:
            encontradas.append(resultado[0])

    return encontradas


def _extrair_fornecedores(text: str) -> list[str]:
    return [f for f in _FORNECEDORES if f in text]


def _extrair_sintomas(text: str) -> list[str]:
    return [s for s in _SINTOMAS if s in text]
