# Changelog — Agente FleetPro

## [2026-04-14] — Agente Inteligente V3

### Adicionado

- **Intent Classifier** (`src/intent_classifier.py`) — 6 classes de intenção via embeddings semânticos (sentence-transformers multilingual) com fallback para regex. Classes: `PRODUCT_LOOKUP`, `SALES_ARGUMENT`, `COMPATIBILITY`, `TECHNICAL_INFO`, `TROUBLESHOOTING`, `PRICE_COMPARISON`
- **Entity Extractor** (`src/entity_extractor.py`) — extrai PNs, equipamentos, categorias, fornecedores e sintomas com fuzzy matching (rapidfuzz) e normalização de sinônimos
- **Hybrid RAG + Reranker** (`src/retriever.py`) — busca híbrida BM25 + ChromaDB com cross-encoder reranker, score mínimo de relevância e k=5 docs
- **Histórico multi-turn no prompt** — últimas 4 trocas injetadas nos prompts de ambos os perfis (vendedor e cliente), habilitando conversas contextuais
- **Normalização de sinônimos na entrada** — `silobag` → `silo bolsa` antes de qualquer busca, resolvendo o bug recorrente de não encontrar variações do termo
- **Arquitetura ASCII Art** (`docs/arquitetura.md`) — diagrama completo do pipeline

### Corrigido

- **`ConversationBufferMemory.clear()`** — LangChain chamava `memory.chat_memory.clear()` internamente; adicionado método à classe customizada

---

## [2026-04-15]

### Adicionado

- **Log de acessos por sessão** (`acessos_log.csv` no GitHub, branch `prod`) — registra `session_id`, `timestamp`, `ip` (X-Forwarded-For), `perfil`, `modelo`, `provedor` em thread background ao inicializar o agente
