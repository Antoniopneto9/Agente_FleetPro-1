# Agente FleetPro — Contexto do Projeto

## O que é

Chatbot RAG para suporte à venda e consulta de peças de reposição **FleetPro** (marca própria CNH Industrial para máquinas Case IH e New Holland). Desenvolvido em Streamlit + LangChain + ChromaDB.

---

## Arquitetura

### Arquivo principal
- `modelo_fleetV2.py` — toda a lógica do app (Streamlit, RAG, busca na Matriz, LLM, feedback)

### Fontes de dados
- `base_docs/Matriz_FP.xlsx` — planilha principal com todos os PNs FleetPro (aba `FP MATRIZ`)
- `base_docs/treinamentos/Tratados/*.md` — documentos de treinamento indexados no RAG
- `erros_reportados.csv` — salvo **apenas no GitHub** via API (não versionado localmente)

### Pipeline de resposta
1. **Busca na Matriz** (determinística): PN → TT code → equipamento+categoria → categoria → equipamento → fallback `description`
2. **Busca RAG** (semântica): ChromaDB + HuggingFace embeddings sobre os `.md` de treinamento
3. **LLM** (Groq llama-3.3-70b-versatile ou OpenAI): monta resposta com contexto das duas fontes
4. **Bloqueio anti-alucinação**: se Matriz e RAG retornam vazio, não chama o LLM

### Perfis de usuário
- **Vendedor de balcão** — prompt focado em argumentos de venda e cross-selling
- **Cliente/Agricultor** — prompt focado em benefícios e confiança no produto

---

## Fluxo Git

```
prod  →  main (Antoniopneto9)  →  master (bonucci0302) — repo oficial CNH
```

| Branch | Função |
|--------|--------|
| `prod` | Desenvolvimento e testes — commits aqui primeiro |
| `main` | Estável pessoal — merge via PR do `prod` |
| `master` | Repo oficial Bonucci — PR do `main` para lá |

**Regra:** nunca commitar direto em `main`. Sempre `prod` → PR → `main`.

### Remotes
```
origin   → https://github.com/Antoniopneto9/Agente_FleetPro-1.git
upstream → https://github.com/bonucci0302/Agente_FleetPro.git
```

### Push padrão
```bash
git push origin prod          # sobe para prod
# PR no GitHub: prod → main   # aprovação antes de ir para main
```

---

## Ambientes publicados (Streamlit Cloud)

| App | Branch | Propósito |
|-----|--------|-----------|
| App de testes | `prod` | Validar experiência real (latência, feedback, RAG) antes de mergear |
| App estável | `main` | Versão que usuários finais usam |

**Por que dois apps online:** permite validar timing, UX real e salvamento no GitHub antes de promover para `main`. Testes locais não simulam latência de rede nem o comportamento do Streamlit Cloud.

### Secrets necessários no Streamlit Cloud
```toml
GITHUB_TOKEN = "ghp_..."        # para salvar erros_reportados.csv via API
GROQ_API_KEY = "gsk_..."        # LLM principal
# OPENAI_API_KEY = "sk-..."     # fallback se Groq indisponível
```

### Secrets locais
Criar `.streamlit/secrets.toml` (não versionado):
```toml
GITHUB_TOKEN = "ghp_..."
```

---

## Feature: Reportar Erro (sidebar)

- Usuário descreve o erro no expander "🐛 Reportar Erro"
- Ao enviar: salva `timestamp`, `descricao`, `modelo`, `provedor`, `historico_chat` no `erros_reportados.csv` via **GitHub Contents API (PUT)**
- Após envio: chat é resetado e text_area limpa — evita reenvio do mesmo histórico
- CSV usa `QUOTE_ALL` para preservar `\n` nos campos de texto

---

## Problemas conhecidos / histórico de bugs

| Bug | Causa | Status |
|-----|-------|--------|
| ChromaDB "client has been closed" | `st.cache_resource` incompatível com ChromaDB 0.4.24 | ✅ Resolvido — vectorstore em `session_state` |
| HuggingFace SSL error | Rede corporativa CNH bloqueia downloads | ✅ Resolvido — try/except, flag `_rag_indisponivel` |
| CSV com 1951 linhas | `buf.write` raw sem re-quoting dos `\n` internos | ✅ Resolvido — `DictReader` parse + `QUOTE_ALL` |
| HTTP 422 GitHub API | SHA desatualizado ou base64 com quebras | ✅ Resolvido — fetch com `?ref=main`, limpar base64 |
| Sidebar se escondendo | Padrão Streamlit ao enviar chat | ✅ Resolvido — `initial_sidebar_state="expanded"` |
| Contagem não detectada | Palavras-chave insuficientes em `_detectar_pergunta_contagem` | ✅ Resolvido — 12 variações adicionadas |
| Busca por modelo (2188, 7088) | `detectar_busca_por_equipamento` não casava números de modelo | ✅ Resolvido — mensagem orientativa + fallback `description` |
| SILOBAG não encontrado | `buscar_por_description` não existia | ✅ Resolvido — nova função fallback por `description` |
| Groq APIConnectionError local | Rede corporativa CNH bloqueia API Groq | ⚠️ Contornar — testar via Streamlit Cloud |

---

## Estrutura de colunas da Matriz FP

Colunas principais (35 no total):
```
marketing_project, description, pn_gen, pn_alternative, pn_nxp, pn_fleetpro,
tractor_case_ih, combine_case_ih, headers_case_ih, sch_case_ih,
sprayers_case_ih, planters_case_ih, other_machines_case_ih,
tractor_nhag, combine_nhag, headers_nhag, sprayers_nhag,
planters_nhag, forage_balers_and_others_nhag, other_machines_nhag,
john_deere, macdon, agco, ideal, massey_ferguson, valtra, cd_tech_type, ...
```

**Busca por equipamento** usa colunas `*_case_ih` e `*_nhag`. Modelos específicos (ex: "2188") ficam dentro das células dessas colunas separados por `;`.

---

## Desenvolvimento

### Rodar localmente
```bash
streamlit run modelo_fleetV2.py
```
> Groq pode falhar na rede corporativa CNH — testar via app online de `prod`.

### Dependências
```
pip install -r requirements.txt
```

### Reindexar RAG
- Automático ao inicializar o app
- Manual: botão "Reindexar" na sidebar
