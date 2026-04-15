# Changelog — Agente FleetPro

## [2026-04-15]

### Adicionado
- **Log de acessos por sessão** (`acessos_log.csv` no GitHub, branch `prod`)
  - Registra: `session_id`, `timestamp`, `ip` (X-Forwarded-For), `perfil`, `modelo`, `provedor`
  - Salvo em thread background no momento da inicialização do agente
  - Permite rastrear quem acessou o app no Streamlit Cloud

### Corrigido
- **`ConversationBufferMemory.clear()`** — adicionado método `clear()` na classe customizada
  - Erro: `'ConversationBufferMemory' object has no attribute 'clear'`
  - Causado pelo LangChain chamando `memory.chat_memory.clear()` internamente ao reportar erro
