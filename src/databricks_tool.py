"""
Databricks SQL Tool — Agente FleetPro
Replica a action 'Deterministico databricks - ExecuteSQL' do Copilot Studio.
Chama POST /api/2.0/sql/statements no workspace Azure Databricks.
"""
from __future__ import annotations
import requests
import json
from typing import Optional

DATABRICKS_HOST = "https://adb-5920274570509254.14.azuredatabricks.net"
WAREHOUSE_ID    = "52e7ced7f3571b26"
TABLE_FLEETPRO  = "silver_part_sandbox.fleetpro.fleetpro_com_cd_tech_type"
WAIT_TIMEOUT    = "30s"


def _executar_sql(sql: str, token: str) -> dict:
    """Executa SQL no Databricks SQL Warehouse. Retorna resposta JSON bruta."""
    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "statement": sql,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": WAIT_TIMEOUT,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=35)
    resp.raise_for_status()
    return resp.json()


def _parse_resultado(raw: dict) -> list[dict]:
    """Converte resposta Databricks (column_names + data_array) em lista de dicts."""
    result = raw.get("result", {})
    manifest = raw.get("manifest", {})
    schema = manifest.get("schema", {})
    columns = [c["name"] for c in schema.get("columns", [])]
    rows = result.get("data_array", [])
    return [dict(zip(columns, row)) for row in rows]


def buscar_por_pn(pn: str, token: str, max_rows: int = 20) -> list[dict]:
    """Busca por PN genuíno, FleetPro ou alternativo na tabela Databricks."""
    pn_safe = pn.replace("'", "''").strip()
    sql = f"""
        SELECT *
        FROM {TABLE_FLEETPRO}
        WHERE UPPER(pn_gen) = UPPER('{pn_safe}')
           OR UPPER(pn_fleetpro) = UPPER('{pn_safe}')
           OR UPPER(pn_alternative) = UPPER('{pn_safe}')
           OR UPPER(pn_nxp) = UPPER('{pn_safe}')
        LIMIT {max_rows}
    """
    raw = _executar_sql(sql.strip(), token)
    return _parse_resultado(raw)


def buscar_por_modelo_categoria(
    modelo: Optional[str],
    categoria: Optional[str],
    token: str,
    max_rows: int = 10,
) -> list[dict]:
    """Busca por modelo de equipamento e/ou categoria (marketing_project)."""
    clausulas = []

    if modelo and modelo != "-":
        m = modelo.replace("'", "''")
        clausulas.append(f"""(
            UPPER(tractor_case_ih) LIKE UPPER('%{m}%') OR
            UPPER(combine_case_ih) LIKE UPPER('%{m}%') OR
            UPPER(sprayers_case_ih) LIKE UPPER('%{m}%') OR
            UPPER(tractor_nhag) LIKE UPPER('%{m}%') OR
            UPPER(combine_nhag) LIKE UPPER('%{m}%') OR
            UPPER(sprayers_nhag) LIKE UPPER('%{m}%') OR
            UPPER(other_machines_case_ih) LIKE UPPER('%{m}%') OR
            UPPER(other_machines_nhag) LIKE UPPER('%{m}%')
        )""")

    if categoria and categoria != "-":
        c = categoria.replace("'", "''")
        clausulas.append(f"UPPER(marketing_project) LIKE UPPER('%{c}%')")

    if not clausulas:
        return []

    where = " AND ".join(clausulas)
    sql = f"SELECT * FROM {TABLE_FLEETPRO} WHERE {where} LIMIT {max_rows}"
    raw = _executar_sql(sql, token)
    return _parse_resultado(raw)


def buscar_fleetpro(
    pn: Optional[str],
    modelo: Optional[str],
    categoria: Optional[str],
    token: str,
    max_rows: int = 10,
) -> dict:
    """
    Entry point principal — replica lógica do Power Automate do Copilot.
    Retorna: {encontrado, muito, resultado (list[dict]), total}
    """
    resultados = []

    if pn and pn != "-":
        resultados = buscar_por_pn(pn, token, max_rows=max_rows)

    if not resultados and (modelo or categoria):
        resultados = buscar_por_modelo_categoria(modelo, categoria, token, max_rows=max_rows)

    total = len(resultados)
    return {
        "encontrado": total > 0,
        "muito": total > 5,
        "total": total,
        "resultado": resultados,
    }


def formatar_resultado_markdown(busca: dict) -> str:
    """Formata o resultado da busca em markdown para exibir no chat."""
    if not busca["encontrado"]:
        return "Nenhum resultado encontrado no Databricks para esses filtros."

    if busca["muito"]:
        return (
            f"{busca['total']} resultados encontrados. "
            "Refine por categoria ou modelo para ver os produtos."
        )

    linhas = []
    for row in busca["resultado"]:
        pn_fp  = row.get("pn_fleetpro", "-")
        pn_gen = row.get("pn_gen", "-")
        desc   = row.get("description", "-")
        cat    = row.get("marketing_project", "-")
        linhas.append(f"- **{pn_fp}** | Genuíno: `{pn_gen}` | {desc} | _{cat}_")

    return "\n".join(linhas)
