"""Geração automatizada de apostila e relatório final do treinamento.

Este módulo consolida os resultados produzidos ao longo das aulas em
um documento único, com linguagem didática e estrutura compatível
com uso em treinamentos, apostilas internas e relatórios técnicos.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Final

import pandas as pd


PROJECT_TITLE: Final[str] = "Treinamento em Inferência Estatística Aplicada à Avaliação Imobiliária"
PROJECT_SUBTITLE: Final[str] = "Material didático, roteiro de aula e relatório automatizado"
DEFAULT_OUTPUT_DIRNAME: Final[str] = "output"


def resolve_project_root() -> Path:
    """Resolve a raiz do projeto a partir da localização deste arquivo."""
    return Path(__file__).resolve().parent.parent


def resolve_output_dir(project_root: Path) -> Path:
    """Resolve e garante a pasta de saída dos relatórios."""
    output_dir = project_root / "data" / DEFAULT_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Lê CSV quando disponível, retornando DataFrame vazio em caso contrário."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_number(value: object, decimals: int = 4) -> str:
    """Formata números para exibição textual padronizada."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"

    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    return str(value)


def format_percentage(value: object, decimals: int = 2) -> str:
    """Formata proporções como percentuais."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value) * 100:.{decimals}f}%".replace(".", ",")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Converte DataFrame em tabela Markdown simples."""
    if df.empty:
        return "_Sem dados disponíveis._"

    preview = df.head(max_rows).copy()
    preview = preview.fillna("-")

    headers = list(preview.columns)
    separator = ["---"] * len(headers)

    lines = [
        "| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for _, row in preview.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")

    return "\n".join(lines)


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Converte DataFrame em tabela HTML simples."""
    if df.empty:
        return "<p><em>Sem dados disponíveis.</em></p>"

    preview = df.head(max_rows).fillna("-")
    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in preview.columns)

    body_rows: list[str] = []
    for _, row in preview.iterrows():
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def load_generated_artifacts(project_root: Path) -> dict[str, pd.DataFrame]:
    """Carrega os artefatos previamente exportados pelas aulas."""
    output_dir = resolve_output_dir(project_root)

    artifacts = {
        "aula_03_coeficientes_ols": safe_read_csv(output_dir / "aula_03_coeficientes_ols.csv"),
        "aula_03_amostra_com_residuos": safe_read_csv(output_dir / "aula_03_amostra_com_residuos.csv"),
        "aula_03_resumo_modelo": safe_read_csv(output_dir / "aula_03_resumo_modelo.csv"),
        "aula_04_diagnosticos_nbr": safe_read_csv(output_dir / "aula_04_diagnosticos_nbr.csv"),
        "aula_04_significancia_individual": safe_read_csv(output_dir / "aula_04_significancia_individual.csv"),
        "aula_04_resumo_aprovacao": safe_read_csv(output_dir / "aula_04_resumo_aprovacao.csv"),
    }

    return artifacts


def build_intro_section() -> str:
    """Monta a introdução da apostila consolidada."""
    return f"""# {PROJECT_TITLE}

## {PROJECT_SUBTITLE}

Este documento consolida o fluxo didático do treinamento de inferência estatística
aplicada à avaliação imobiliária, em alinhamento com a NBR 14653.

O objetivo é reunir, em um único material, a trilha de aprendizagem baseada em:
1. Coleta e unitarização.
2. Sanitização amostral por Chauvenet.
3. Ajuste regressivo por OLS.
4. Diagnósticos estatísticos e validação técnica.

Data de geração: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
"""


def build_training_overview_section() -> str:
    """Monta a seção de visão geral das aulas."""
    return """## Visão geral do treinamento

| Aula | Tema principal | Entregável em código |
| --- | --- | --- |
| Aula 1 | Coleta e unitarização | Script de carga e verificação de integridade |
| Aula 2 | Sanitização amostral | Filtro iterativo com histórico de remoções |
| Aula 3 | Modelagem regressiva | Ajuste do modelo multivariado com intercepto |
| Aula 4 | Diagnósticos NBR 14653 | Relatório automatizado de aprovação/reprovação |
"""


def build_aula_03_section(artifacts: dict[str, pd.DataFrame]) -> str:
    """Monta a seção referente à Aula 3."""
    summary_df = artifacts["aula_03_resumo_modelo"]
    coeff_df = artifacts["aula_03_coeficientes_ols"]
    residuals_df = artifacts["aula_03_amostra_com_residuos"]

    if summary_df.empty:
        summary_text = "_Resumo do modelo ainda não gerado._"
    else:
        row = summary_df.iloc[0]
        summary_text = "\n".join(
            [
                f"- Observações usadas no ajuste: {int(row['n_obs'])}",
                f"- Parâmetros estimados: {int(row['n_params'])}",
                f"- Graus de liberdade do modelo: {int(row['gl_modelo'])}",
                f"- Graus de liberdade dos resíduos: {int(row['gl_residuos'])}",
                f"- R²: {format_number(row['r2'], 4)}",
                f"- R² ajustado: {format_number(row['r2_ajustado'], 4)}",
                f"- RMSE: {format_number(row['rmse'], 4)}",
                f"- Estatística F: {format_number(row['f_statistic'], 6)}",
                f"- p-valor do teste F: {format_number(row['f_p_value'], 6)}",
            ]
        )

    return f"""## Aula 3 - Modelagem regressiva por OLS

A terceira etapa do treinamento apresenta o ajuste do modelo linear múltiplo
com intercepto, enfatizando leitura de coeficientes, resíduos e graus de liberdade.

### Resumo do modelo

{summary_text}

### Tabela de coeficientes

{dataframe_to_markdown(coeff_df)}

### Amostra com resíduos

{dataframe_to_markdown(residuals_df, max_rows=10)}
"""


def build_aula_04_section(artifacts: dict[str, pd.DataFrame]) -> str:
    """Monta a seção referente à Aula 4."""
    diagnostics_df = artifacts["aula_04_diagnosticos_nbr"]
    coeff_significance_df = artifacts["aula_04_significancia_individual"]
    approval_summary_df = artifacts["aula_04_resumo_aprovacao"]

    if approval_summary_df.empty:
        summary_text = "_Resumo de aprovação ainda não gerado._"
    else:
        row = approval_summary_df.iloc[0]
        summary_text = "\n".join(
            [
                f"- Status geral do modelo: {row['status_geral_modelo']}",
                f"- Itens aprovados ou normais: {row['itens_aprovados_ou_normais']} de {row['itens_avaliados']}",
                f"- Proporção de aprovação: {format_percentage(row['proporcao_aprovacao'])}",
                f"- Coeficientes significativos a 10%: {row['coeficientes_significativos_10pct']} de {row['coeficientes_totais']}",
                f"- Observação sobre normalidade: {row['teste_normalidade_observacao']}",
                f"- Faixa de Durbin-Watson usada: {row['durbin_watson_faixa']}",
                f"- Valor observado de Durbin-Watson: {format_number(row['durbin_watson_valor'], 6)}",
            ]
        )

    return f"""## Aula 4 - Diagnósticos estatísticos e validação técnica

A quarta etapa consolida a avaliação final do modelo, cobrindo poder explicativo,
significância global, significância individual, normalidade dos resíduos e
independência serial.

### Resumo executivo

{summary_text}

### Relatório de diagnóstico

{dataframe_to_markdown(diagnostics_df)}

### Significância individual dos coeficientes

{dataframe_to_markdown(coeff_significance_df)}
"""


def build_conclusion_section() -> str:
    """Monta a conclusão do material consolidado."""
    return """## Encerramento

Este material sintetiza a trilha de treinamento em inferência estatística
aplicada à avaliação imobiliária, integrando teoria, automação em Python
e verificação técnica dos pressupostos do modelo.

O documento final pode ser usado como apostila-base, roteiro de exposição
em sala e registro objetivo da execução do projeto.
"""


def build_full_markdown_document(project_root: Path) -> str:
    """Monta o documento completo em Markdown."""
    artifacts = load_generated_artifacts(project_root)

    parts = [
        build_intro_section(),
        build_training_overview_section(),
        build_aula_03_section(artifacts),
        build_aula_04_section(artifacts),
        build_conclusion_section(),
    ]
    return "\n\n".join(parts)


def markdown_to_basic_html(markdown_text: str, artifacts: dict[str, pd.DataFrame]) -> str:
    """Gera versão HTML simples e independente do relatório."""
    aula_03_coeff_html = dataframe_to_html_table(artifacts["aula_03_coeficientes_ols"])
    aula_03_residuals_html = dataframe_to_html_table(
        artifacts["aula_03_amostra_com_residuos"],
        max_rows=10,
    )
    aula_04_diag_html = dataframe_to_html_table(artifacts["aula_04_diagnosticos_nbr"])
    aula_04_coeff_html = dataframe_to_html_table(artifacts["aula_04_significancia_individual"])

    escaped_markdown = escape(markdown_text)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <title>Relatório Final - Treinamento Inferencial</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px auto;
            max-width: 1080px;
            padding: 0 20px;
            color: #222;
            background: #fff;
        }}
        h1, h2, h3 {{
            color: #0f3d5e;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 24px;
        }}
        th, td {{
            border: 1px solid #cccccc;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background: #f2f4f7;
        }}
        code, pre {{
            background: #f7f7f7;
            border-radius: 4px;
        }}
        .block {{
            margin-bottom: 36px;
        }}
        .muted {{
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="block">
        <h1>{escape(PROJECT_TITLE)}</h1>
        <h2>{escape(PROJECT_SUBTITLE)}</h2>
        <p class="muted">Versão HTML derivada do relatório consolidado em Markdown.</p>
    </div>

    <div class="block">
        <h2>Conteúdo consolidado em Markdown</h2>
        <pre>{escaped_markdown}</pre>
    </div>

    <div class="block">
        <h2>Tabela de coeficientes da Aula 3</h2>
        {aula_03_coeff_html}
    </div>

    <div class="block">
        <h2>Amostra com resíduos da Aula 3</h2>
        {aula_03_residuals_html}
    </div>

    <div class="block">
        <h2>Diagnósticos da Aula 4</h2>
        {aula_04_diag_html}
    </div>

    <div class="block">
        <h2>Significância individual da Aula 4</h2>
        {aula_04_coeff_html}
    </div>
</body>
</html>
"""


def export_reports(project_root: Path) -> tuple[Path, Path]:
    """Exporta relatório final em Markdown e HTML."""
    output_dir = resolve_output_dir(project_root)
    artifacts = load_generated_artifacts(project_root)

    markdown_text = build_full_markdown_document(project_root)
    html_text = markdown_to_basic_html(markdown_text, artifacts)

    markdown_path = output_dir / "relatorio_final_treinamento_inferencial.md"
    html_path = output_dir / "relatorio_final_treinamento_inferencial.html"

    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")

    return markdown_path, html_path


def main() -> None:
    """Executa a geração consolidada da apostila final."""
    project_root = resolve_project_root()
    markdown_path, html_path = export_reports(project_root)

    print("\n" + "=" * 72)
    print("Relatórios gerados com sucesso")
    print("=" * 72)
    print(f"Markdown: {markdown_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
