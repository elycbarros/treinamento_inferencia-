"""Aula 03 - Regressão linear por mínimos quadrados ordinários.

Script executável da terceira aula do treinamento inferencial.
O foco é ajustar um modelo de regressão linear múltipla com
intercepto, apresentar os graus de liberdade e organizar a leitura
didática dos coeficientes e do poder explicativo do modelo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.unitarizacao import add_unit_price_column
from servicos.regressao import (
    attach_predictions_and_residuals,
    build_coefficients_table,
    build_model_summary,
    fit_ols_regression,
)

DATASET_CANDIDATES: Final[tuple[str, ...]] = (
    "amostras_residencial35.csv",
    "amostrasresidencial35.csv",
    "amostras_residencial.csv",
)

TARGET_COLUMN: Final[str] = "preco"
PREFERRED_FEATURES: Final[tuple[str, ...]] = (
    "areaprivativa",
    "vagas",
    "distanciacentrokm",
    "dist_praia",
)


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza automaticamente a base padrão das aulas iniciais."""
    data_dir = project_root / "data"

    for filename in DATASET_CANDIDATES:
        candidate = data_dir / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Nenhum arquivo padrão foi encontrado na pasta `data`. "
        f"Arquivos procurados: {searched}."
    )


def print_section_title(title: str) -> None:
    """Imprime um cabeçalho visual para a execução da aula."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def resolve_equation_terms(coeff_table: pd.DataFrame) -> str:
    """Gera uma representação textual da equação estimada."""
    pieces: list[str] = []

    for _, row in coeff_table.iterrows():
        variable = str(row["variavel"])
        coefficient = float(row["coeficiente"])

        if variable == "const":
            pieces.append(f"{coefficient:,.2f}")
            continue

        signal = "+" if coefficient >= 0 else "-"
        pieces.append(f"{signal} {abs(coefficient):,.2f}·{variable}")

    return "Preço = " + " ".join(pieces)


def print_sample_context(df: pd.DataFrame) -> None:
    """Exibe contexto inicial da base usada no ajuste."""
    print_section_title("Base utilizada")
    print(f"Quantidade de registros disponíveis: {len(df)}")

    preview_columns = [
        column
        for column in (
            "id",
            "preco",
            "areaprivativa",
            "vagas",
            "distanciacentrokm",
            "dist_praia",
            "vu",
        )
        if column in df.columns
    ]
    print(df[preview_columns].head(8).to_string(index=False))


def print_model_overview(summary: dict[str, float], equation: str) -> None:
    """Imprime visão geral do modelo ajustado."""
    print_section_title("Visão geral do modelo OLS")
    print(equation)
    print(f"Observações usadas no ajuste: {summary['n_obs']}")
    print(f"Parâmetros estimados: {summary['n_params']}")
    print(f"Graus de liberdade do modelo: {summary['gl_modelo']}")
    print(f"Graus de liberdade dos resíduos: {summary['gl_residuos']}")
    print(f"R²: {summary['r2']:.4f}")
    print(f"R² ajustado: {summary['r2_ajustado']:.4f}")
    print(f"RMSE: {summary['rmse']:,.2f}")
    print(f"Teste F: {summary['f_statistic']:.4f}")
    print(f"p-valor do teste F: {summary['f_p_value']:.6f}")


def print_coefficients_table(coeff_table: pd.DataFrame) -> None:
    """Exibe a tabela didática de coeficientes."""
    print_section_title("Coeficientes e significância individual")
    display_table = coeff_table.copy()
    numeric_columns = ["coeficiente", "erro_padrao", "estatistica_t", "p_valor"]
    display_table[numeric_columns] = display_table[numeric_columns].round(6)
    print(display_table.to_string(index=False))


def print_predictions_preview(enriched_df: pd.DataFrame) -> None:
    """Exibe amostra dos valores ajustados e resíduos."""
    print_section_title("Amostra de predições e resíduos")
    preview_columns = [
        column
        for column in (
            "id",
            "preco",
            "valor_ajustado",
            "residuo",
        )
        if column in enriched_df.columns
    ]
    print(enriched_df[preview_columns].head(10).round(4).to_string(index=False))


def export_outputs(
    project_root: Path,
    coeff_table: pd.DataFrame,
    enriched_df: pd.DataFrame,
    summary: dict[str, float],
) -> None:
    """Exporta os artefatos da Aula 3 para a pasta data/output."""
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients_path = output_dir / "aula_03_coeficientes_ols.csv"
    fitted_path = output_dir / "aula_03_amostra_com_residuos.csv"
    summary_path = output_dir / "aula_03_resumo_modelo.csv"

    coeff_table.to_csv(coefficients_path, index=False)
    enriched_df.to_csv(fitted_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print_section_title("Arquivos exportados")
    print(f"Coeficientes OLS: {coefficients_path}")
    print(f"Amostra com resíduos: {fitted_path}")
    print(f"Resumo do modelo: {summary_path}")


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo principal da Aula 3."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 03 - Modelagem regressiva por OLS")
    print(f"Raiz do projeto: {project_root}")
    print(f"Arquivo utilizado: {dataset_path}")

    df_raw = load_raw_dataset(dataset_path)
    df_prepared = add_unit_price_column(df_raw)

    print_sample_context(df_prepared)

    artifacts = fit_ols_regression(
        df=df_prepared,
        target_col=TARGET_COLUMN,
        feature_columns=list(PREFERRED_FEATURES),
        add_intercept=True,
    )

    coeff_table = build_coefficients_table(artifacts)
    summary = build_model_summary(artifacts)
    equation = resolve_equation_terms(coeff_table)
    enriched_df = attach_predictions_and_residuals(df_prepared, artifacts)

    print_model_overview(summary, equation)
    print_coefficients_table(coeff_table)
    print_predictions_preview(enriched_df)
    export_outputs(project_root, coeff_table, enriched_df, summary)


if __name__ == "__main__":
    main()


"""
Bloco 13 - Extração organizada dos coeficientes e p-valores.

Este bloco transforma o resultado bruto do ajuste OLS em uma
tabela didática de interpretação, destacando coeficientes,
erros-padrão, estatísticas t, p-valores e a decisão de
significância individual para cada parâmetro estimado.

Objetivos do bloco:
- extrair os parâmetros estimados do modelo;
- organizar coeficientes, erro-padrão, t e p-valor em DataFrame;
- classificar significância individual com base no nível de 10%;
- facilitar a leitura do teste t para fins normativos e didáticos;
- deixar a saída pronta para relatório e apostila.
"""

coeficientes_df = pd.DataFrame({
    "variavel_explicativa": ols_results.params.index,
    "coeficiente": ols_results.params.values,
    "erro_padrao": ols_results.bse.values,
    "estatistica_t": ols_results.tvalues.values,
    "p_valor_t": ols_results.pvalues.values,
})

coeficientes_df["significativo_10pct"] = coeficientes_df["p_valor_t"].apply(
    lambda p: "SIM" if p <= 0.10 else "NAO"
)

coeficientes_df = coeficientes_df.round({
    "coeficiente": 4,
    "erro_padrao": 4,
    "estatistica_t": 4,
    "p_valor_t": 4,
})

coeficientes_relatorio_df = coeficientes_df.rename(columns={
    "variavel_explicativa": "Variável Explicativa",
    "coeficiente": "Coeficiente",
    "erro_padrao": "Erro Padrão",
    "estatistica_t": "Estatística t",
    "p_valor_t": "p-valor t",
    "significativo_10pct": "Significativo 10%",
})

display(coeficientes_relatorio_df)


"""
Bloco 14 - Predições e resíduos do modelo ajustado.

Este bloco organiza a saída observacional do ajuste OLS,
anexando ao conjunto de dados os valores preditos pelo modelo
e os respectivos resíduos, de modo a preparar a etapa seguinte
de diagnósticos formais e validações normativas.

Objetivos do bloco:
- calcular os valores ajustados para cada observação;
- calcular os resíduos brutos do modelo;
- consolidar uma base enriquecida para inspeção linha a linha;
- apoiar análises visuais e testes diagnósticos posteriores;
- deixar a saída pronta para exportação e relatório.
"""

amostra_com_residuos_df = df_model.copy()
amostra_com_residuos_df["valor_ajustado"] = ols_results.fittedvalues
amostra_com_residuos_df["residuo"] = ols_results.resid

colunas_visualizacao = [
    coluna
    for coluna in (
        "id",
        "preco",
        "valor_ajustado",
        "residuo",
        "areaprivativa",
        "vagas",
        "distanciacentrokm",
        "dist_praia",
    )
    if coluna in amostra_com_residuos_df.columns
]

display(amostra_com_residuos_df[colunas_visualizacao].head(10).round(4))


"""
Bloco 15 - Diagnósticos formais de normalidade e resíduos.

Este bloco inicia a etapa formal de verificação dos pressupostos
estatísticos do modelo ajustado, com foco na distribuição dos
resíduos e em métricas descritivas que antecedem a consolidação
normativa dos diagnósticos da NBR 14653.

Objetivos do bloco:
- resumir o comportamento dos resíduos do modelo;
- executar teste formal de normalidade dos resíduos;
- organizar a leitura do p-valor do teste de Shapiro-Wilk;
- registrar a situação do ajuste quanto à normalidade;
- preparar a transição para os diagnósticos normativos seguintes.
"""

from scipy import stats

residuos_modelo = pd.Series(ols_results.resid, name="residuo")

resumo_residuos_df = pd.DataFrame({
    "metrica": [
        "quantidade",
        "media",
        "desvio_padrao",
        "minimo",
        "q1",
        "mediana",
        "q3",
        "maximo",
    ],
    "valor": [
        residuos_modelo.count(),
        residuos_modelo.mean(),
        residuos_modelo.std(ddof=1),
        residuos_modelo.min(),
        residuos_modelo.quantile(0.25),
        residuos_modelo.median(),
        residuos_modelo.quantile(0.75),
        residuos_modelo.max(),
    ],
})

estatistica_shapiro, p_valor_shapiro = stats.shapiro(residuos_modelo)

diagnostico_normalidade_df = pd.DataFrame({
    "teste": ["Shapiro-Wilk"],
    "estatistica": [estatistica_shapiro],
    "p_valor": [p_valor_shapiro],
    "criterio": ["p-valor > 0.05"],
    "status": ["NORMAL" if p_valor_shapiro > 0.05 else "NAO_NORMAL"],
})

display(resumo_residuos_df.round(4))
display(diagnostico_normalidade_df.round(6))


"""
Bloco 16 - Independência dos resíduos com Durbin-Watson.

Este bloco avalia a independência serial dos resíduos do modelo,
utilizando a estatística de Durbin-Watson como métrica formal de
diagnóstico. A leitura didática busca verificar se os resíduos se
mantêm aproximadamente independentes, condição importante para a
consistência interpretativa do ajuste regressivo.

Objetivos do bloco:
- calcular a estatística de Durbin-Watson;
- registrar o valor obtido em estrutura tabular;
- classificar o diagnóstico com base em faixa de referência;
- apoiar a leitura técnica da independência dos resíduos;
- preparar a consolidação final dos pressupostos normativos.
"""

residuos_modelo = pd.Series(ols_results.resid, name="residuo")

durbin_watson = (
    ((residuos_modelo.diff().dropna() ** 2).sum()) /
    ((residuos_modelo ** 2).sum())
)

if 1.5 <= durbin_watson <= 2.5:
    status_dw = "APROVADO"
else:
    status_dw = "REPROVADO"

diagnostico_dw_df = pd.DataFrame({
    "teste": ["Durbin-Watson"],
    "estatistica": [durbin_watson],
    "criterio": ["proximo de 2.0; aceitavel entre 1.5 e 2.5"],
    "status": [status_dw],
})

display(diagnostico_dw_df.round(6))


"""
Bloco 17 - Consolidação de todos os diagnósticos normativos.

Este bloco reúne, em uma única tabela de leitura executiva,
os principais pressupostos estatísticos e normativos verificados
ao longo da modelagem regressiva. A consolidação final facilita
a interpretação técnica do ajuste e aproxima a saída do formato
de relatório exigido no treinamento orientado à NBR 14653.

Objetivos do bloco:
- reunir poder explicativo, significância global, normalidade e independência;
- consolidar métricas, valores obtidos, critérios e status;
- padronizar a apresentação final dos diagnósticos;
- facilitar exportação para relatório técnico e apostila;
- encerrar a aula com um quadro sintético de aprovação normativa.
"""

r2_ajustado = float(ols_results.rsquared_adj)
p_valor_f = float(ols_results.f_pvalue)
estatistica_dw = float(durbin_watson)
p_valor_shapiro_final = float(p_valor_shapiro)

diagnosticos_normativos_df = pd.DataFrame({
    "pressuposto_normativo_estatistico": [
        "Poder Explicativo",
        "Significancia Global",
        "Normalidade dos Residuos",
        "Independencia de Residuos",
    ],
    "metrica_teste_executado": [
        "R² Ajustado",
        "Teste F (p-valor)",
        "Shapiro-Wilk (p-valor)",
        "Durbin-Watson",
    ],
    "valor_obtido": [
        r2_ajustado,
        p_valor_f,
        p_valor_shapiro_final,
        estatistica_dw,
    ],
    "criterio_aceitacao": [
        "R² Ajustado > 0.70",
        "p-valor (F) < 0.05",
        "p-valor (W) >= 0.05",
        "Proximo de 2.0 (1.5 a 2.5)",
    ],
    "status": [
        "APROVADO" if r2_ajustado > 0.70 else "REPROVADO",
        "APROVADO" if p_valor_f < 0.05 else "REPROVADO",
        "NORMAL" if p_valor_shapiro_final >= 0.05 else "NAO_NORMAL",
        "APROVADO" if 1.5 <= estatistica_dw <= 2.5 else "REPROVADO",
    ],
})

display(diagnosticos_normativos_df.round(6))


"""
Bloco 18 - Exportação dos resultados finais em CSV.

Este bloco grava, na pasta de saída do projeto, os principais
artefatos tabulares produzidos ao longo da aula de modelagem:
coeficientes, amostra com resíduos, resumo dos resíduos e quadro
consolidado de diagnósticos normativos.

Objetivos do bloco:
- criar a pasta data/output caso ela ainda não exista;
- exportar tabelas finais em formato CSV;
- padronizar nomes de arquivos para uso didático;
- facilitar reaproveitamento em relatórios e apostilas;
- encerrar a aula com artefatos persistidos em disco.
"""

project_root = resolve_project_root()
output_dir = project_root / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)

caminho_coeficientes = output_dir / "aula_03_coeficientes_significancia_individual.csv"
caminho_predicoes_residuos = output_dir / "aula_03_predicoes_residuos.csv"
caminho_resumo_residuos = output_dir / "aula_03_resumo_residuos.csv"
caminho_diagnostico_normalidade = output_dir / "aula_03_diagnostico_normalidade.csv"
caminho_diagnostico_dw = output_dir / "aula_03_diagnostico_durbin_watson.csv"
caminho_diagnosticos_normativos = output_dir / "aula_03_diagnosticos_normativos.csv"

coeficientes_relatorio_df.to_csv(caminho_coeficientes, index=False, encoding="utf-8-sig")
amostra_com_residuos_df.to_csv(caminho_predicoes_residuos, index=False, encoding="utf-8-sig")
resumo_residuos_df.to_csv(caminho_resumo_residuos, index=False, encoding="utf-8-sig")
diagnostico_normalidade_df.to_csv(caminho_diagnostico_normalidade, index=False, encoding="utf-8-sig")
diagnostico_dw_df.to_csv(caminho_diagnostico_dw, index=False, encoding="utf-8-sig")
diagnosticos_normativos_df.to_csv(caminho_diagnosticos_normativos, index=False, encoding="utf-8-sig")

print("Arquivos CSV exportados com sucesso:")
print(f"- {caminho_coeficientes}")
print(f"- {caminho_predicoes_residuos}")
print(f"- {caminho_resumo_residuos}")
print(f"- {caminho_diagnostico_normalidade}")
print(f"- {caminho_diagnostico_dw}")
print(f"- {caminho_diagnosticos_normativos}")


"""
Bloco 19 - Exportação em HTML e Markdown do relatório final.

Este bloco consolida os principais resultados da aula em dois
formatos de comunicação técnica: Markdown, útil para apostilas
e versionamento, e HTML, útil para visualização direta em navegador
ou composição de relatórios executivos.

Objetivos do bloco:
- gerar um relatório final em Markdown;
- gerar um relatório final em HTML;
- reutilizar os quadros já calculados ao longo da aula;
- facilitar distribuição dos resultados para leitura humana;
- encerrar o fluxo com saídas textuais prontas para documentação.
"""

relatorio_md_path = output_dir / "relatorio_final_treinamento_inferencial.md"
relatorio_html_path = output_dir / "relatorio_final_treinamento_inferencial.html"

equacao_modelo_relatorio = "Não disponível"
if "equation" in globals():
    equacao_modelo_relatorio = equation

markdown_relatorio = f"""# Relatório Final - Treinamento Inferencial NBR 14653

## Visão geral do modelo
- Equação estimada: {equacao_modelo_relatorio}
- R² ajustado: {r2_ajustado:.6f}
- p-valor do teste F: {p_valor_f:.6f}
- p-valor do teste Shapiro-Wilk: {p_valor_shapiro_final:.6f}
- Estatística Durbin-Watson: {estatistica_dw:.6f}

## Diagnósticos normativos
{diagnosticos_normativos_df.round(6).to_markdown(index=False)}

## Coeficientes e significância individual
{coeficientes_relatorio_df.round(6).to_markdown(index=False)}

## Resumo dos resíduos
{resumo_residuos_df.round(6).to_markdown(index=False)}

## Diagnóstico de normalidade
{diagnostico_normalidade_df.round(6).to_markdown(index=False)}

## Diagnóstico de independência serial
{diagnostico_dw_df.round(6).to_markdown(index=False)}
"""

html_relatorio = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <title>Relatório Final - Treinamento Inferencial NBR 14653</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            color: #1f2937;
            background: #ffffff;
        }}
        h1, h2 {{
            color: #1f3a5f;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 24px;
        }}
        th, td {{
            border: 1px solid #d1d5db;
            padding: 8px 10px;
            text-align: left;
            font-size: 14px;
        }}
        th {{
            background: #e5e7eb;
        }}
        .bloco-metrico {{
            margin-bottom: 24px;
            padding: 16px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <h1>Relatório Final - Treinamento Inferencial NBR 14653</h1>

    <div class="bloco-metrico">
        <h2>Visão geral do modelo</h2>
        <p><strong>Equação estimada:</strong> {equacao_modelo_relatorio}</p>
        <p><strong>R² ajustado:</strong> {r2_ajustado:.6f}</p>
        <p><strong>p-valor do teste F:</strong> {p_valor_f:.6f}</p>
        <p><strong>p-valor do teste Shapiro-Wilk:</strong> {p_valor_shapiro_final:.6f}</p>
        <p><strong>Estatística Durbin-Watson:</strong> {estatistica_dw:.6f}</p>
    </div>

    <h2>Diagnósticos normativos</h2>
    {diagnosticos_normativos_df.round(6).to_html(index=False)}

    <h2>Coeficientes e significância individual</h2>
    {coeficientes_relatorio_df.round(6).to_html(index=False)}

    <h2>Resumo dos resíduos</h2>
    {resumo_residuos_df.round(6).to_html(index=False)}

    <h2>Diagnóstico de normalidade</h2>
    {diagnostico_normalidade_df.round(6).to_html(index=False)}

    <h2>Diagnóstico de independência serial</h2>
    {diagnostico_dw_df.round(6).to_html(index=False)}
</body>
</html>
"""

relatorio_md_path.write_text(markdown_relatorio, encoding="utf-8")
relatorio_html_path.write_text(html_relatorio, encoding="utf-8")

print("Relatórios exportados com sucesso:")
print(f"- {relatorio_md_path}")
print(f"- {relatorio_html_path}")

