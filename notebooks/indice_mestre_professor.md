# Índice mestre — notebooks do professor

Material de apoio para condução do treinamento em **Inferência Estatística Aplicada à Avaliação Imobiliária**, organizado em quatro encontros sequenciais.

## Visão geral do ciclo

O percurso didático foi estruturado para que cada aula resolva um problema específico da análise e, ao mesmo tempo, prepare tecnicamente a aula seguinte:

1. **Aula 1 — Coleta e unitarização**: transforma a base bruta em uma amostra comparável.
2. **Aula 2 — Sanitização por Chauvenet**: depura a massa amostral com critério estatístico.
3. **Aula 3 — Regressão OLS**: ajusta a equação explicativa do preço.
4. **Aula 4 — Diagnósticos e aderência à NBR 14653**: valida tecnicamente o modelo ajustado.

---

## Aula 1 — Coleta e unitarização

**Notebook:** `aula_01_coleta_unitarizacao_professor.ipynb`

**Objetivo da aula**  
Ensinar a sair da base bruta de mercado para uma amostra comparável, introduzindo o cálculo do valor unitário em R$/m² como primeira condição para inferência confiável.

**Problema que esta aula resolve**  
Preço absoluto, sozinho, não permite comparação justa entre imóveis de tamanhos diferentes.

**Entrega didática principal**  
A turma termina a aula com uma base lida, validada e unitarizada, pronta para leitura estatística inicial.

**Mensagem central para insistir**  
“A unitarização não encerra a análise, mas cria a escala mínima de comparabilidade.”

**Fala de abertura de 1 minuto**  
“Hoje nós vamos começar pela base de toda a análise inferencial em avaliação imobiliária: tornar comparáveis dados que, na forma bruta, ainda não são diretamente comparáveis. Quando olhamos apenas o preço total, misturamos tamanho, localização e outras características em um único número. Por isso, nosso primeiro passo é organizar a base e calcular o valor unitário, criando uma escala comum de leitura. Essa etapa parece simples, mas é aqui que a análise começa a ganhar consistência técnica.”

**Transição para a próxima aula**  
“Agora que os dados estão em uma escala comparável, precisamos decidir se toda a massa amostral pode permanecer na análise ou se existem observações discrepantes que exigem saneamento.”

---

## Aula 2 — Sanitização amostral por Chauvenet

**Notebook:** `aula_02_sanitizacao_chauvenet_professor.ipynb`

**Objetivo da aula**  
Ensinar que a retirada de observações da amostra só é aceitável quando existe critério estatístico, rastreabilidade e justificativa técnica.

**Problema que esta aula resolve**  
Nem toda observação extrema deve permanecer no conjunto analítico, mas também não pode ser removida por impressão visual ou conveniência do analista.

**Entrega didática principal**  
A turma termina a aula com uma amostra saneada, um histórico de iterações e uma tabela de observações removidas, quando existentes.

**Mensagem central para insistir**  
“Extremo observado não é extremo removível; exclusão exige critério.”

**Fala de abertura de 1 minuto**  
“Na aula passada, nós colocamos os imóveis em uma escala comparável. Hoje, vamos enfrentar outro problema clássico da inferência: nem todo dado que entra em uma amostra contribui de forma adequada para a modelagem. Alguns valores podem estar excessivamente distantes da massa principal e comprometer a leitura estatística. Mas cuidado: isso não significa sair apagando extremos. O que vamos aprender hoje é a diferença entre perceber um valor discrepante e justificar tecnicamente sua remoção com o critério de Chauvenet.”

**Transição para a próxima aula**  
“Com a amostra em escala comparável e depurada com critério estatístico, nós passamos a ter condições melhores para ajustar um modelo de regressão.”

---

## Aula 3 — Regressão linear por OLS

**Notebook:** `aula_03_regressao_ols_professor.ipynb`

**Objetivo da aula**  
Apresentar a regressão linear múltipla por mínimos quadrados ordinários como instrumento para explicar o preço do imóvel a partir de suas características observáveis.

**Problema que esta aula resolve**  
Até aqui, a turma organizou e depurou a base, mas ainda não construiu uma equação capaz de explicar o comportamento do preço.

**Entrega didática principal**  
A turma termina a aula com um modelo ajustado, tabela de coeficientes, resumo do ajuste e resíduos calculados.

**Mensagem central para insistir**  
“Regressão não é uma caixa-preta; ela produz uma equação que precisa ser lida tecnicamente.”

**Fala de abertura de 1 minuto**  
“Depois de organizar os dados e sanear a amostra, chegou a hora de construir um modelo. Nesta aula, vamos ajustar uma regressão linear por mínimos quadrados ordinários para explicar o preço do imóvel com base em variáveis como área, vagas e localização. Mas o foco aqui não é apenas rodar o software. O foco é entender o que significa cada coeficiente, o que o modelo consegue explicar, qual é o tamanho típico do erro e por que os resíduos importam. Em outras palavras: hoje nós saímos da preparação dos dados e entramos na modelagem propriamente dita.”

**Transição para a próxima aula**  
“Agora que a equação foi ajustada, precisamos verificar se ela se sustenta estatisticamente e se atende aos critérios técnicos de validação.”

---

## Aula 4 — Diagnósticos estatísticos e aderência à NBR 14653

**Notebook:** `aula_04_diagnosticos_nbr_professor.ipynb`

**Objetivo da aula**  
Validar o modelo ajustado por meio de critérios de significância, normalidade, independência residual e síntese técnica de aprovação.

**Problema que esta aula resolve**  
Um modelo ajustado pode parecer bom numericamente e ainda assim não satisfazer as condições técnicas mínimas para ser defendido.

**Entrega didática principal**  
A turma termina a aula com um relatório estruturado de diagnósticos, leitura dos coeficientes e uma síntese de aprovação técnica do modelo.

**Mensagem central para insistir**  
“Modelo ajustado não é automaticamente modelo validado.”

**Fala de abertura de 1 minuto**  
“Na aula anterior, nós estimamos uma equação para explicar o preço dos imóveis. Hoje, vamos fazer a pergunta decisiva: essa equação se sustenta tecnicamente? É aqui que entram os diagnósticos estatísticos. Vamos olhar significância global, significância individual, normalidade e independência dos resíduos, além de organizar uma síntese que ajude a comunicar a aderência do modelo aos critérios técnicos. Esta é a aula que fecha o ciclo e transforma cálculo em procedimento defensável.”

**Fechamento do ciclo**  
“Com isso, fechamos a trajetória completa: coletamos a base, tornamos os dados comparáveis, saneamos a amostra, ajustamos a regressão e validamos tecnicamente o modelo.”

---

## Ordem recomendada de uso em sala

1. Abrir o notebook da aula correspondente.
2. Ler a fala de abertura antes de projetar o primeiro bloco técnico.
3. Conduzir as células em sequência, pausando nas perguntas marcadas ao professor.
4. Encerrar com a transição para a aula seguinte.
5. Retomar no encontro seguinte pela frase de transição da aula anterior.

---

## Estratégia geral de condução

- **Turma iniciante:** enfatize interpretação, linguagem simples e exemplos verbais.
- **Turma intermediária:** equilibre interpretação e leitura de saídas estatísticas.
- **Turma avançada:** aprofunde critérios, limitações e discussão de especificação de modelo.

---

## Frases-curinga para o professor

Use estas frases ao longo do curso para manter unidade de discurso:

- “Primeiro tornamos os dados comparáveis; depois decidimos o que permanece na amostra.”
- “Toda exclusão precisa ser justificável e documentada.”
- “Uma equação ajustada não dispensa interpretação técnica.”
- “Resíduo não é sobra irrelevante; é parte da validação do modelo.”
- “Indicador estatístico orienta decisão, mas não substitui juízo técnico.”

---

## Checklist rápido antes de cada aula

- conferir se o ambiente Jupyter está funcionando;
- verificar se a base de dados está na pasta `data/`;
- abrir o notebook professor correto;
- revisar a fala de abertura de 1 minuto;
- definir se a aula terá foco mais conceitual ou mais operacional.
