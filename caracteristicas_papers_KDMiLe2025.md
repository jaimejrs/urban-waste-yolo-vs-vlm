# Características de papers aceitos — KDMiLe / SBBD 2025

*Análise de 10 artigos publicados nos anais do evento (Fortaleza, outubro de 2025), com o objetivo de identificar padrões recorrentes que favorecem a aceitação.*

---

## O veículo: KDMiLe (confirmado)

O alvo é o **KDMiLe 2025** (*Symposium on Knowledge Discovery, Mining and Learning* — "Procs. of the 13rd Symp. on Knowledge Discovery, Mining and Learning"). Isso importa para ler a análise corretamente, porque **9 dos 10 artigos enviados são justamente do KDMiLe**; o único de fora é o DENETHOR (proveniência em computação serverless), publicado nos **Anais do XL Simpósio Brasileiro de Bancos de Dados (SBBD)**.

KDMiLe e SBBD acontecem na mesma semana, na mesma cidade, sob a Sociedade Brasileira de Computação — daí a confusão comum —, mas têm escopos diferentes: o SBBD gira em torno de bancos de dados/sistemas de dados, e o KDMiLe em torno de mineração de dados, aprendizado de máquina e aplicações. Para o seu paper, portanto:

- **Os 9 artigos do KDMiLe são o seu gabarito principal.** Eles são o arquétipo do que esse veículo aceita — estudos aplicados de ML, com comparação e insight prático —, exatamente o formato do seu TCC.
- **O DENETHOR é de outro veículo e de outro tipo** (paper de *sistema/ferramenta* com forte componente de banco de dados). Não o use como molde de estrutura ou de contribuição. Ainda assim, as normas transversais que ele ilustra — reprodutibilidade, agradecimento a fomento, posicionamento claro frente à literatura — valem para os dois veículos, porque são padrões da SBC como um todo. Por isso ele ainda aparece como evidência em alguns pontos abaixo, mas sempre nesse papel de norma compartilhada, não de template a seguir.

Detalhe prático que decorre da confirmação: o KDMiLe **aceita submissões em português** (o paper sobre Concept Drift na Petrobras está inteiramente em português) e o formato é de *short paper* (6 a 8 páginas no padrão dos anais). Isso condiciona o tipo e o tamanho da contribuição esperada — e favorece contribuições focadas, como a sua comparação YOLO × VLM.

---

## Síntese: o que realmente diferencia estes papers

Antes da lista detalhada, três observações de fundo que valem mais do que qualquer checklist:

1. **A contribuição central quase nunca é um algoritmo novo.** Nenhum dos 10 papers propõe uma arquitetura ou método fundamentalmente inédito. Eles aplicam métodos já existentes (LightGBM, YOLO, BERT, Isolation Forest, Random Forest, J48…) a um problema concreto e extraem dali um *insight acionável*. A novidade está na **combinação problema + dados + comparação + conclusão prática**, não na matemática. Isso é libertador para um TCC: você não precisa inventar uma rede neural nova — comparar YOLO com VLM e caracterizar quando cada um vence já é, por si só, uma contribuição válida nesse veículo.

2. **O enquadramento em "trade-off" é o padrão dominante e o mais valorizado.** Os papers mais sólidos não afirmam "o método X é o melhor"; eles afirmam "X é melhor *em acurácia*, mas Y é melhor *em custo/robustez/velocidade*, então a escolha depende do cenário". Essa nuance é recompensada — ela demonstra maturidade analítica e utilidade prática.

3. **Honestidade sobre limitações e resultados negativos fortalece, não enfraquece.** Vários papers foram aceitos *apesar de* (ou justamente *por*) reportarem que algo não funcionou. O rigor em mostrar e explicar o fracasso é tratado como sinal de qualidade científica.

A seguir, as características concretas, cada uma com os exemplos dos papers analisados.

---

## 1. Problema bem delimitado e com relevância prática ou social

Todos os papers abrem ancorando-se em um problema concreto, frequentemente com peso social ou econômico, e muitas vezes brasileiro:

- **Previsão de vazão em redes 5G** → gestão proativa de QoS para serviços sensíveis a latência.
- **Detecção de vazamento de água** → o paper abre com o dado do Instituto Trata Brasil de que ~37,78% da água tratada é perdida na distribuição.
- **Predição de mortalidade infantil** → saúde pública, com a taxa brasileira de 2023 (12,51 óbitos/1.000 nascidos vivos) como gancho.
- **Concept drift em logs do SLURM** → otimização de HPC em um caso real da Petrobras (setor de óleo e gás).
- **Perfis socioeconômicos via detecção de comunidades** → apoio à alocação de bolsas estudantis em uma universidade federal.
- **Viés de gênero em anotação** → fairness em NLP, agravado pela riqueza de gênero gramatical do português.

**Lição:** dedique o primeiro parágrafo a estabelecer *por que o problema importa*, de preferência com um número ou fato concreto. Resíduos urbanos em vias públicas é um problema de gestão pública com forte apelo — explore isso (custo de fiscalização manual, escala do monitoramento, etc.).

---

## 2. A comparação e o insight são a contribuição — não um método inédito

A esmagadora maioria estrutura-se como um **estudo comparativo controlado** entre abordagens:

- Previsão 5G: estatístico × ML × deep learning (AutoARIMA, LightGBM, Random Forest, N-BEATS, Block RNN, Transformer…).
- Anotação de gênero: SVM, NB, RF, DT e LR treinados separadamente.
- Mortalidade infantil: DT, LR, NB e XGBoost.
- Perfis socioeconômicos: 3 modelos de grafo × 3 métodos de detecção de comunidades.
- DNA: GPT-2, BERT e DNABERT.
- Lesão tibial: CNN, VGG19, LWCNN + 6 baselines de ML clássico.
- Vazamento de água: Isolation Forest × Elliptic Envelope.

A pergunta de pesquisa é tipicamente *"qual abordagem é mais adequada para esta tarefa e sob quais condições?"* — exatamente o seu caso (YOLOv11m × Gemma 3). O valor está na **comparação justa e bem controlada**, não na originalidade dos modelos.

---

## 3. Enquadramento em "trade-off" (o padrão mais comum e mais premiado)

Os papers de maior qualidade recusam o veredito simplista "X vence" e entregam um mapa de compromissos:

- **Previsão 5G:** o AutoARIMA tem o *menor* RMSE mediano (3,49), mas o *pior* RMSE máximo (108,88) — ou seja, é frágil. O LightGBM perde por pouco na mediana, mas é muito mais robusto a outliers *e* treina em 1,65 s. A conclusão não é "use AutoARIMA", e sim "o LightGBM oferece o melhor equilíbrio entre acurácia, robustez e eficiência". A própria Figura 1 cruza **RMSE × tempo de treino**.
- **Vazamento de água:** o Elliptic Envelope tem o melhor F1, mas treino mais lento (até 20,46 s); o Isolation Forest treina rápido (0,53 s), mas com inferência que cresce com o volume. A recomendação depende do cenário de uso (tempo real × custo de treino).
- **DNA:** BERT vence em acurácia (0,9905); DNABERT é o mais rápido (3 min/época) mas menos preciso; GPT-2 é competitivo porém o mais caro.
- **Lesão tibial:** VGG19 vence em F-Score (0,894), mas exige 4 h de treino em CPU; os modelos de ML clássico rodam em minutos e ainda oferecem interpretabilidade.

**Lição direta para você:** seu achado de que o Gemma é mais acurado (0,92 vs 0,71) enquanto o YOLO é ~141× mais rápido é *exatamente* esse tipo de trade-off. Construa o paper em torno dele — um gráfico acurácia × latência/custo é praticamente obrigatório nesse veículo.

---

## 4. Uso de dados reais, preferencialmente de domínio ou brasileiros

Dados sintéticos puros são exceção; quase todos usam dados reais, muitos brasileiros:

- Dataset 5G de Raca et al. (83 traces, condições reais de mobilidade).
- DataSUS — SINASC (31,3 mi de registros) e SIM (382 mil), via *Record Linkage*.
- MQD-1465/1174 — sentenças reais de diário em português brasileiro.
- Logs reais do SLURM da Petrobras (6 mi+ de registros, 300+ engenheiros).
- Dados reais de bolsas de uma universidade federal (3 anos).
- Imagens térmicas reais do Hospital Universitário Federal do Paraná.
- Dados próprios coletados por IoT + datasets públicos (DAIAD, Di Mauro).

Quando o dado é proprietário ou sintético, os autores **justificam explicitamente** e contextualizam a limitação. O paper do gerador causal (CSDG), por ser sobre dados sintéticos, dedica espaço a defender por que isso é legítimo (avaliar inferência causal exige contrafactuais conhecidos, indisponíveis em dados reais).

**Você já está bem posicionado aqui:** 560 imagens de ruas brasileiras (majoritariamente Google Street View) é um dataset real e de domínio. Descreva a coleta, a curadoria e os critérios com o mesmo cuidado dos papers (eles detalham origem, tamanho, balanceamento e tratamento).

---

## 5. Rigor metodológico e protocolo de validação apropriado

Há um cuidado consistente com protocolos que evitam vazamento de dados e respeitam a natureza do problema:

- **Validação walk-forward** em séries temporais (previsão 5G e Concept Drift), respeitando a ordem temporal.
- **Teste ADF** para estacionariedade antes de modelar (previsão 5G).
- **SMOTE** aplicado *apenas no conjunto de treino* para classes desbalanceadas (mortalidade infantil) — com a ressalva explícita de não contaminar o teste.
- **Validação cruzada** (5-fold em gênero, 10-fold em Concept Drift) e **holdout estratificado repetido com 10 seeds** (lesão tibial).
- **Múltiplas execuções com seeds diferentes** para robustez (DNA: 3 execuções; lesão tibial: 10).
- Split temporal 80/20 com inferência simulando "o futuro" (vazamento de água).

**Lição:** descreva o protocolo de avaliação com precisão suficiente para reprodução. No seu caso, o limiar fixo do YOLO em 0,25 *definido a priori*, a remoção do mAP como métrica não comparável e os 15% de negativos de fundo são exatamente o tipo de decisão metodológica defensável que um comitê valoriza — explicite a justificativa de cada uma.

---

## 6. Métricas adequadas + teste de significância estatística

A escolha de métricas é justificada pela natureza da tarefa, e os melhores papers vão além de relatar números: testam se as diferenças são estatisticamente significativas.

- Previsão: MAE e RMSE (e a justificativa de por que o mAP/médias simples enganam).
- Classificação: acurácia, precisão, recall e F1 — com a ressalva recorrente de que *acurácia alta engana sob desbalanceamento* (mortalidade infantil tem 99% de acurácia mas F1 máximo de 0,44).
- Concordância: Cohen's κ, κ ponderado, χ², Cramér's V e entropia de Shannon (anotação de gênero).
- **Significância:** teste de **Wilcoxon** (α = 0,05) para comparar distribuições de F-Score (lesão tibial); χ² com p-valores reportados (gênero).

**Você já faz isso:** o **teste de McNemar** confirmando a significância da diferença entre YOLO e Gemma é precisamente o ingrediente que separa um "X teve número maior" de uma conclusão defensável. Destaque-o.

---

## 7. Reprodutibilidade tratada como requisito de primeira ordem

Reprodutibilidade não é um detalhe ao final — é parte da contribuição. Vários papers publicam repositório:

- `github.com/ejs94/5g-forecasting` (previsão 5G)
- `github.com/angeruzzi/causal-synthetic-data-gen` (gerador causal)
- `github.com/UFFeScience/denethor` (DENETHOR)
- `github.com/ic2d/tibiaInjuryDetection` (lesão tibial)
- Datasets no Kaggle (vazamento de água) e DOI para o dataset (perfis socioeconômicos)

Além do código, há **tabelas explícitas de hiperparâmetros e setup** (a Tabela I do paper de lesão tibial lista *tudo*: resampling, bibliotecas, otimizador, épocas, batch size, critério de early stopping, número de seeds).

**Lição:** você já tem o repositório (`github.com/jaimejrs/comparativo_yolo_vlm`) e o notebook (`comparativo_lixo1.ipynb`). Inclua o link no paper e adicione uma tabela de configuração reproduzindo o nível de detalhe acima (versão do YOLO, limiar, datasets de treino TACO/Trash Detection, hardware RTX 5090, etc.).

---

## 8. Honestidade sobre limitações e resultados negativos

Talvez o padrão mais contraintuitivo: papers foram aceitos **reportando que algo não funcionou**, porque o fizeram com rigor e análise das causas.

- **Previsão 5G:** o achado secundário central é que *as covariáveis de qualidade de rede foram insuficientes* para melhorar consistentemente a previsão — um resultado negativo, reportado e discutido (atribuído à necessidade de features de contexto mais ricas).
- **Mortalidade infantil:** o paper inteiro conclui que os modelos *não* conseguiram prever bem o evento raro (precisão < 0,5), e analisa por quê (SMOTE insuficiente, prevalência de 0,81%, baixo poder discriminativo das variáveis).
- **Anotação de gênero:** o Decision Tree apresentou alinhamento fraco (κ = 0,3838), reportado abertamente como contraste.
- **DNA:** o DNABERT, *apesar* de ser especializado para DNA, teve desempenho inferior — e os autores explicam o porquê (tokenização k-mer fixa limita dependências de longo alcance).

Todos delimitam limitações e **trabalhos futuros** de forma específica e concreta (não genérica). O paper de gênero, com apenas 8 anotadores, se reposiciona honestamente como "análise de caso fundacional".

**Lição:** seu YOLO tem o que você chamou de "cegueira específica de domínio" e o Gemma tem "sobre-sinalização contextual". Essa análise qualitativa de *erros* é ouro — é o tipo de discussão que demonstra que você entendeu o comportamento dos modelos, não só mediu acurácia.

---

## 9. Posicionamento explícito frente à literatura (a "lacuna")

Toda seção de Trabalhos Relacionados/Correlatos faz duas coisas: (a) resume o estado da arte e (b) **declara explicitamente a lacuna** que o paper preenche e como se diferencia.

- O DENETHOR diz que abordagens existentes capturam proveniência serverless, mas "nenhuma mira as necessidades específicas de aplicações CSE" — e é isso que ele resolve.
- O paper de água afirma combinar características de trabalhos anteriores, mas usando **aprendizado não supervisionado** (sem rótulos), diferenciando-se dos supervisionados citados.
- O paper de DNA aponta que DNABERT/DNAGPT focam em tarefas gerais, "sem mirar diretamente a classificação íntron/éxon".

**Lição:** mapeie os trabalhos que aplicam visão computacional a resíduos urbanos (e os que comparam detectores objeto-específicos com VLMs) e deixe claro o que o seu faz de diferente — por exemplo, a comparação direta YOLO × VLM em cenário brasileiro real com análise de custo de implantação.

---

## 10. Estrutura canônica e perguntas de pesquisa explícitas

Todos seguem a mesma espinha dorsal: **Resumo → Introdução (com objetivos) → Trabalhos Relacionados → Metodologia → Resultados/Discussão → Conclusão e Trabalhos Futuros → Referências**, com **CCS Concepts** e **Keywords** logo após o resumo.

Os mais fortes tornam as **perguntas de pesquisa explícitas**:
- Previsão 5G: "RQ1: Quais métodos são mais eficazes…? RQ2: Quais as vantagens de modelos locais vs globais?" — e a Conclusão responde *uma a uma*, citando a RQ.
- Concept Drift: lista 3 objetivos numerados e os responde.

**Lição:** declare 1–2 perguntas de pesquisa na introdução (ex.: *"RQ1: Em detecção binária de resíduos em vias públicas, qual o trade-off entre acurácia e custo computacional de um detector objeto-específico (YOLOv11m) e um VLM (Gemma 3 27B-IT)?"*) e estruture a conclusão respondendo-as diretamente. Para um comitê de Administração, isso ancora a narrativa gerencial.

---

## 11. Análise qualitativa complementando a quantitativa

Os papers de maior densidade não param na tabela de métricas — adicionam uma camada interpretativa:

- **Lesão tibial:** matrizes de confusão dos top-3 modelos + análise de *importância de features* do Random Forest, mostrando que canal vermelho, saturação e textura (LBP) explicam a classificação — e conectando isso à fisiologia (fluxo sanguíneo, inflamação). Ou seja, o paper *explica o porquê*, não só o quê.
- **Anotação de gênero:** matrizes de contingência e exemplos concretos de sentenças classificadas de forma divergente por gênero.
- **Perfis socioeconômicos:** uma "visão mesoscópica" das comunidades, traduzindo clusters em perfis socioeconômicos interpretáveis (quem mora de aluguel, quem usa transporte público, etc.).

**Lição:** sua análise de erros (sobre-sinalização contextual do Gemma vs cegueira de domínio do YOLO) é exatamente essa camada. Ilustre com exemplos visuais de imagens onde cada modelo acerta/erra — isso é especialmente persuasivo para um público não técnico.

---

## 12. Ética e financiamento

Quando o dado envolve seres humanos, há **aprovação ética explícita**:
- Gênero: protocolo CAAE 82267824.8.0000.5289 + Termo de Consentimento.
- Lesão tibial: Resolução 466/12 + protocolos dos Comitês de Ética da UTFPR e do HC-UFPR.
- Mortalidade infantil: dados anonimizados do DataSUS.

E quase todos trazem **Agradecimentos a agências de fomento** (FAPESP, CNPq, CAPES, FAPEMIG, FAPERJ) — sinal de respaldo institucional.

**Lição:** se seu dataset usa Google Street View, vale uma frase sobre os termos de uso/anonimização (rostos, placas). E inclua agradecimentos à sua instituição/orientação.

---

## Checklist para a sua submissão

Use como verificação final antes de submeter:

- [ ] **Veículo confirmado** (KDMiLe vs SBBD) e formato/limite de páginas respeitado.
- [ ] Introdução abre com a **relevância prática** do problema (número/fato concreto).
- [ ] **Pergunta(s) de pesquisa explícita(s)** declaradas e respondidas na conclusão.
- [ ] Contribuição enquadrada como **comparação/insight** (não como algoritmo inédito).
- [ ] **Trade-off** central destacado, idealmente com um gráfico (acurácia × custo/latência).
- [ ] **Dados reais** descritos com rigor (origem, tamanho, balanceamento, curadoria).
- [ ] **Protocolo de avaliação** sem vazamento, descrito para reprodução.
- [ ] **Métricas justificadas** + **teste de significância** (você tem o McNemar — destaque-o).
- [ ] **Reprodutibilidade**: link do repositório + tabela de hiperparâmetros/setup.
- [ ] **Limitações e resultados negativos** reportados com honestidade e análise de causa.
- [ ] **Análise qualitativa de erros** complementando os números.
- [ ] **Trabalhos relacionados** posicionam a lacuna e a diferenciação.
- [ ] **Ética** (uso de imagens/Street View) e **agradecimentos** incluídos.
- [ ] Estrutura canônica + **CCS Concepts** e **Keywords**.

---

## Como isso conversa com o seu trabalho

Seu TCC (comparação YOLOv11m × Gemma 3 27B-IT para detecção binária de resíduos urbanos em vias públicas brasileiras) encaixa-se notavelmente bem no perfil dos papers aceitos:

- **Comparação como contribuição** → como o paper de previsão 5G (estatística × DL) ou o de DNA (3 transformers).
- **Trade-off explícito** (Gemma mais acurado, 0,92 vs 0,71; YOLO ~141× mais rápido) → como o eixo RMSE × tempo de treino do paper 5G e o F1 × tempo de inferência do paper de água.
- **Dados reais brasileiros** (560 imagens de ruas) → na mesma linha de DataSUS, Petrobras e português brasileiro.
- **Significância estatística** (McNemar) → como o Wilcoxon do paper de lesão tibial.
- **Análise qualitativa de erros** (sobre-sinalização contextual vs cegueira de domínio) → como a análise de importância de features do paper de lesão tibial.
- **Análise de custo de implantação** → reforça a narrativa gerencial, alinhada ao apelo prático que todos os papers buscam, e é um diferencial seu.
- **Reprodutibilidade** (repositório no GitHub) → padrão recorrente nos aceitos.

Os dois pontos onde vale investir mais atenção, à luz desta análise, são: (1) tornar a(s) **pergunta(s) de pesquisa explícita(s)** e respondê-la(s) na conclusão; e (2) garantir que o **trade-off acurácia × custo** apareça em um gráfico claro — esse é o "produto" que o veículo mais recompensa.
