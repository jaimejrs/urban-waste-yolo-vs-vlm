# Plano estratégico de aprovação — KDMiLe 2026
### Comparação YOLOv11m × VLM para detecção de resíduos em vias públicas · janela de 8 dias de dedicação exclusiva

> Documento-companheiro da auditoria (`auditoria_KDMiLe2026_yolo_vs_gemma.md`). A auditoria diz **o que está errado e como corrigir**; este plano diz **em que ordem fazer, em quanto tempo, e com base em quê** — ancorado em papers realmente aceitos no KDMiLe de 2021 a 2025. Onde a correção já foi detalhada na auditoria, aqui eu apenas referencio (ex.: "ver A1") para não repetir.

---

## 0. A mensagem que vale mais que o resto do documento

Você **já está dentro do perfil** que o KDMiLe aceita (vou provar isso na Seção 1 com papers reais). A taxa de aceitação recente é de **~39–43%** — não é um veículo hiperseletivo. Isso muda completamente a natureza dos seus 8 dias:

**Estes 8 dias NÃO são para fazer mais ciência. São para empacotar e posicionar melhor a ciência que você já tem.**

O maior risco do seu projeto agora **não** é "falta de experimento" — é você gastar a janela inteira numa ideia nova e brilhante (trocar para o Gemma 4, fazer um mAP completo de localização) que (a) desestabiliza um resultado que hoje está limpo e verificado, (b) estoura o limite de 8 páginas, e (c) consome o tempo que deveria ir para o que de fato decide aceitação: tirar do `.odt`, matar a objeção da comparação confundida, marcar a diferença frente ao Malla, fechar a reprodutibilidade e produzir a figura de trade-off. Guarde a ambição: os melhores papers do KDMiLe são convidados a estender no **JIDM** (política mantida em todas as edições) — é lá que o Gemma 4 e a localização completa têm casa.

Lema dos 8 dias: **resista à vontade de fazer mais; faça melhor o que está pronto.**

---

## 1. O que o KDMiLe realmente aceita — evidência, não achismo

### 1.1 As regras são estáveis e a barra é alcançável

As regras são praticamente idênticas desde pelo menos 2021 (verificado nas páginas oficiais de 2021, 2023, 2024 e 2026): corpo em português ou inglês mas **título, abstract e keywords em inglês**; **single-blind**; **máximo 8 páginas, com rejeição automática sem revisão se exceder**; **somente PDF** ("formatos diferentes de PDF NÃO serão aceitos"); template LaTeX oficial; e o convite para versão estendida no JIDM para os melhores trabalhos.

Taxas de aceitação recentes (fonte: anais no SOL/SBC):

| Edição | Local / co-evento | Aceitos / submetidos | Taxa |
| --- | --- | --- | --- |
| XII KDMiLe 2024 | Belém, com BRACIS | 21 / 54 | **38,9%** |
| XIII KDMiLe 2025 | Fortaleza, com SBBD | 19 / 44 | **43,2%** |
| XIV KDMiLe 2026 | Cuiabá, com BRACIS | — (prazo 22/06/2026) | — |

*(Não localizei a taxa exata de 2022; as de 2024–2025 e a estabilidade das regras desde 2021 já dão o quadro. ~2 em cada 5 trabalhos entram.)*

### 1.2 O "DNA" recorrente dos aceitos (e como você se encaixa)

Cruzando os 10 papers da sua análise prévia (majoritariamente 2025) com os que verifiquei de 2023–2024, o padrão é consistente:

1. **Contribuição = comparação aplicada + insight prático, não método novo.** Praticamente nenhum aceito propõe arquitetura inédita.
2. **Dados reais, de preferência brasileiros.**
3. **Enquadramento em trade-off** (acurácia × custo/velocidade/robustez).
4. **Protocolo de validação sério + teste de significância.**
5. **Honestidade sobre limitações e resultados negativos.**
6. **Reprodutibilidade** (repositório + tabela de configuração).
7. **Perguntas de pesquisa explícitas, respondidas na conclusão.**
8. **Estrutura canônica + CCS Concepts + keywords + título/abstract/keywords em inglês.**

Seu paper já entrega 1, 2, 3, 4, 5, 7 (parcial) e 8. Os furos reais são **formato (bloqueador)**, **enquadramento da comparação**, **posicionamento de novidade** e **reprodutibilidade (prompt + repo)** — todos de empacotamento.

### 1.3 Precedentes reais que você deve estudar e citar

Estes existem nos anais (verificados no SOL) e servem de **molde** e de **munição de citação**:

**(a) O precedente mais valioso — do grupo do seu orientador:**
- **Almeida, F. C.; Caminha, C.** *Evaluation of Entry-Level Open-Source Large Language Models for Information Extraction from Digitized Documents.* KDMiLe 2024 (XII), Belém. (SOL: `/index.php/kdmile/article/view/30944`).
- Por que importa: é **o mesmo veículo, o mesmo grupo (Caminha é coautor do seu paper), e o mesmo gênero** — avaliação de modelos generativos open-source (7–14B) numa tarefa prática, com dados reais. **Use a estrutura, o tom e a forma de enquadrar contribuição desse paper como gabarito do seu.** Seu orientador já sabe o que esse comitê recompensa — explore isso nas revisões internas.

**(b) Precedentes de visão computacional com o padrão "comparação + trade-off":**
- **Marcato, B. U. et al.** *Automatic Area Estimation of Mice Wound Images.* KDMiLe 2024. Testa a hipótese de que **thresholding clássico pode ser tão acurado quanto DL** em segmentação — exatamente o espírito "método simples × método pesado, qual compensa". (SOL: `/index.php/kdmile/article/download/30941/30744/`).
- *Detecção de Lesão Tibial em imagens térmicas* (KDMiLe 2025, da sua análise prévia): CNN/VGG19 × ML clássico, com **Wilcoxon** e análise de importância de features. É o seu paralelo mais próximo em rigor estatístico e análise qualitativa.
- **Ferreira, B. V. et al.** *CNN-DFT Based Approach Applied to Image Inspection of Railcar Component: A Comparison with Machine Learning Methods.* JIDM (extensão de KDMiLe). Comparação CNN × ML clássico em inspeção por imagem.

**(c) Precedentes do padrão "estudo comparativo + dado brasileiro" (fora de visão, mas mesmo molde):**
- *A Comparative Study of BERT Models for Semantic Retrieval of Brazilian Legal Precedents* (KDMiLe 2024).
- Valeriano, M. G.; Kiffer, C. R. V.; Lorena, A. C. *Improving models performance in a data-centric approach applied to the healthcare domain* (KDMiLe 2024; DOI 10.5753/kdmile.2024.244519).
- Augusto, H. T. B. V. et al. *Unraveling Emotional Dimensions in Brazilian Portuguese Speech through Deep Learning* (KDMiLe 2024; DOI 10.5753/kdmile.2024.243865).
- Silva, R. A. et al. (UFC) *Memory Error Driven Server Failure Detection* (KDMiLe 2024; DOI 10.5753/kdmile.2024.244764).
- *Data stratification analysis on the propagation of discriminatory effects in binary classification* (KDMiLe 2023).

> **Ação:** confirme as strings de citação e DOIs exatos direto no SOL (`sol.sbc.org.br/index.php/kdmile`) antes de colocar no `.bib`. Não copie daqui sem conferir.

---

## 2. Diagnóstico rápido: seu paper × o perfil aceito

| Critério do perfil | Seu paper hoje | Situação |
| --- | --- | --- |
| Comparação aplicada como contribuição | YOLOv11m × Gemma 3 zero-shot | ✅ Forte |
| Dado real brasileiro | 560 imagens de ruas BR | ✅ Forte |
| Trade-off | Acurácia (Gemma) × velocidade (YOLO ~142×) | ✅ Forte (mas falta a **figura**) |
| Validação + significância | pHash anti-vazamento, Wilson, **McNemar χ²=74,40 (recalculei: correto)** | ✅ Acima da média |
| Honestidade/limitações | §5 com 6 ameaças à validade | ✅ Forte |
| Reprodutibilidade | Tabela 2 boa; **prompt ausente; repo não acessível** | ⚠️ Furo corrigível |
| RQs explícitas respondidas na conclusão | Existem, mas RQ1 induz objeção | ⚠️ Reenquadrar (A1) |
| Estrutura + CCS + EN | OK; **mas está em `.odt`** | ⛔ Bloqueador de formato |
| Posicionamento de novidade | Trade-off se sobrepõe ao Malla | ⚠️ Diferenciar (A2) |

Tradução: **um candidato competitivo preso atrás de problemas de empacotamento.** É exatamente o tipo de caso que 8 dias bem gastos resolvem.

---

## 3. Decisões estratégicas a travar no Dia 1 (não reabrir depois)

1. **VLM do experimento principal = Gemma 3 27B-IT, intocado.** **Não** trocar para o Gemma 4 agora. Motivo: trocar obriga a refazer todas as tabelas, McNemar, ICs, latência (o Gemma 4 26B-A4B é MoE, perfil de latência diferente — mexeria no seu "142×") e o prompt, a 8 dias do prazo, em cima de um resultado verificado. Gemma 4 entra em **Trabalhos Futuros** (e é a estrela da versão JIDM).
2. **Experimentos opcionais — no máximo dois, ambos baratos:** (i) **baseline CLIP zero-shot** nas mesmas 560 (recomendado: dá um terceiro ponto de referência e conecta à literatura citada); (ii) **taxa de sucesso de localização** com o Gemma 3 (opcional; ver Seção 6). **Proibido nesta janela:** fine-tuning do YOLO em BR, mAP completo de localização, troca de modelo. Esses são JIDM.
3. **Enquadramento central = comparação de estratégias de implantação**, não de superioridade arquitetural (A1). Decisão de redação, custo quase zero, maior retorno de aceitação.
4. **Repositório público no Dia 1.** Sem isso, a contribuição de reprodutibilidade não existe.

---

## 4. Plano dia a dia (8 dias)

> Sequenciado por **dependência**. Cada dia tem um "pronto quando" (definition of done). Prazo final **22/06/2026, 23:59 AoE (UTC-12)**, via **JEMS 3** (`jems3.sbc.org.br/kdmile2026`). Mire submeter no **Dia 7–8 com folga**, nunca na última hora.

### Dia 1 — Fundação e decisões (sem isso, o resto escorrega)
- Travar as 4 decisões da Seção 3.
- **Portar todo o conteúdo para o template LaTeX oficial do KDMiLe (Overleaf) já hoje** — assim você conhece a paginação real desde o início (B1). Não escreva mais nada em `.odt`.
- Tornar o repositório **público** e criar o esqueleto: `README`, `requirements`/ambiente, script de avaliação, pasta de configs, arquivo com o prompt (A5).
- Ler de ponta a ponta o paper **Almeida & Caminha 2024** e folhear 2 dos precedentes de visão (Marcato; lesão tibial), anotando estrutura e como enquadram contribuição.
- **Pronto quando:** projeto compila em PDF no template (mesmo com furos), você sabe quantas páginas tem, e o repo está público.

### Dia 2 — Experimentos opcionais e congelamento de números
- Rodar o **baseline CLIP zero-shot** nas 560 (Seção 6.1). Registrar acurácia/F1/sensibilidade/especificidade + McNemar contra Gemma e contra YOLO.
- (Opcional) Rodar a **taxa de sucesso de localização** do Gemma 3 (Seção 6.2), em **query separada** para não tocar no experimento binário.
- Extrair e salvar o **prompt verbatim** do Gemma + o esquema JSON + como respostas inválidas foram tratadas e quantas ocorreram (A3).
- **Congelar todos os números** num único arquivo de resultados versionado. A partir daqui, nada de re-rodar sem motivo forte.
- **Pronto quando:** todos os valores que entrarão no paper estão num só lugar, conferidos, e o prompt está documentado.

### Dia 3 — Reenquadramento (o dia de maior impacto na aceitação)
- Reescrever **RQ1 como comparação de estratégias de implantação** e **limitar o claim de acurácia** (texto pronto em A1 da auditoria).
- Reescrever **Abstract** (inglês, 100–300 palavras) e o **gancho de relevância** da Introdução (número concreto: custo de fiscalização manual / escala de monitoramento de descarte irregular).
- Reescrever a lista de **contribuições** alinhada ao molde do Almeida & Caminha.
- **Pronto quando:** abstract, RQs e contribuições não permitem mais que um revisor diga "comparação injusta" sem que o texto já tenha respondido.

### Dia 4 — Trabalhos relacionados e a lacuna
- Inserir o **parágrafo de diferenciação frente a Malla/Funk** (texto pronto em A2): tarefa de *litter/descarte em via pública* × triagem/reciclagem; dado BR in-the-wild; confronto contra detector ajustado + McNemar pareado.
- Acrescentar 1–2 precedentes de **comparação em visão** (Marcato; lesão tibial) para ancorar o gênero "comparação aplicada".
- **Pronto quando:** a seção declara explicitamente o estado da arte e a sua lacuna, nomeando os concorrentes mais próximos.

### Dia 5 — Resultados, discussão e as figuras
- **Figura de trade-off** acurácia/F1 × latência (eixo de latência em escala log; pontos YOLO, Gemma e — se feito — CLIP). É o item que o veículo mais recompensa (A4).
- Integrar o **CLIP** nas Tabelas 3/6 e na discussão.
- **Figura qualitativa** com 2–4 exemplos de acerto/erro de cada modelo (sobre-sinalização do Gemma × cegueira de domínio do YOLO) (M4) — fortíssima para o apelo prático.
- **Contextualizar a latência** ("~142×") como ciclo HTTP+LM Studio já no texto/abstract (M2).
- **Pronto quando:** o trade-off é visual, e a discussão explica *por que* cada modelo erra, não só *quanto*.

### Dia 6 — Metodologia, reprodutibilidade e ética
- Reconciliar a **Tabela 2** com o config real do repositório (versões, limiar 0,25, épocas, otimizador, seeds, quantização Q5_K_M).
- Inserir **prompt do Gemma** + tratamento de saídas inválidas no texto (A3).
- Descrever o **ground truth** (inclusive das bboxes, se você usar a localização) com origem/critério; frase sobre **Street View / anonimização** de rostos e placas (ética).
- Condensar **§5 (ameaças)** para itens enxutos (libera espaço — ver B2).
- **Pronto quando:** um pesquisador externo conseguiria, em tese, reproduzir o pipeline a partir do texto + repo.

### Dia 7 — Caça às páginas e polimento
- Passar o **fio da navalha do limite de 8 páginas** (plano de corte em B2: comprimir §4.5 e §5, enxugar histórico do YOLO/VLM, eventualmente mover a Tabela 5 para a figura).
- Revisar **figuras/tabelas, .bib, CCS Concepts, keywords, título/abstract em inglês**.
- **Re-verificar a consistência de todos os números** (as matrizes de confusão e o McNemar precisam continuar batendo após a edição).
- **Pronto quando:** PDF final ≤ 8 páginas, sem placeholders ("[INSERIR...]") sobrando, números consistentes.

### Dia 8 — Revisão externa, folga e submissão
- Enviar para **coautor/orientador** revisar (idealmente já no fim do Dia 7).
- Última leitura de prova; conferir cabeçalho (e-mails institucionais UFC, não `@gmail`), agradecimentos preenchidos, declaração de não-duplicidade.
- **Dry-run no JEMS 3** (criar submissão, conferir campos, anexar PDF) e **submeter com folga** antes das 23:59 AoE de 22/06.
- **Pronto quando:** submetido, com confirmação do JEMS salva.

> **Folga proposital:** se algo atrasar, sacrifique primeiro a **localização (6.2)**, depois o **CLIP (6.1)**, depois a **figura qualitativa (M4)** — nessa ordem. Os Dias 1, 3, 4, 6 e 7 são inegociáveis; são eles que tiram o paper do desk reject e das objeções letais.

---

## 5. Esqueleto do paper alinhado ao padrão aceito

Estrutura canônica (a mesma de todos os aceitos) com orçamento de páginas sugerido para caber em 8:

| Seção | Conteúdo-chave | Orçamento |
| --- | --- | --- |
| Título + Abstract (EN) + CCS + Keywords (EN) | Gancho de relevância + achado de trade-off; abstract 100–300 palavras | ~0,5 pg |
| 1. Introdução | Problema (número concreto) → lacuna → **RQ1/RQ2 explícitas** → contribuições | ~1 pg |
| 2. Trabalhos Relacionados | Estado da arte + **lacuna nomeando Malla/Funk** + precedentes de comparação | ~1 pg |
| 3. Metodologia | Dados (origem/curadoria/pHash), modelos, **prompt do Gemma**, protocolo, **Tabela de config** | ~1,5–2 pg |
| 4. Resultados e Discussão | Tabelas + **figura de trade-off** + **figura qualitativa** + McNemar/Wilson + leitura operacional | ~2–2,5 pg |
| 5. Ameaças à Validade | 6 itens enxutos | ~0,5 pg |
| 6. Conclusão e Trabalhos Futuros | **Responder RQ1/RQ2 uma a uma**; Gemma 4 + localização como futuro | ~0,5 pg |
| Referências | — | ~0,5–1 pg |

Elementos prontos para reusar estão na auditoria: **RQ1 reformulada** (A1), **parágrafo de diferenciação Malla/Funk** (A2), **molde do prompt** (A3), **caveat de latência** (A1/M2). Não vou repeti-los aqui — copie de lá.

---

## 6. Experimentos opcionais (apenas os baratos e de alto valor)

### 6.1 Baseline CLIP zero-shot (RECOMENDADO)
- **Hipótese:** a vantagem do Gemma é significativa frente a um VLM contrastivo leve de referência; e CLIP entre YOLO e Gemma reforça a leitura de *domain shift*.
- **Dados/divisão:** as **mesmas 560**, sem treino; prompts de classe (ex.: "rua com lixo" × "rua limpa").
- **Métricas:** acurácia, F1, sensibilidade, especificidade, FPR; **McNemar** vs. Gemma e vs. YOLO.
- **Execuções:** 1 (determinístico).
- **Saída esperada:** uma linha nas Tabelas 3/6 + um ponto na figura de trade-off.
- **Conclusão que pode (ou não) sustentar:** se CLIP < Gemma e ≈ YOLO, fortalece "a capacidade multimodal generalista do Gemma é o diferencial, e parte da desvantagem do YOLO é domínio". **Resultados reais: [VALIDAR COM EXPERIMENTO].**
- **Custo:** ~meio dia. **Referência a citar:** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.

### 6.2 Taxa de sucesso de localização do Gemma 3 (OPCIONAL)
- **Por que NÃO mAP completo:** comparar o mAP de um detector (que tem de achar *todas* as instâncias) com coordenadas que o VLM cospe em texto é injusto por formulação de tarefa, e reintroduz o mAP que você corretamente removeu.
- **Métrica honesta:** dentre os **verdadeiros positivos** do Gemma, em que fração ele colocou **uma** caixa com **IoU > 0,5** contra o ground truth. Transforma sua afirmação atual ("o YOLO fornece localização, uma vantagem") em **número medido**.
- **Cuidados:** query **separada** do experimento binário; coordenadas no formato do Gemma (0–1000, ordem y,x); tratar saída inválida; reportar quantas imagens tinham múltiplas instâncias.
- **Conclusão que pode sustentar:** quantifica a vantagem de localização do YOLO sem mAP. **Resultados reais: [VALIDAR COM EXPERIMENTO].**
- **Custo:** ~meio a um dia. Se apertar, é o primeiro a cair.

---

## 7. Obras e referências (organizadas)

> Confirme strings/DOIs exatos no SOL e nas fontes originais antes de incluir. Não inventar citação.

**A. Precedentes KDMiLe/JIDM (estudar a estrutura e citar quando couber):**
- Almeida, F. C.; Caminha, C. *Evaluation of Entry-Level Open-Source LLMs for Information Extraction from Digitized Documents.* KDMiLe 2024. ← **molde principal**
- Marcato, B. U. et al. *Automatic Area Estimation of Mice Wound Images.* KDMiLe 2024.
- *Detecção de Lesão Tibial em imagens térmicas* (CNN/VGG19 × ML clássico, Wilcoxon). KDMiLe 2025.
- Ferreira, B. V. et al. *CNN-DFT … Image Inspection of Railcar Component: A Comparison with ML Methods.* JIDM.
- *A Comparative Study of BERT Models for … Brazilian Legal Precedents.* KDMiLe 2024.
- Valeriano et al. (DOI 10.5753/kdmile.2024.244519); Augusto et al. (10.5753/kdmile.2024.243865); Silva et al. (10.5753/kdmile.2024.244764), todos KDMiLe 2024.

**B. Núcleo VLM / detecção / resíduos (eixo da sua contribuição e da diferenciação):**
- Malla, Bazli & Arashpour — VLMs para classificação de resíduos em instalações de recuperação de materiais (trade-off acurácia × velocidade). ← **principal alvo de diferenciação**
- Funk et al., 2025 — VLMs zero/few-shot para resíduos.
- Terven, Córdova-Esparza & Romero-González, *A comprehensive review of YOLO architectures*, 2023 (já no seu .bib); + documentação YOLO11/Ultralytics.
- Gemma 3 Technical Report (modelo usado); Gemma 4 model card / Gemma 4 (apenas para **Trabalhos Futuros**).
- PaliGemma / PaliGemma 2 (grounding com tokens de localização) — para a discussão de localização como futuro.
- Zhang et al., *Vision-language models for vision tasks: a survey*, 2024 (já no seu .bib).
- (Se fizer 6.1) Radford et al., CLIP, 2021.

**C. Metodologia (sustentam o rigor que o veículo valoriza):**
- McNemar, *Note on the sampling error of the difference between correlated proportions or percentages*, 1947 (ou referência equivalente para o teste de McNemar).
- Wilson, intervalo de score para proporções binomiais, 1927 (ou referência equivalente).
- Referência de *perceptual hashing*/pHash para deduplicação (justifica seu controle anti-vazamento — um dos seus pontos mais fortes).

---

## 8. Riscos da janela de 8 dias e mitigação

| Risco | Sinal de alerta | Mitigação |
| --- | --- | --- |
| **Scope creep** (trocar modelo / mAP completo) | "Já que estou aqui, vou só testar o Gemma 4…" | Decisões da Seção 3 travadas; Gemma 4 = JIDM |
| **Estouro de 8 páginas** | PDF em 8,5+ no Dia 5 | Portar ao template no Dia 1; plano de corte B2 no Dia 7 |
| **Submissão na última hora** | Deixar JEMS para 22/06 à noite | Dry-run + submissão no Dia 7–8; AoE dá fuso extra, mas não conte com ele |
| **Repo não público / prompt ausente** | Revisor tenta acessar e não encontra | Repo público no Dia 1; prompt no Dia 2 |
| **Quebrar a consistência numérica ao editar** | Tabelas deixam de bater após cortes | Re-verificação no Dia 7 (recalcular matrizes + McNemar) |
| **Esquecer um bloqueador formal** | — | Checklist da Seção 9 antes de submeter |

---

## 9. Checklist de submissão (condensado)

- [ ] PDF gerado no **template oficial**, **≤ 8 páginas** (B1, B2)
- [ ] Título, **abstract e keywords em inglês**; CCS Concepts no formato ACM
- [ ] **RQ1/RQ2 explícitas** e **respondidas** na conclusão; claim de acurácia **limitado** (A1)
- [ ] Parágrafo de **diferenciação vs. Malla/Funk** (A2)
- [ ] **Prompt do Gemma** + tratamento de saída inválida (A3)
- [ ] **Figura de trade-off** acurácia/F1 × latência (A4) + figura qualitativa (M4)
- [ ] **Repositório público** com configs, script de avaliação e prompt (A5)
- [ ] (Recomendado) **baseline CLIP** com McNemar (6.1)
- [ ] **Tabela de config** reconciliada com o repo; números **re-verificados** e consistentes
- [ ] Ética (Street View/anonimização) + **agradecimentos** preenchidos + e-mails institucionais
- [ ] Declaração de **não-duplicidade**
- [ ] Submetido via **JEMS 3** antes de **22/06/2026, 23:59 AoE**, com confirmação salva

---

### Síntese
Você entra nesta reta final como **um candidato competitivo dentro de um veículo de ~40% de aceitação**, com um precedente direto do seu próprio grupo mostrando que esse tipo de paper (avaliação aplicada de modelos generativos open-source) é aceito aqui. Os 8 dias decidem o resultado não por mais experimentos, mas por **formato, enquadramento, posicionamento e reprodutibilidade**. Faça os Dias 1, 3, 4, 6 e 7 como inegociáveis, trate CLIP e localização como bônus sacrificáveis, e reserve a ambição do Gemma 4 + localização completa para a versão estendida no JIDM.
