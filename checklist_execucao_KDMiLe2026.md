# Checklist de execução — KDMiLe 2026 (em ordem)

> Companheiro da auditoria (`auditoria_KDMiLe2026_yolo_vs_gemma.md`) e do plano (`plano_estrategico_KDMiLe2026_8dias.md`).
> Siga de cima para baixo. Cada item é uma ação. Onde houver "(→ X)", o detalhe/texto pronto está na auditoria sob aquele código.

**Restrições fixas:** prazo **22/06/2026, 23:59 AoE (UTC-12)** · **≤ 8 páginas** (excedeu = rejeição automática) · **somente PDF** no **template oficial** · submissão via **JEMS 3** (`jems3.sbc.org.br/kdmile2026`).

**Itens inegociáveis (se faltar tempo, NUNCA corte estes):** todos da Fase 0, 1, 3, 4, 6 e 7. Sacrificáveis nesta ordem: localização (2.4) → CLIP (2.3) → figura qualitativa (5.4).

---

## Fase 0 — Decisões e fundação (Dia 1)
- [ ] Decidir e anotar: **VLM principal = Gemma 3 27B-IT, sem troca** (Gemma 4 = só Trabalhos Futuros).
- [ ] Decidir e anotar: **experimentos opcionais permitidos = CLIP zero-shot (sim) + localização (talvez)**; proibidos = fine-tuning YOLO, mAP completo, troca de modelo.
- [ ] Decidir e anotar: **enquadramento = comparação de estratégias de implantação** (não de superioridade arquitetural).
- [ ] Abrir o **template LaTeX oficial do KDMiLe** (Overleaf) e criar o projeto.
- [ ] **Portar todo o conteúdo do `.odt` para o template** (texto + tabelas), mesmo bruto/incompleto.
- [ ] Compilar o **primeiro PDF** e **anotar a contagem de páginas atual**.
- [ ] Ler de ponta a ponta **Almeida & Caminha 2024** (molde principal) e folhear **Marcato 2024** + **lesão tibial 2025**.

## Fase 1 — Repositório (Dia 1)
- [ ] **Renomear** o repo para **`urban-waste-yolo-vs-vlm`**.
- [ ] Tornar o repositório **público**.
- [ ] Criar estrutura mínima: `README`, arquivo de ambiente/`requirements`, pasta de `configs`, **script de avaliação** (gera Tabelas 3–6), arquivo com o **prompt do Gemma**.
- [ ] Conferir que **nenhuma imagem com restrição de licença/privacidade** foi subida (só scripts, configs e referências).

## Fase 2 — Experimentos e congelamento de números (Dia 2)
- [ ] Rodar **baseline CLIP zero-shot** nas mesmas 560 (→ 6.1 do plano): acurácia, F1, sensibilidade, especificidade, FPR.
- [ ] Rodar **McNemar** do CLIP contra Gemma e contra YOLO.
- [ ] (Opcional) Rodar **taxa de sucesso de localização** do Gemma 3 em **query separada** (→ 6.2 do plano): fração de VPs com IoU > 0,5.
- [ ] **Extrair e salvar o prompt verbatim** do Gemma + esquema JSON + regra para saída inválida + **quantas** ocorreram (→ A3).
- [ ] **Congelar todos os números** num único arquivo de resultados versionado (a partir daqui, não re-rodar sem motivo forte).

## Fase 3 — Reenquadramento (Dia 3) — maior impacto na aceitação
- [ ] Reescrever **RQ1** como comparação de estratégias de implantação (→ A1, texto pronto).
- [ ] Inserir o parágrafo que **limita o claim de acurácia** (domain shift / treino estrangeiro) na §4.1 e na conclusão (→ A1).
- [ ] Reescrever o **Abstract** (inglês, 100–300 palavras): gancho + trade-off + achado.
- [ ] Reescrever o **gancho de relevância** da Introdução com um **número concreto** (custo de fiscalização manual / escala do monitoramento).
- [ ] Reescrever a lista de **contribuições** no molde do Almeida & Caminha.

## Fase 4 — Trabalhos relacionados e lacuna (Dia 4)
- [ ] Inserir o **parágrafo de diferenciação frente a Malla/Funk** (→ A2, texto pronto): litter × triagem; dado BR in-the-wild; confronto vs. detector ajustado + McNemar.
- [ ] Acrescentar 1–2 **precedentes de comparação em visão** (Marcato; lesão tibial) para ancorar o gênero.
- [ ] Conferir no SOL e adicionar ao `.bib` as **citações dos precedentes** (strings/DOIs exatos — não inventar).

## Fase 5 — Resultados, discussão e figuras (Dia 5)
- [ ] Integrar **CLIP** nas Tabelas 3/6 e no texto.
- [ ] Produzir a **figura de trade-off** acurácia/F1 × latência (latência em escala log; pontos YOLO, Gemma, CLIP) (→ A4).
- [ ] Produzir a **figura qualitativa** com 2–4 exemplos de acerto/erro de cada modelo (→ M4).
- [ ] **Contextualizar a latência ("~142×")** como ciclo HTTP+LM Studio no texto e no abstract (→ M2).
- [ ] Garantir que a discussão explica **por que** cada modelo erra (sobre-sinalização do Gemma × cegueira de domínio do YOLO), não só quanto.

## Fase 6 — Metodologia, reprodutibilidade e ética (Dia 6)
- [ ] **Inserir o prompt do Gemma** + tratamento de saída inválida no texto (→ A3).
- [ ] **Reconciliar a Tabela de configuração** com o config real do repo (versões, limiar 0,25, épocas, otimizador, seeds, Q5_K_M).
- [ ] Descrever o **ground truth** (e o das bboxes, se usar localização): origem, critério, anotador.
- [ ] Adicionar frase de **ética**: Street View + anonimização de rostos/placas.
- [ ] Condensar a **§5 (ameaças à validade)** para itens enxutos (libera espaço — → B2).
- [ ] Inserir/atualizar o **link do repositório** (`urban-waste-yolo-vs-vlm`) no paper.
- [ ] Preencher os **agradecimentos** (remover qualquer placeholder).

## Fase 7 — Páginas e polimento (Dia 7)
- [ ] Aplicar o **plano de corte** para caber em ≤ 8 páginas (→ B2: comprimir §4.5 e §5, enxugar histórico YOLO/VLM, eventualmente mover Tabela 5 para a figura).
- [ ] **Re-verificar a consistência de todos os números** (matrizes de confusão + McNemar precisam continuar batendo após a edição).
- [ ] Conferir **CCS Concepts** (formato ACM), **keywords**, **título e abstract em inglês**.
- [ ] Conferir **figuras e tabelas** (legendas, referências cruzadas, legibilidade).
- [ ] Revisar o **`.bib`** (todas as citações compilam e aparecem).
- [ ] Confirmar **ausência de placeholders** ("[INSERIR...]") e **PDF final ≤ 8 páginas**.
- [ ] Trocar e-mails do cabeçalho para **institucionais (UFC)**.

## Fase 8 — Revisão externa e submissão (Dia 8)
- [ ] Enviar para **orientador/coautor** revisar (idealmente já no fim do Dia 7).
- [ ] Última **leitura de prova** completa.
- [ ] Conferir **declaração de não-duplicidade** (sem submissão/publicação simultânea).
- [ ] **Dry-run no JEMS 3**: criar submissão, preencher campos, anexar PDF.
- [ ] **Submeter com folga** antes de 22/06 23:59 AoE.
- [ ] **Salvar a confirmação** do JEMS.

---

### Gates de verificação (não avance sem cumprir)
- [ ] **Fim do Dia 1:** compila em PDF no template + contagem de páginas conhecida + repo público e renomeado.
- [ ] **Fim do Dia 2:** todos os números congelados num só arquivo + prompt documentado.
- [ ] **Fim do Dia 6:** texto sem furos de conteúdo (todas as inserções A1–A5, M2, M4 feitas).
- [ ] **Fim do Dia 7:** PDF ≤ 8 páginas, sem placeholders, números re-verificados e consistentes.
- [ ] **Dia 8:** submetido + confirmação salva.
