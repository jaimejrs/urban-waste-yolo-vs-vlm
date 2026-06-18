# Metodologia e resultados do notebook otimizado

Este documento resume a metodologia experimental e os resultados produzidos pelo notebook `urban-waste-yolo-vs-vlm-otimizado.ipynb`. O objetivo do experimento e comparar um detector supervisionado baseado em YOLOv11m com um modelo vision-language, Gemma 4 31B-QAT, para identificar e localizar lixo ou residuos descartados em imagens de ambiente urbano.

## Objetivo experimental

O estudo avalia dois tipos de tarefa:

1. **Classificacao binaria por imagem**: determinar se uma imagem contem lixo/residuo visivel.
2. **Localizacao de residuos**: comparar caixas previstas com caixas anotadas, usando IoU >= 0,50 como criterio de acerto.

O notebook otimizado foi organizado para reproduzir os resultados ja gerados sem retreinar ou reinferir por padrao. As flags registradas na ultima execucao foram:

| Flag | Valor |
|---|---:|
| `RUN_TREINO_YOLO` | `False` |
| `RUN_YOLO_INFERENCIA` | `False` |
| `RUN_VLM_CLASSIFICACAO` | `False` |
| `RUN_VLM_LOCALIZACAO` | `False` |
| `RUN_DOWNLOAD_DADOS` | `False` |
| `RUN_RECONSTRUIR_UNIFIED` | `False` |
| `RUN_AUDITORIA_PHASH` | `True` |

Assim, um `Run All` no estado atual usa os pesos YOLO, predicoes e caches Gemma ja existentes, gerando tabelas, graficos, metricas, amostras visuais e auditorias.

## Conjuntos de dados

O conjunto unificado de treinamento e validacao foi construido a partir de fontes positivas e negativas:

| Fonte | Papel | Train imagens | Train caixas | Val imagens | Val caixas |
|---|---|---:|---:|---:|---:|
| `garbage-8uzha` | positivo | 2.895 | 3.648 | 506 | 636 |
| `garbage-mvzg3` | positivo | 739 | 1.145 | 130 | 190 |
| TACO | positivo | 1.309 | 4.204 | 191 | 580 |
| `sidewalk-segmentation` | negativo | 1.614 | 0 | 314 | 0 |

Total do `data/unified`:

| Split | Imagens | Caixas |
|---|---:|---:|
| Treino | 6.557 | 8.997 |
| Validacao | 1.141 | 1.406 |

O conjunto de teste usado na avaliacao final contem **560 imagens**, balanceadas em:

| Classe | Imagens |
|---|---:|
| Com lixo/residuo | 280 |
| Sem lixo/residuo | 280 |

Cada imagem de teste possui arquivo de label correspondente em `data/teste/labels`, e o arquivo `data/teste/ground_truth.csv` contem 560 linhas.

## Modelos avaliados

### YOLOv11m

O modelo YOLO foi treinado para uma unica classe: `lixo`. O checkpoint avaliado foi:

`runs/detect/models/yolov11m_lixo_garbage_8uzha_garbage_mvzg3_taco/weights/best.pt`

Hash SHA-256 dos pesos:

`9fc264281e38467feb2167fc84c6473d2febf0178640c8d7c761ab2f4aca8575`

O limiar principal usado na classificacao binaria foi `0,25`. A classificacao binaria do YOLO foi derivada da existencia de ao menos uma deteccao acima desse limiar.

### Gemma 4 31B-QAT

O VLM avaliado foi o **Gemma 4 31B-QAT**, executado via LM Studio. O modelo registrado nos resultados foi:

`google/gemma-4-31b-qat`

Foram usadas duas chamadas distintas:

1. classificacao binaria da presenca de lixo/residuo;
2. localizacao, solicitando caixas no formato estruturado.

Hashes dos prompts:

| Prompt | Hash |
|---|---|
| Classificacao | `591256762fe7185b` |
| Localizacao | `ddec5b514bcd591a` |

## Metricas de avaliacao

Para classificacao binaria foram calculadas:

- acuracia;
- intervalo de confianca de Wilson para acuracia;
- precisao;
- recall;
- especificidade;
- balanced accuracy;
- F1-score;
- MCC;
- Cohen's kappa;
- matriz de confusao;
- teste pareado exato tipo McNemar;
- intervalo de confianca bootstrap pareado para a diferenca de acuracia.

Para localizacao foram calculadas:

- numero de caixas GT;
- numero de caixas previstas;
- TP, FP e FN por caixas usando IoU >= 0,50;
- precisao, recall e F1 por caixas;
- taxa de imagens positivas com ao menos uma localizacao correta;
- melhor IoU medio por imagem positiva;
- taxa de imagens negativas com predicao;
- comparacao pareada entre YOLO e Gemma.

## Resultados de classificacao

| Modelo | Acuracia | IC95 acuracia | Precisao | Recall | Especificidade | F1 | MCC | Kappa | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv11m | 0,7857 | [0,7498; 0,8177] | 0,8636 | 0,6786 | 0,8929 | 0,7600 | 0,5850 | 0,5714 | 30 | 90 |
| Gemma 4 31B-QAT | 0,9839 | [0,9697; 0,9915] | 0,9927 | 0,9750 | 0,9929 | 0,9838 | 0,9680 | 0,9679 | 2 | 7 |

Matrizes de confusao no formato `[[TN, FP], [FN, TP]]`:

| Modelo | Matriz |
|---|---|
| YOLOv11m | `[[250, 30], [90, 190]]` |
| Gemma 4 31B-QAT | `[[278, 2], [7, 273]]` |

Comparacao pareada:

| Comparacao | Diferenca de acuracia | IC95 bootstrap | Discordantes | p-valor exato |
|---|---:|---:|---:|---:|
| Gemma - YOLO | 0,1982 | [0,1625; 0,2321] | 125 | 3,98e-27 |

O Gemma obteve desempenho substancialmente superior na classificacao binaria, com ganho absoluto de 20 pontos percentuais de acuracia sobre o YOLO no conjunto de teste.

## Varredura de limiar do YOLO

| Limiar | Acuracia | Precisao | Recall | Especificidade | F1 | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,25 | 0,7857 | 0,8636 | 0,6786 | 0,8929 | 0,7600 | 30 | 90 |
| 0,35 | 0,7768 | 0,9016 | 0,6214 | 0,9321 | 0,7357 | 19 | 106 |
| 0,50 | 0,7625 | 0,9401 | 0,5607 | 0,9643 | 0,7025 | 10 | 123 |
| 0,65 | 0,7357 | 0,9925 | 0,4750 | 0,9964 | 0,6425 | 1 | 147 |

O aumento do limiar reduz falsos positivos, mas tambem aumenta falsos negativos. O limiar `0,25` foi mantido como principal por preservar maior recall, ainda com especificidade proxima de 0,89.

## Resultados de localizacao

A localizacao foi avaliada em 560 imagens, sendo 280 positivas e 280 negativas. O conjunto positivo contem 794 caixas GT.

| Modelo | Caixas previstas | TP caixas | FP caixas | FN caixas | Precisao caixas | Recall caixas | F1 caixas | Sucesso em imagens positivas | IoU medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv11m | 511 | 92 | 419 | 702 | 0,1800 | 0,1159 | 0,1410 | 0,2714 | 0,2866 |
| Gemma 4 31B-QAT | 527 | 112 | 415 | 682 | 0,2125 | 0,1411 | 0,1696 | 0,3893 | 0,4403 |

Taxa de imagens negativas com predicao:

| Modelo | Taxa |
|---|---:|
| YOLOv11m | 0,1071 |
| Gemma 4 31B-QAT | 0,0214 |

Comparacao pareada de localizacao:

| Metrica | Diferenca Gemma - YOLO | IC95 bootstrap | p-valor |
|---|---:|---:|---:|
| Taxa de sucesso IoU >= 0,50 em positivos | 0,1179 | [0,0393; 0,1964] | 0,00345 |
| Melhor IoU medio em positivos | 0,1537 | [0,1070; 0,1974] | 4,39e-07 |

Apesar de ambos os modelos apresentarem baixa performance absoluta na localizacao por caixas, o Gemma teve melhor taxa de sucesso por imagem positiva e maior melhor IoU medio.

## Latencia

| Modelo | Inferencia media | End-to-end medio |
|---|---:|---:|
| YOLOv11m | 7,20 ms | 9,39 ms |
| Gemma 4 31B-QAT | 1.329,94 ms | 1.332,39 ms |

O YOLO e significativamente mais rapido, enquanto o Gemma apresenta melhor desempenho preditivo no conjunto de teste. Esse contraste sustenta a discussao entre eficiencia computacional e desempenho semantico.

## Auditoria pHash

Foi adicionada auditoria perceptual com pHash usando distancia maxima `<= 5` para identificar duplicatas ou possiveis vazamentos.

| Grupo | Imagens | Hashes unicos | Erros | Hashes duplicados exatos | Imagens em hashes duplicados |
|---|---:|---:|---:|---:|---:|
| Teste | 560 | 560 | 0 | 0 | 0 |
| Treino | 6.557 | 6.344 | 0 | 191 | 404 |
| Validacao | 1.141 | 1.133 | 0 | 8 | 16 |

Foram encontrados:

- 309 grupos de duplicatas exatas por pHash;
- 145 pares suspeitos entre treino e validacao;
- 0 pares suspeitos entre treino e teste.

O resultado mais importante para a avaliacao final e que nao foram encontrados pares suspeitos entre treino e teste. Ainda assim, ha indicios de possivel proximidade entre treino e validacao, provavelmente associados a imagens derivadas ou aumentadas. Corrigir isso exigiria reconstruir o `data/unified` com split agrupado por imagem original e treinar o YOLO novamente.

## Artefatos gerados

Os principais artefatos de saida estao em:

- `outputs/gemma-4-31b-qat/metricas_finais_otimizado.json`
- `outputs/gemma-4-31b-qat/analises/tabela_metricas_classificacao.csv`
- `outputs/gemma-4-31b-qat/analises/tabela_estatistica_classificacao.csv`
- `outputs/gemma-4-31b-qat/analises/tabela_metricas_localizacao.csv`
- `outputs/gemma-4-31b-qat/analises/tabela_estatistica_localizacao.csv`
- `outputs/gemma-4-31b-qat/analises/auditoria_phash_resumo.csv`
- `outputs/gemma-4-31b-qat/analises/auditoria_phash_vazamento_train_val.csv`
- `outputs/gemma-4-31b-qat/analises/auditoria_phash_vazamento_treino_teste.csv`
- `outputs/figures/`

## Reprodutibilidade sem imagens

O notebook foi ajustado para funcionar tambem em modo cache, sem exigir que as imagens dos datasets sejam publicadas no repositorio. Nesse modo, `data/teste/ground_truth.csv`, os labels YOLO em `data/teste/labels/`, as predicoes agregadas em JSON e os CSVs analiticos leves sao usados como fonte para reconstruir tabelas, metricas e graficos quantitativos.

As figuras qualitativas que desenham caixas sobre as imagens sao puladas quando `data/teste/images/` nao esta disponivel. Isso evita falhas em clones publicos do repositorio, respeitando restricoes de licenca dos datasets.

O preparo dos dados foi movido para `scripts/prepare_datasets.py`. O script documenta e executa:

- download das fontes Roboflow/TACO quando as credenciais locais estao disponiveis;
- reconstrucao de `data/teste`;
- reconstrucao de `data/unified`;
- split agrupado por imagem original, registrado em `data/unified/split_audit.json`.

## Limitacoes e pendencias metodologicas

1. **Split nao agrupado por imagem original**: a auditoria pHash encontrou pares suspeitos entre treino e validacao. Para uma versao metodologicamente mais forte, o conjunto `data/unified` deveria ser reconstruido com split agrupado por imagem original, seguido de novo treinamento YOLO.

2. **pHash usado como auditoria, nao como filtro**: o pHash atualmente documenta duplicatas e vazamentos potenciais, mas nao remove amostras. Remover amostras alteraria o conjunto de treino e exigiria novo treinamento.

3. **Localizacao ainda e fraca em termos absolutos**: tanto YOLO quanto Gemma apresentaram baixo recall por caixas. Isso sugere que a localizacao de residuos em cenas urbanas continua sendo uma tarefa dificil no conjunto de teste usado.

4. **Comparacao eficiencia versus desempenho**: o Gemma apresentou melhor desempenho de classificacao e localizacao, mas com latencia muito maior que o YOLO.

## Conclusao

No conjunto de teste balanceado com 560 imagens, o Gemma 4 31B-QAT superou o YOLOv11m na classificacao binaria e tambem apresentou vantagem na localizacao por IoU. O YOLO, por outro lado, foi muito mais rapido, operando em escala de milissegundos por imagem.

Os resultados indicam que o VLM e mais robusto semanticamente para reconhecer lixo em imagens urbanas variadas, enquanto o YOLO permanece mais adequado para cenarios em que latencia e custo computacional sao restricoes centrais. Para uma versao futura mais rigorosa, a principal melhoria metodologica e reconstruir o split de treino/validacao de forma agrupada por imagem original e retreinar o YOLO com essa nova divisao.
