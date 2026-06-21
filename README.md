# Specialized Detection or Zero-Shot Vision-Language Modeling?

Repositório do artigo **"Specialized Detection or Zero-Shot Vision-Language Modeling? Garbage Bag Identification and Localization in Brazilian Street Images"**, de Jaime Teixeira de Araújo Júnior e Carlos Caminha, **Kunumi Lab - Universidade Federal do Ceará (UFC)**. O experimento compara **YOLOv11m** e **Gemma 4 31B-QAT** para identificar a presença e a localização de sacos de lixo em vias públicas brasileiras. O notebook canônico do projeto é `garbage-bag-yolov11m-vs-gemma4.ipynb`.

O projeto foi organizado para que um clone do repositório consiga reproduzir as principais tabelas e figuras sem redistribuir imagens originais não autorizadas. Para isso, artefatos leves de cache foram versionados.

## Resultados principais

| Métrica | YOLOv11m | Gemma 4 31B-QAT |
|---|---|---|
| F1 classificação binária | 0,8685 | **0,9650** |
| F1 caixas (IoU ≥ 0,5) | **0,5744** | 0,1546 |
| Taxa de sucesso por imagem | **0,8071** | 0,3821 |
| Latência média por imagem | **10,2 ms** | 1.278,6 ms |

O Gemma supera o YOLO na identificação binária (presença/ausência); o YOLO supera o Gemma na localização precisa com caixas e é ~126× mais rápido.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Copie `env.example` para `.env`, informe `ROBOFLOW_API_KEY`, carregue o modelo no LM Studio e execute **Run All** em `garbage-bag-yolov11m-vs-gemma4.ipynb`. O progresso é salvo em `outputs/pipeline_state.json`; etapas concluídas e caches compatíveis são reutilizados automaticamente.

O download e a preparação dos datasets de treino ocorrem antes do treinamento. O dataset `urban-waste-brazil` é baixado e preparado somente depois que o YOLO termina, imediatamente antes das inferências. Enquanto o teste não estiver pronto, use `TRAIN_ONLY_PIPELINE=true`: o Run All concluirá o download dos dados de treino e o treinamento, sem tentar baixar o teste ou executar inferências. Quando o teste estiver pronto, altere para `false`.

O treinamento YOLO salva `last.pt` a cada época e retoma desse checkpoint após uma interrupção. As inferências YOLO e VLM são salvas por imagem, de modo que somente imagens pendentes são processadas novamente. Para apenas renderizar os artefatos versionados, defina `RUN_ALL_PIPELINE=false`.

## Artefatos versionados

- `data/teste/ground_truth.csv` e `data/teste/labels/*.txt`
- `data/unified/manifest.csv`, `data/unified/data.yaml`, `data/unified/summary.json` e `data/unified/split_audit.json`
- `outputs/predictions_yolo.json`
- `outputs/gemma-4-31b-qat/predictions_vlm.json`
- `outputs/gemma-4-31b-qat/predictions_vlm_localizacao.json`
- `outputs/varredura_limiares.csv`
- `outputs/gemma-4-31b-qat/analises/*.csv`
- `outputs/figures/fig_amostras_localizacao_fp_fn.png`
- `prompt_gemma_classificacao.txt` e `prompt_gemma_localizacao.txt`

As imagens, pesos e checkpoints intermediários não são versionados. Configure o comportamento no `.env` a partir de `env.example`.

## Artigo

Os arquivos LaTeX estão em `manuscript/`. O artigo foi submetido ao **KDMiLe 2025** com 8 páginas e 20 referências. Para compilar:

```bash
cd manuscript
latexmk -pdf garbage-bag-yolov11m-vs-gemma4.tex
```

Ou, manualmente:

```bash
cd manuscript
pdflatex garbage-bag-yolov11m-vs-gemma4.tex
bibtex garbage-bag-yolov11m-vs-gemma4
pdflatex garbage-bag-yolov11m-vs-gemma4.tex
pdflatex garbage-bag-yolov11m-vs-gemma4.tex
```

## Preparação de dados

A construção de `data/unified` foi modularizada em `scripts/prepare_datasets.py`. Conforme o paper, o treinamento do YOLOv11m foi realizado com 4.235 imagens positivas e 424 imagens negativas de fundo:

- `garbage-8uzha` (v4): 3.370 imagens positivas, usando apenas as classes `black_bag` e `white_bag`;
- `garbage-mvzg3` (v1): 865 imagens positivas, agregadas na classe única `saco_de_lixo`;
- `Roadway` como fundo: 424 imagens negativas sem sacos anotados, equivalentes a 10% das positivas, limitadas pela disponibilidade elegível.

O teste é baixado de `jaime-teixeira/urban-waste-brazil` e contém 560 capturas do Google Street View de diferentes estados brasileiros, sendo 280 imagens positivas e 280 negativas. Dois pesquisadores anotaram o conjunto de forma independente: cada um anotou 50% das imagens e revisou os 50% restantes. As divergências foram discutidas e ajustadas por consenso, e todos os sacos de lixo visíveis foram anotados individualmente antes da subida para o Roboflow. No total, o teste contém 785 caixas anotadas. Todas as classes não selecionadas são descartadas antes da unificação, e as classes positivas são remapeadas para `saco_de_lixo`.

As imagens do Google Street View foram utilizadas exclusivamente para avaliação, com anonimização dos elementos sensíveis, respeito aos termos de uso aplicáveis e sem redistribuição não autorizada.

## Modelo VLM

O modelo foi carregado pelo identificador [`google/gemma-4-31b-qat`](https://lmstudio.ai/models/google/gemma-4-31b-qat), que no catálogo do LM Studio é baseado no repositório GGUF [`lmstudio-community/gemma-4-31B-it-QAT-GGUF`](https://huggingface.co/lmstudio-community/gemma-4-31B-it-QAT-GGUF). A inferência foi executada no LM Studio 0.4.15 (Build 2) com o arquivo `gemma-4-31B-it-QAT-Q4_0.gguf`: formato GGUF, quantização Q4_0, arquitetura `gemma4` e tamanho de 18,85 GB em disco. O experimento usa dois prompts versionados, um para classificação binária e outro para localização individual de até 20 sacos por imagem. Esses dados são registrados separadamente nos artefatos de proveniência.
