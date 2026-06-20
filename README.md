# Garbage Bag Detection: YOLO vs VLM

Repositório do experimento comparando **YOLOv11m** e **Gemma 4 31B-QAT** para identificar a presença e a localização de sacos de lixo em vias públicas. O notebook principal é `urban-waste-yolo-vs-vlm-otimizado.ipynb`.

O projeto foi organizado para que um clone do repositório consiga renderizar as principais tabelas e figuras sem redistribuir as imagens originais, que dependem das licenças das bases e das fontes de coleta. Para isso, artefatos leves de cache foram versionados.

## Como reproduzir em modo cache

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Abra `urban-waste-yolo-vs-vlm-otimizado.ipynb` e execute as células. Por padrão, as flags de treino, download, reconstrução de dados, inferência YOLO/VLM e auditoria pHash ficam desligadas. O notebook carrega os artefatos versionados em `data/` e `outputs/`.

## Artefatos versionados

- `data/teste/ground_truth.csv` e `data/teste/labels/*.txt`
- `data/unified/manifest.csv`, `data/unified/data.yaml`, `data/unified/summary.json` e `data/unified/split_audit.json`
- `outputs/predictions_yolo.json`
- `outputs/gemma-4-31b-qat/predictions_vlm.json`
- `outputs/gemma-4-31b-qat/predictions_vlm_localizacao.json`
- `outputs/varredura_limiares.csv`
- `outputs/gemma-4-31b-qat/analises/*.csv`
- `outputs/figures/fig_tradeoff_f1_latencia.png`

As imagens e pesos de modelo não são versionados. Para reconstruir os dados ou treinar novamente, configure `.env` a partir de `env.example`, coloque as fontes locais necessárias e altere explicitamente as flags no notebook.

## Resultados em cache

Os artefatos atualmente versionados correspondem à execução anterior. Como a classe-alvo e os prompts foram restringidos a sacos de lixo e o TACO foi removido do treinamento, as métricas deverão ser regeneradas antes da versão final do artigo.

## Artigo

Os arquivos LaTeX do artigo estão em `manuscript/`. Para compilar:

```bash
cd manuscript
pdflatex urban-waste-yolo-vs-vlm.tex
bibtex urban-waste-yolo-vs-vlm
pdflatex urban-waste-yolo-vs-vlm.tex
pdflatex urban-waste-yolo-vs-vlm.tex
```

## Preparação de dados

A construção de `data/unified` foi modularizada em `scripts/prepare_datasets.py`. O script é chamado pelo notebook somente quando `RUN_RECONSTRUIR_UNIFIED=True`.

## Modelo VLM

O modelo foi carregado pelo identificador [`google/gemma-4-31b-qat`](https://lmstudio.ai/models/google/gemma-4-31b-qat), que no catálogo do LM Studio é baseado no repositório GGUF [`lmstudio-community/gemma-4-31B-it-QAT-GGUF`](https://huggingface.co/lmstudio-community/gemma-4-31B-it-QAT-GGUF). A inferência foi executada no LM Studio 0.4.15 (Build 2) com o arquivo `gemma-4-31B-it-QAT-Q4_0.gguf`: formato GGUF, quantização Q4_0, arquitetura `gemma4` e tamanho de 18,85 GB em disco. Esses dados são registrados separadamente nos artefatos de proveniência.
