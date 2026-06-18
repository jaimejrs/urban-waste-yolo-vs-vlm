# Urban Waste YOLO vs VLM

Repositório do experimento comparando **YOLOv11m** e **Gemma 4 31B-QAT** para monitoramento visual de resíduos em vias públicas. O notebook principal é `urban-waste-yolo-vs-vlm-otimizado.ipynb`.

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

## Resultados atuais

No conjunto de teste balanceado com 560 imagens, o Gemma 4 31B-QAT obteve F1 de 0,9838 na classificação binária, enquanto o YOLOv11m obteve F1 de 0,7600. Em contrapartida, o YOLO apresentou latência end-to-end média de 9,39 ms, contra 1.291,9 ms do Gemma no fluxo HTTP avaliado.

## Preparação de dados

A construção de `data/unified` foi modularizada em `scripts/prepare_datasets.py`. O script é chamado pelo notebook somente quando `RUN_RECONSTRUIR_UNIFIED=True`.

## Modelo VLM

O identificador operacional do projeto é `gemma-4-31b-qat`, usado no LM Studio. A família pública oficial possui o checkpoint `google/gemma-4-31B-it`; a variante QAT pode aparecer como artefato/quantização local no servidor de inferência.
