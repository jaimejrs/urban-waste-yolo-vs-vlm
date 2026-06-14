# Specialized Detector or Vision-Language Model? Evaluating YOLOv11m and Gemma 3 27B-IT for Urban Waste Monitoring

This repository contains the LaTeX source code, templates, configuration scripts, and documentation for the paper submitted to the **XIV Symposium on Knowledge Discovery, Mining and Learning (KDMiLe 2026)**.

---

## Abstract

Urban waste monitoring on public roads is a relevant challenge for municipal management because it requires large-scale inspection under limited operational resources. This study evaluates two distinct artificial intelligence strategies for identifying visible waste in real images of Brazilian streets: YOLOv11m, fine-tuned as a specialized object detector, and Gemma 3 27B-IT, employed as a zero-shot vision-language model. 

Our findings reveal a clear accuracy-latency trade-off: Gemma is more suitable for targeted analyses that prioritize predictive performance, whereas YOLO is more appropriate for large-scale screening, real-time processing, and applications requiring explicit visual localization.

---

## Experimental Results

### 1. Predictive Performance (Test set of 560 Brazilian street images)

| Metric | YOLOv11m | Gemma 3 27B-IT | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7071 | **0.9179** | +0.2108 |
| **Precision** | 0.7117 | **0.8611** | +0.1494 |
| **Sensibility (Recall)** | 0.6964 | **0.9964** | +0.3000 |
| **Specificity** | 0.7179 | **0.8393** | +0.1214 |
| **F1-score** | 0.7040 | **0.9238** | +0.2198 |
| **FPR (False Positive Rate)** | 0.2821 | **0.1607** | -0.1214 |
| **MCC** | 0.4140 | **0.8460** | +0.4320 |
| **Cohen's Kappa** | 0.4140 | **0.8360** | +0.4220 |

*The difference is statistically significant according to McNemar's test ($\chi^2 = 74.40$, $p \approx 6.4 \times 10^{-18}$).*

### 2. Computational Efficiency (NVIDIA GeForce RTX 5090)

| Metric | YOLOv11m | Gemma 3 27B-IT | Comparison |
| :--- | :---: | :---: | :---: |
| **Average Latency** | **8.4 ms** | 1,191.9 ms | Gemma is 141.9x slower |
| **P95 Latency** | **9.3 ms** | 1,346.9 ms | YOLO is extremely stable |
| **Throughput** | **119 img/s** | 0.839 img/s | YOLO enables real-time |

---

## Repository Structure

```text
.
├── .gitignore                    # Git ignore rules
├── env.example                   # Environment configuration template
├── prompt_gemma.txt              # Specification of the Gemma 3 prompt & schema
├── README.md                     # This file
├── sbc-template.sty              # Official SBC LaTeX template style file
├── sbc.bst                       # Official SBC BibTeX bibliography style file
├── urban-waste-yolo-vs-vlm.ipynb # Evaluation pipeline notebook
└── urban-waste-yolo-vs-vlm.tex   # LaTeX paper source code
```

*Note: The directory `orientacoes/` containing local planning templates, checklists, and draft notes is excluded from Git tracking.*

---

## How to Compile the LaTeX Paper

To build the PDF of the paper locally, ensure you have a working LaTeX distribution installed (such as TeX Live) and run the following command in the root directory:

```bash
pdflatex urban-waste-yolo-vs-vlm.tex
pdflatex urban-waste-yolo-vs-vlm.tex
```

This compiles the document twice to resolve all cross-references and outputs `urban-waste-yolo-vs-vlm.pdf`.
