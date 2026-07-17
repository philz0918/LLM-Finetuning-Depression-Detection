# Beyond Positive or Negative: Detecting Mixed Emotions for Depressive Symptom Prediction with Finetuned RoBERTa and LoRA-Finetuned LLaMA

This repository contains the code and data for **"Beyond Positive or Negative: How Mixed Emotions on Reddit Predict Depressive Symptoms in Asian American Families."** The project investigates whether mixed emotional expressions in Reddit posts can help predict depressive symptoms in the context of Asian American family experiences.

Beyond the research question itself, this repo demonstrates an end-to-end **LLM finetuning pipeline** for a low-resource, domain-specific classification task — comparing a fully finetuned encoder model (RoBERTa) against a parameter-efficient, **LoRA-finetuned decoder LLM (LLaMA)**.

## Key Technical Highlights

- **Task**: Multi-signal text classification — detecting mixed (co-occurring positive and negative) emotions in Reddit posts and using them as predictors of depressive symptoms.
- **RoBERTa finetuning**: Full supervised finetuning of a pretrained RoBERTa encoder for emotion/symptom classification (`RoBERTa_finetuning.py`), with a separate evaluation script (`RoBERTa_testing.py`).
- **LLaMA + LoRA (PEFT)**: Parameter-efficient finetuning of LLaMA using **Low-Rank Adaptation (LoRA)**, training small rank-decomposition adapter matrices while keeping base model weights frozen (`llama_finetuning.py`). This dramatically reduces trainable parameters and GPU memory requirements while retaining strong downstream performance. Evaluation in `llama_testing.py`.
- **Data pipeline**: Collection, cleaning, and exploratory analysis of Reddit submissions (r/AsianParentStories-style data) in `Reddit_asian_parents.ipynb`, producing the processed dataset in `data/`.
- **Model comparison**: Encoder-based full finetuning vs. decoder-based PEFT, evaluated on the same held-out data for a controlled comparison.

## Datasets

- **Reddit mixed-emotion corpus** — collected and processed in this repo (`data/`, `Reddit_asian_parents.ipynb`).
- **RedSM5 (depressive symptom labels)** — gated dataset; request access from the authors at [huggingface.co/datasets/irlab-udc/redsm5](https://huggingface.co/datasets/irlab-udc/redsm5). It is **not** included in this repository.

## Repository Structure

```
.
├── data/
│   └── updated_asian_parents_submissions.csv   # Processed Reddit submissions dataset
└── src/
    ├── README.md
    ├── Reddit_asian_parents.ipynb   # Data collection, cleaning & exploratory analysis
    ├── RoBERTa_finetuning.py        # Full finetuning of RoBERTa for classification
    ├── RoBERTa_testing.py           # Evaluation of the finetuned RoBERTa model
    ├── llama_finetuning.py          # LoRA (PEFT) finetuning of LLaMA
    └── llama_testing.py             # Evaluation of the LoRA-finetuned LLaMA model
```

## Methods

### 1. Data — Reddit Mixed-Emotion Corpus
Reddit posts related to Asian American family experiences are collected and preprocessed in `Reddit_asian_parents.ipynb`. Posts are labeled for emotional content, with particular attention to *mixed* emotional expressions (simultaneous positive and negative affect), which prior work suggests may be a meaningful signal for depressive symptoms.

> **⚠️ Depressive symptom dataset access**: The depressive symptom annotations used for finetuning come from the **RedSM5** dataset, which is gated and not redistributed in this repository. To obtain access, please contact the dataset authors via its Hugging Face page: [irlab-udc/redsm5](https://huggingface.co/datasets/irlab-udc/redsm5).

### 2. RoBERTa — Full Finetuning Baseline
A pretrained RoBERTa model is finetuned end-to-end with a classification head. All transformer weights are updated during training. This serves as a strong encoder baseline for the task.

### 3. LLaMA — LoRA Finetuning (Parameter-Efficient)
LLaMA is adapted using **LoRA**: low-rank adapter matrices are injected into the attention projection layers and trained while the base model remains frozen. Compared to full finetuning, this approach:

- Trains only a small fraction of parameters (typically <1% of the base model)
- Fits on consumer/single-GPU hardware
- Produces lightweight adapter checkpoints that can be merged or swapped

### 4. Evaluation
Both models are evaluated on held-out Reddit posts (`RoBERTa_testing.py`, `llama_testing.py`), enabling a direct comparison of full encoder finetuning vs. parameter-efficient LLM adaptation on the depressive symptom prediction task.

## Getting Started

```bash
# Finetune RoBERTa
python src/RoBERTa_finetuning.py

# Finetune LLaMA with LoRA
python src/llama_finetuning.py

# Evaluate
python src/RoBERTa_testing.py
python src/llama_testing.py
```

**Dependencies** (typical): `transformers`, `peft`, `torch`, `datasets`, `pandas`, `scikit-learn`.

## Motivation

Depression in Asian American communities is often underreported and expressed indirectly. Binary sentiment analysis (positive/negative) misses the ambivalent, mixed emotional language common in these narratives. This project moves *beyond positive or negative* by explicitly modeling mixed emotions with modern finetuned language models.
