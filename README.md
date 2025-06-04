<div align="center">
  <h1>Adaptive Thinking-Time Control and Problem-Aware Reasoning for Math Problems via Reinforcement Learning</h1>
</div>

---

## 🧠 Overview

This repository contains code to reproduce the results of our work on **Adaptive Thinking-Time Control** and **Problem-Aware Reasoning** for solving MATH problems using **Reinforcement Learning** techniques. The code supports both evaluation and fine-tuning using **GRPO (Gradient Reward Policy Optimization)**.

---

## 📦 Setup

Run all commands below from the **project root**.

### 📁 Clone the Repository

```bash
git clone https://github.com/cayasryan/RL4Math.git
cd RL4Math
```
---

## 🧪 Reproducing Evaluation Results Only

If you're only interested in evaluating models (e.g., zero-shot performance on MATH500), set up your environment as follows:

```bash
conda create -n math_eval python=3.12 -y
conda activate math_eval
pip install -r reqs_eval.txt
```
---

## 🔁 Full Pipeline: Fine-tuning + Evaluation

If you intend to fine-tune models via GRPO and then evaluate them:

```bash
conda create -n math_finetune python=3.12 -y
conda activate math_finetune
pip install -r reqs_train.txt
```
---

## 🛠️ I. Generating Processed MATH Datasets

We prepare **three different types of datasets** based on prompt content and target token behavior:
- `no_level_type`
- `w_level_type`
- `variable_targets`

### 1. `no_level_type`

In this setting, the prompt **does not include** any information about problem type or difficulty. The format is:

> `<prompt> = <problem> Let's think step by step and output the final answer within \boxed{}. Think for maximum <target_tokens> tokens.`

```bash
python3 scripts/generate_processed_math.py \
  --math_dir MATH_data \
  --output_dir MATH_processed/no_level_type \
  --type no_level_type \
  --target_tokens 3600
```
---

### 2. `w_level_type`

This configuration includes **both difficulty and problem type** in the prompt. The model is instructed to adjust its reasoning length accordingly:

> `<prompt> = <problem> Let's think step by step and output the final answer within \boxed{}. The problem is of difficulty level <level> and its type is <type>. Think for maximum <target_tokens> tokens and shorten thinking time based on the difficulty and type.`

```bash
python3 scripts/generate_processed_math.py \
  --math_dir MATH_data \
  --output_dir MATH_processed/w_level_type \
  --type w_level_type \
  --target_tokens 3600
```
---

### 3. `variable_targets`

This uses the **same prompt format** as `w_level_type`, but the value of `<target_tokens>` **varies automatically** depending on the problem type and difficulty level.

```bash
python3 scripts/generate_processed_math.py \
  --math_dir MATH_data \
  --output_dir MATH_processed/variable_targets \
  --type variable_targets
```

---

## 🧬 II. Fine-tuning Models with GRPO

We use [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning via GRPO. To train your own model on the processed MATH data:

```bash
python train.py \
  --model_name_or_path unsloth/llama-3-1b-instruct \
  --train_file /path/to/train.parquet \
  --val_file /path/to/val.parquet \
  --output_dir ./checkpoints/grpo_llama3 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16
```
&gt; Make sure your processed datasets are generated beforehand (see Section I).

---

## 📊 III. Evaluating Models on MATH500

To evaluate a model on the MATH500 benchmark:

```bash
python evaluate_math500.py \
  --model_path ./checkpoints/grpo_llama3 \
  --test_file /path/to/test.parquet \
  --output_dir ./results/math500_eval \
  --batch_size 32
```
Results will be saved in the specified `output_dir` as a `.json` or `.csv` file depending on your implementation.

---

<!-- ## 📁 Directory Structure

your-repo-name/
├── MATH_processed/
│   └── generate_math.py
├── scripts/
│   └── utils.py
├── checkpoints/
│   └── grpo_llama3/
├── results/
├── train.py
├── evaluate_math500.py
├── reqs_eval.txt
├── reqs_train.txt
└── README.md -->

<!-- --- -->

<!-- ## 📌 Notes

- Ensure that your input CSVs (`math_train.csv`, `math_val.csv`, `math500_test.csv`) are inside the `MATH/` folder or your custom `--math_dir`.
- The GRPO implementation uses Unsloth for memory-efficient training with LoRA adapters.
- The code supports evaluation using chain-of-thought prompting and boxed-answer extraction. -->

<!-- --- -->

## 📜 Citation

Coming soon.
