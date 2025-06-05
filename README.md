<div align="center">
  <h1>Adaptive Thinking-Time Control and Problem-Aware Reasoning for Math Problems via Reinforcement Learning</h1>
</div>

---

## 🧠 Overview

This repository contains code to reproduce the results of our work on **Adaptive Thinking-Time Control** and **Problem-Aware Reasoning** for solving MATH problems using **Reinforcement Learning**. The code supports both evaluation and fine-tuning using **GRPO (Gradient Reward Policy Optimization)**.

---

## 📦 Setup

Run all commands below from the **project root**.

### 📁 Clone the Repository

```bash
git clone https://github.com/cayasryan/AdaptiveMathReasoning.git
cd AdaptiveMathReasoning
```
---

## 🧪 Set up the Environment

Create a new environment and install all package requirements:

```bash
conda create -n math_finetune python=3.12 -y
conda activate math_finetune
pip install -r reqs.txt
```
---

## 🛠️ I. Generating Processed MATH Datasets

We prepare **two different types of datasets** based on prompt content and target token behavior:
- `no_level_type`
- `w_level_type`

### 1. `no_level_type`

In this setting, the prompt **does not include** any information about problem type or difficulty. The format is:

> `<prompt> = <problem> Let's think step by step and output the final answer within \boxed{}. Think for maximum <target_tokens> tokens.`

```bash
python3 scripts/generate_processed_math.py \
  --math_dir MATH_data \
  --output_dir MATH_processed/no_level_type \
  --type no_level_type \
  --target_tokens 2000
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
  --target_tokens 2000
```
---

## 🧬 II. Fine-tuning Models with GRPO

We use [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning via GRPO.

Before running the training script, feel free to customize the training behavior by modifying the configuration file at `scripts/train_config.yaml`.

This file includes model paths, data paths, training hyperparameters, LoRA settings, and logging options (e.g. `wandb`). When reporting to Weights and Biases, make sure to login first via `wandb login`.

Make any changes you need before launching training:
```bash
python3 scripts/train_grpo_unsloth.py
```
> [!warning]
> Loading weights with BitsAndBytes quantization will take a while.  
> Do **not** exit prematurely — this is expected behavior.
---

## 📊 III. Evaluating Models on MATH500

To evaluate a model on the MATH500 benchmark, run the command below. The model_path can also be a Hugging Face location. For this work, we used `l3lab/L1-Qwen-1.5B-Max`.

```bash
python3 scripts/get_and_eval_responses.py \
  --model_path l3lab/L1-Qwen-1.5B-Max \
  --data_path MATH_processed/no_level_type/test.parquet \
  --output_dir eval_results/ \
  --n_samples 1 \
  --max_new_tokens 2000 \
  --batch_size 32
```
Results will be saved in the specified `output_dir` as `.parquet` files and a `.csv` file will also log evaluation scores.

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

<!-- ## 📜 Citation

Coming soon. -->
