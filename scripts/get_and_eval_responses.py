import torch
import numpy as np
import pandas as pd

import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from tabulate import tabulate

from transformers import AutoTokenizer, AutoModelForCausalLM

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from math_reward import math_reward_fn


# FOR REPRODUCIBILITY
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_responses(model, tokenizer, prompts, n_samples=1, max_new_tokens=256, batch_size=4):
    import torch
    from tqdm import tqdm

    responses = []
    token_lengths = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = [[] for _ in range(len(batch_prompts))]
        batch_token_lengths = [[] for _ in range(len(batch_prompts))]

        for _ in range(n_samples):
            # Tokenize the full batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )

            # Move to GPU
            inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate responses for the batch
            outputs = model.module.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # ← Disable sampling
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode and store responses
            for j, output in enumerate(outputs):
                gen_tokens = output[inputs["input_ids"].shape[1]:]  # Remove prompt tokens
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                batch_responses[j].append(decoded)
                batch_token_lengths[j].append(len(tokenizer.encode(decoded)))

        responses.extend(batch_responses)
        token_lengths.extend(batch_token_lengths)

    return responses, token_lengths

def evaluate_responses(responses, ground_truths, target_tokens, token_lengths):
    total_scores = []
    total_rewards = []
    passes = 0
    for resp_list, gt, target, used in tqdm(zip(responses, ground_truths, target_tokens, token_lengths), total=len(responses), desc="Evaluating"):
        used = list(used)
        resp_list = list(resp_list)
        rewards_list = [math_reward_fn(r, gt, target, u) for r, u in zip(list(resp_list), list(used))]
        total_rewards.append(rewards_list)
        score_list = [0 if reward == -1 else 1 for reward in rewards_list]
        total_scores.append(score_list)
        if np.max(score_list) == 1:
            passes += 1
    pass_at_1 = np.mean([s[0] for s in total_scores])
    pass_at_n = passes / len(responses)
    return pass_at_1, pass_at_n, total_scores, total_rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference with configurable paths and settings.")
    parser.add_argument('--model_path', type=str, default="models/L1-Qwen-1.5B-Max",
                        help="Path to the fine-tuned model directory.")
    parser.add_argument('--data_path', type=str, default="MATH_processed/val.parquet",
                        help="Path to the evaluation dataset.")
    parser.add_argument('--output_dir', type=str, default="eval_resuts/",
                        help="Directory to save model outputs.")
    parser.add_argument('--n_samples', type=int, default=1, help="Number of samples to generate.")
    parser.add_argument('--max_new_tokens', type=int, default=8000, help="Max number of new tokens to generate.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)  

    model_path = args.model_path
    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    n_samples = args.n_samples
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size

    # Extract model name and split name from paths
    model_name = os.path.basename(model_path.rstrip("/"))
    split_name = os.path.splitext(os.path.basename(data_path))[0]


    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = torch.nn.DataParallel(model)  # This will use all available GPUs
    model = model.cuda()

    model.eval()

    # Load data
    df = pd.read_parquet(data_path)

    prompts = [p[0]["content"] if isinstance(p, np.ndarray) and len(p) > 0 else "" for p in df["prompt"]]
    ground_truths = [rm["ground_truth"] for rm in df["reward_model"]]
    target_tokens = [rm["num_tokens"] for rm in df["reward_model"]]

    # Generate responses
    responses, token_lengths = generate_responses(model, tokenizer, prompts, n_samples=n_samples, max_new_tokens=max_new_tokens, batch_size=batch_size)

    # Save responses
    df["responses"] = responses
    df["token_lengths"] = token_lengths

    output_filename = f"Responses_{model_name}_{split_name}.parquet"
    output_parquet = os.path.join(output_dir, output_filename)
    df.to_parquet(output_parquet)
    print(f"Saved responses to {output_parquet}")

    # Evaluate
    pass_at_1, pass_at_n, total_scores, total_rewards = evaluate_responses(responses, ground_truths, target_tokens, token_lengths)
    print(f"pass@1: {pass_at_1:.4f}, pass@{n_samples}: {pass_at_n:.4f}")

    # Save metrics
    csv_path = os.path.join(output_dir, "pass.csv")
    row_data = {
        "model_path": model_path,
        "dataset": os.path.basename(data_path),
        "pass@1": pass_at_1,
        f"pass@{n_samples}": pass_at_n
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Save total_scores
    total_scores_df = pd.DataFrame(total_scores)
    total_rewards_df = pd.DataFrame(total_rewards)

    total_scores_filename = f"TotalScores_{model_name}_{split_name}.parquet"
    total_scores_df.to_parquet(os.path.join(output_dir, total_scores_filename))

    total_rewards_filename = f"TotalRewards_{model_name}_{split_name}.parquet"
    total_rewards_df.to_parquet(os.path.join(output_dir, total_rewards_filename))

    # Print summary
    table_data = [[k, v] for k, v in row_data.items()]
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))