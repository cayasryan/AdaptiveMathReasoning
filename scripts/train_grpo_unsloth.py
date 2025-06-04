from unsloth import FastLanguageModel, FastModel


from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

import os
import sys

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from math_reward import math_reward_fn


yaml_path = os.path.join(current_dir, "train_config.yaml")
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

if isinstance(config["learning_rate"], str):
    config["learning_rate"] = float(config["learning_rate"])

# Settings
model_path = config["model_path"]
data_path = config["data_path"]
output_dir = config["output_dir"]
run_name = config["run_name"]

max_seq_length = config["max_seq_length"]
lora_rank = config["lora_rank"]
use_small_dataset = config["use_small_dataset"]
gpu_model_memory_utilization = config["gpu_model_memory_utilization"]


train_dataset = load_dataset("parquet", data_files={"train": data_path}, split="train")
if use_small_dataset:
    train_dataset = train_dataset.select(range(10)) # for testing


report_to = config.get("report_to", "none")

if report_to == "wandb":
    import wandb
    wandb.init(
        project=config.get("wandb_project", "default_project"),
        name=config.get("wandb_run_name", "default_run"),
    )



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = gpu_model_memory_utilization, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)




# Math-specific reward function
def math_reward_func(prompts, completions, completion_ids, data_source, ability, reward_model, extra_info):
    rewards = []
    for i, (prompt, completion, c_ids, rm) in enumerate(zip(prompts, completions, completion_ids, reward_model)):
        # Calculate math-specific reward
        ground_truth = rm["ground_truth"]
        num_tokens = rm["num_tokens"]

        solution = completion[0]["content"].strip()

        # Count the number of tokens in the solution
        if isinstance(solution, str):
            solution_tokens = tokenizer(solution, return_tensors="pt").input_ids[0]
            num_solution_tokens = len(solution_tokens)
        else:
            num_solution_tokens = 0

        reward = math_reward_fn(solution, ground_truth, num_tokens=num_tokens, valid_response_length = num_solution_tokens)
        rewards.append(reward)

    return rewards




training_keys = GRPOConfig.__init__.__code__.co_varnames
training_args_dict = {k: v for k, v in config.items() if k in training_keys}

training_args = GRPOConfig(**training_args_dict)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=math_reward_func,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)


trainer.train()