import os
import pandas as pd
import random
from typing import Dict, Any, Optional, List

import sys
import argparse


# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import last_boxed_only_string, remove_boxed

def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string."""
    return remove_boxed(last_boxed_only_string(solution_str))

def make_map_fn(split: str, type_: str, num_tokens: int = -1):
    def process_fn_no_level_type(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example['problem']
        answer = example['answer'] 
        target_tokens = num_tokens if num_tokens != -1 else random.randint(100, 4000)
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        instruction += f" Think for maximum {abs(target_tokens)} tokens."
        question = f"{question} {instruction}"
        data = {
            "data_source": "MATH",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
                "num_tokens": target_tokens
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'level': example.get('level', 'unknown'),
                'type': example.get('type', 'unknown'),
            }
        }
        return data
    
    def process_fn_w_level_type(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example['problem']
        answer = example['answer'] 
        target_tokens = num_tokens if num_tokens != -1 else random.randint(100, 4000)
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        level = example.get('level', 'unknown')
        type_ = example.get('type', 'unknown')

        instruction += f"The problem is of difficulty level {level} and its type is {type_}. Think for maximum {abs(target_tokens)} tokens and shorten thinking time based on the difficulty and type."
        question = f"{question} {instruction}"
        data = {
            "data_source": "MATH",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
                "num_tokens": target_tokens
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'level': example.get('level', 'unknown'),
                'type': example.get('type', 'unknown'),
            }
        }
        return data 
    
    def process_fn_variable_targets(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example['problem']
        answer = example['answer'] 
     
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        level = example.get('level', 'unknown')
        type_ = example.get('type', 'unknown')

        # IF ELSE CONDITIONAL FOR VARYING TARGET TOKENS
        target_tokens = num_tokens if num_tokens != -1 else random.randint(100, 4000)
        instruction += f"The problem is of difficulty level {level} and its type is {type_}. Think for maximum {abs(target_tokens)} tokens and shorten thinking time based on the difficulty and type."
        # END IF ELSE CONDITIONAL
        
        question = f"{question} {instruction}"
        data = {
            "data_source": "MATH",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
                "num_tokens": target_tokens
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'level': example.get('level', 'unknown'),
                'type': example.get('type', 'unknown'),
            }
        }
        return data    

    if type_ == "no_level_type":
        process_fn = process_fn_no_level_type   
    elif type_ == "w_level_type":
        process_fn = process_fn_w_level_type
    elif type_ == "variable_targets":
        process_fn = process_fn_variable_targets
    else:
        raise ValueError(f"Unknown type: {type}. Supported types are 'no_level_type', 'w_level_type', and 'variable_targets'.")

    return process_fn

def process_math_csv(input_csv: str, split: str, output_parquet: str, type_: str, num_tokens: int = -1):
    df = pd.read_csv(input_csv)
    process_fn = make_map_fn(split, type_, num_tokens)
    processed_data = []
    for idx, row in df.iterrows():
        processed = process_fn(row, idx)
        if processed is not None:
            processed_data.append(processed)
    pd.DataFrame(processed_data).to_parquet(output_parquet)
    print(f"Saved {len(processed_data)} examples to {output_parquet}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process MATH dataset CSV files.")
    
    default_math_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "MATH"))
    default_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "MATH_processed"))
    
    parser.add_argument('--math_dir', type=str, default=default_math_dir, help="Directory containing MATH CSV files")
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help="Directory to save output parquet files")
    parser.add_argument('--type', type=str, default="w_level_type", choices=["no_level_type", "w_level_type", "variable_targets"],
                        help="Type of processing to apply to the dataset. Options: 'no_level_type', 'w_level_type', 'variable_targets'.")
    parser.add_argument('--target_tokens', type=int, default=3600, help="Number of tokens to process (use -1 for random)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    math_dir = args.math_dir
    output_dir = args.output_dir
    target_tokens = args.target_tokens
    type_ = args.type

    os.makedirs(output_dir, exist_ok=True)

    files = [
        ("math_train.csv", "train", "train.parquet"),
        ("math_val.csv", "val", "val.parquet"),
        ("math500_test.csv", "test", "test.parquet"),
    ]

    for csv_name, split, out_name in files:
        input_csv = os.path.join(math_dir, csv_name)
        output_parquet = os.path.join(output_dir, out_name)
        process_math_csv(input_csv, split, output_parquet, type_=type_, num_tokens=target_tokens)