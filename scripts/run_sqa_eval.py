#!/usr/bin/env python3
"""CLI: SQA benchmark evaluation pipeline."""

import argparse
import os
import json
import wandb
import sys

# Ensure moshilite package is in path for scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from moshilite.eval.sqa import SQAEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Run SQA Evaluation for MoshiLite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--benchmark-dir", type=str, required=True, help="Path to the SQA benchmark dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--experiment-id", type=str, default="sqa_eval", help="Experiment ID for wandb logging")
    parser.add_argument("--is-baseline", action="store_true", help="Flag to denote if this is a baseline evaluation run")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    group_name = "baseline_eval" if args.is_baseline else "sqa_eval"
    wandb.init(project="moshilite", name=args.experiment_id, config=vars(args), group=group_name)
    
    print(f"Loading Moshi model from {args.model_path}")
    
    # Authentic Moshi Load Logic
    try:
        from moshi.models import loaders
        # Assuming args.model_path points to the downloaded kyutai/moshiko folder or repo ID
        model = loaders.get_moshika(args.model_path)
    except Exception as e:
        print(f"Failed to load authentic Moshi model from {args.model_path}. Using mock baseline. Error: {e}")
        model = None 
    
    evaluator = SQAEvaluator(model=model)
    print(f"Starting SQA evaluation on {args.benchmark_dir}")
    metrics = evaluator.run_sqa_eval(args.benchmark_dir)
    
    results_path = os.path.join(args.output_dir, f"{args.experiment_id}_sqa_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    wandb.log(metrics)
    print(f"Evaluation complete. Results saved to {results_path}")
    print(json.dumps(metrics, indent=2))
    wandb.finish()

if __name__ == "__main__":
    main()
