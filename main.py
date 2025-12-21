import os
os.environ["WANDB_MODE"] = "disabled"

from utils.logging_wandb import add_wandb_args
from workflow import run_pruning_workflow
from utils.timing import time_it, time_it_async
from utils import logging_wandb
import asyncio
from utils.timing import profiler
import copy
from torchvision import datasets, transforms
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from deepseek_llm import DeepSeekLLM
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import torch
import torch.nn as nn
import torch_pruning as tp
import timm
import pbench
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
import re
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from dotenv import load_dotenv, find_dotenv
import openai
import nest_asyncio
import asyncio
import json
import random
from torch.utils.data import DataLoader
import traceback
from datasets import load_dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import glob
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import pbench.data.presets
import pbench.extension
import pbench.forward_patch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import functools
from contextlib import contextmanager
from collections import defaultdict
import wandb
from datetime import datetime
import math
import threading
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import gc
import psutil
import warnings
import traceback
from utils.timing import time_it, time_it_async, profiler
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dataset-aware pruning workflow with master agent')
    parser.add_argument('--model', type=str, default='resnet50', help='Model to prune')
    parser.add_argument('--macs_target_g', type=float, default=None, help='Target absolute MACs in G (e.g., 4.2 means 4.2G).')
    parser.add_argument('--macs_target_ratio', type=float, default=None, help='Target MACs as a fraction of baseline (e.g., 0.24).')
    parser.add_argument('--macs_overshoot_tolerance_pct', type=float, default=1.0, 
                    help='Tolerance for MACs overshoot (higher than target) in percent (default 1%)')
    parser.add_argument('--macs_undershoot_tolerance_pct', type=float, default=5.0, 
                    help='Tolerance for MACs undershoot (lower than target) in percent (default 5%)')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'], default='cifar10', 
                    help='Dataset to use (cifar10 or imagenet)')
    parser.add_argument('--data_path', type=str, default='./data', 
                    help='Path to dataset (for CIFAR-10: download dir, for ImageNet: arrow files dir)')
    parser.add_argument('--max_revisions', type=int, default=10, help='Maximum number of pruning iterations')
    parser.add_argument('--accuracy_threshold', type=float, default=None, 
                    help='Minimum acceptable accuracy (%) - auto-set based on dataset if not provided')
    parser.add_argument('--compact', action='store_true', help='Show extremely compact output')
    parser.add_argument('--validate', action='store_true', help='Run validation checks on results')
    parser.add_argument('--imagenet_test', action='store_true', 
                    help='Use 10% of ImageNet for testing (no effect on CIFAR-10)')
    parser.add_argument('--imagenet_subset', type=float, default=1.0, 
                    help='Fraction of ImageNet to use (0.1 = 10%, no effect on CIFAR-10)')
    parser.add_argument('--output_dir', type=str, default='./models', 
                   help='Directory to save the final best model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                   help='Directory to save intermediate checkpoints')

    # Add WandB arguments
    parser = add_wandb_args(parser)
    
    args = parser.parse_args()

    # Initialize WandB
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mac_tag = (f"{args.macs_target_g:.2f}G" if args.macs_target_g is not None
           else (f"{args.macs_target_ratio*100:.0f}pctMAC" if args.macs_target_ratio is not None else "MAC"))
        args.wandb_name = f"{args.model}_{args.dataset}_macs{mac_tag}_{timestamp}"
    
    # print(f"[WandB] Initializing WandB run for project '{args.wandb_project}' with name '{args.wandb_name}'")
    
    wandb_config = {
        # Model and dataset info
        'model_name': args.model,
        'dataset': args.dataset,
        'macs_target_g': args.macs_target_g,
        'macs_target_ratio': args.macs_target_ratio,
        'data_path': args.data_path,
        'max_revisions': args.max_revisions,
        'accuracy_threshold': args.accuracy_threshold,
        
        # All argparse arguments
        **vars(args),
        
        # Additional metadata
        'framework': 'torch_pruning',
        'script_version': '1.0',
        'timestamp': datetime.now().isoformat(),
    }
    
    os.environ["WANDB_SERVICE_WAIT"] = "360"  # Increase timeout to 2 minutes
    os.environ["WANDB_START_METHOD"] = "thread"  # Use thread instead of fork

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=wandb_config,
        mode=args.wandb_mode
    )
    
    # print("[WandB] WandB initialized successfully.", flush=True)

    # Set subset fraction
    subset_fraction = 1.0
    if args.dataset.lower() == 'imagenet':
        if args.imagenet_test:
            subset_fraction = 0.1
        else:
            subset_fraction = args.imagenet_subset

    # Log subset info to WandB
    if args.dataset.lower() == 'imagenet' and subset_fraction < 1.0:
        wandb.log({
            "config/imagenet_testing_mode": True,
            "config/imagenet_subset_fraction": subset_fraction,
            "config/estimated_speedup": 1/subset_fraction
        })

    # Set dataset-specific defaults if not provided
    if args.accuracy_threshold is None:
        if args.dataset.lower() == 'imagenet':
            args.accuracy_threshold = 1.0
        else:
            args.accuracy_threshold = 85.0

    # Create query with dataset information
    if args.macs_target_g is not None:
        query = f"Prune {args.model} to {args.macs_target_g:.2f}G MACs on {args.dataset.upper()}."
    elif args.macs_target_ratio is not None:
        query = f"Prune {args.model} to {args.macs_target_ratio*100:.0f}% of baseline MACs on {args.dataset.upper()}."
    else:
        query = f"Prune {args.model} to a specified MACs budget on {args.dataset.upper()}."

    # print(f"[🚀] Starting {args.dataset.upper()} pruning workflow with query: {query}")
    
    # Log initial configuration to WandB
    wandb.log({
        "workflow/started": True,
        "workflow/query": query,
        "config/final_accuracy_threshold": args.accuracy_threshold,
        "config/final_max_revisions": args.max_revisions,
    })

    # print(f"[📋] Dataset: {args.dataset}")
    # print(f"[📋] Data path: {args.data_path}")
    # print(f"[📋] Accuracy threshold: {args.accuracy_threshold}%")

    # Validate data path for ImageNet
    if args.dataset.lower() == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')

        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            error_msg = f"ImageNet directory structure not found in '{args.data_path}'!"
            # print(f"[❌] Error: {error_msg}")
            # print("Expected folders:")
            # print(f"  - {train_dir}")
            # print(f"  - {val_dir}")
            wandb.log({"workflow/error": error_msg, "workflow/failed": True})
            wandb.finish()
            exit(1)
        
        # print(f"[✅] Found ImageNet train/val folders at {args.data_path}")

    def setup_macs_targets(args):
        """Setup MACs targets from command line arguments"""
        macs_config = {}
        
        if args.macs_target_g is not None:
            macs_config['target_macs'] = args.macs_target_g * 1e9  # Convert G to ops
            print(f"[🔧] Set target MACs to {args.macs_target_g}G ({macs_config['target_macs']} operations)")
        elif args.macs_target_ratio is not None:
            macs_config['macs_target_ratio'] = args.macs_target_ratio
            print(f"[🔧] Set target MACs ratio to {args.macs_target_ratio}")
        
        macs_config['macs_overshoot_tolerance_pct'] = args.macs_overshoot_tolerance_pct
        macs_config['macs_undershoot_tolerance_pct'] = args.macs_undershoot_tolerance_pct

        return macs_config
    
    # Add this after parsing arguments
    macs_config = setup_macs_targets(args)

    # Update initial_state_mods
    initial_state_mods = {
        'max_revisions': args.max_revisions,
        'accuracy_threshold': args.accuracy_threshold,
        **macs_config  # ✅ ADD: Include MACs configuration
    }

    try:
        # Run the workflow with dataset support
        results = asyncio.run(run_pruning_workflow(
            model_name=args.model, 
            query=query, 
            dataset=args.dataset,
            data_path=args.data_path,
            state_mods=initial_state_mods,
            imagenet_subset=subset_fraction,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir
        ))

        if not results:
            print(f"[❌] {args.dataset.upper()} workflow failed!")
            wandb.log({"workflow/failed": True})
            exit(1)

        # Log successful completion
        wandb.log({"workflow/completed_successfully": True})
        
        # Print final model location
        if 'final_model_weights_path' in results and 'final_model_full_path' in results:
            print(f"\n[🎉] WORKFLOW COMPLETE!")
            # Log final paths to WandB
            wandb.log({
                "final_output/weights_path": results['final_model_weights_path'],
                "final_output/full_model_path": results['final_model_full_path'],
                "final_output/metadata_path": results.get('final_model_metadata_path'),
                "final_output/directory": args.output_dir
            })

        # Validate results if requested
        if args.validate:
            validate_results(results)

        # ✅ SIMPLIFIED: Use the workflow's selection instead of re-selecting
        dataset = args.dataset.lower()
        # Try to recover baseline MACs from common places (and handle *_g variants)
        baseline_macs = (
            results.get("baseline_macs")
            or results.get("pruning_results", {}).get("baseline_macs")
            or results.get("prune", {}).get("pruning_results", {}).get("baseline_macs")
        )
        if baseline_macs is None:
            baseline_macs = (
                results.get("baseline_macs")
                or results.get("pruning_results", {}).get("baseline_macs")
                or results.get("prune", {}).get("pruning_results", {}).get("baseline_macs")
            )
            if baseline_macs is not None:
                baseline_macs = float(baseline_macs) * 1e9  # convert G->raw MACs

        # Compute target_macs from args (prefer absolute, then ratio)
        if getattr(args, "macs_target_g", None) is not None:
            target_macs = float(args.macs_target_g) * 1e9
        elif getattr(args, "macs_target_ratio", None) is not None:
            if baseline_macs is None:
                raise ValueError(
                    "macs_target_ratio requires a known baseline_macs. "
                    "Pass --macs_target_g or ensure baseline_macs is available."
                )
            r = float(args.macs_target_ratio)
            if not (0 < r <= 1.0):
                raise ValueError("macs_target_ratio must be in (0, 1].")
            # If your ratio means 'target/baseline', this is correct:
            target_macs = float(baseline_macs) * r
            # If instead your ratio means 'reduction fraction', use:
            # target_macs = float(baseline_macs) * (1.0 - r)
        else:
            raise ValueError("Provide either --macs_target_g or --macs_target_ratio.")

        macs_overshoot_tolerance_pct = args.macs_overshoot_tolerance_pct
        macs_undershoot_tolerance_pct = args.macs_undershoot_tolerance_pct
        macs_overshoot_tol = float(target_macs) * (macs_overshoot_tolerance_pct / 100.0)
        macs_undershoot_tol = float(target_macs) * (macs_undershoot_tolerance_pct / 100.0)

        history = results.get("history", [])

        # Check for systematic failure - ONLY consider fine-tuned models
        accuracy_threshold = args.accuracy_threshold  # e.g. 1.0
        finetuned_models = 0
        successful_finetuned_models = 0

        for entry in history:
            # Try common accuracy field names in order of preference
            final_acc = (entry.get("fine_tuned_top1_accuracy") or 
                        entry.get("fine_tuned_accuracy") or 
                        entry.get("accuracy"))
            
            # Only count models that were actually fine-tuned
            if final_acc is not None:
                finetuned_models += 1
                if final_acc >= accuracy_threshold:
                    successful_finetuned_models += 1

        # Systematic failure = we have fine-tuned models but none meet threshold
        is_systematic_failure = (finetuned_models > 0 and successful_finetuned_models == 0)
        

        def _entry_achieved_macs(e):
            # direct absolute MACs if present
            val = e.get("achieved_macs") or e.get("final_macs") or e.get("macs_after")
            if val is not None:
                return float(val)
            # derive from baseline + reduction if available
            red = e.get("macs_reduction")
            if red is not None and baseline_macs is not None:
                # red assumed as fraction reduced (e.g., 0.35 → 35% reduction)
                return float(baseline_macs) * (1.0 - float(red))
            # last resort: None
            return None

        def _entry_achieved_macs_within_tolerance(achieved_macs, target_macs, overshoot_tol, undershoot_tol):
            if achieved_macs is None:
                return False
            overshoot = achieved_macs - target_macs
            if overshoot > 0:  # Model is bigger than target
                return overshoot <= overshoot_tol
            else:  # Model is smaller than target  
                return abs(overshoot) <= undershoot_tol


        all_candidates = []
        for entry in history:
            achieved_macs = _entry_achieved_macs(entry)
            if achieved_macs is None:
                continue

            within_tolerance = _entry_achieved_macs_within_tolerance(achieved_macs, target_macs, macs_overshoot_tol, macs_undershoot_tol)

            # Check if fine-tuned
            if dataset == 'imagenet':
                ft_acc = entry.get("fine_tuned_top1_accuracy")
            else:
                ft_acc = entry.get("fine_tuned_accuracy")

            if within_tolerance and ft_acc is not None:
                all_candidates.append(entry)

        print(f"\n[📊] Found {len(all_candidates)} candidate models within tolerance that were fine-tuned")

        # ✅ SIMPLIFIED: Use workflow's selection
        if 'final_model_selected' in results:
            best_result = results['final_model_selected']
            print(f"[🏆] Using workflow's selected best model")

        else:
            print(f"[ℹ️] No final model was preselected by the workflow; choosing best within MACs tolerance")
            # Pick the best by fine-tuned accuracy within tolerance; fall back to best zero-shot if needed
            def _score(e):
                if dataset == 'imagenet':
                    return (e.get("fine_tuned_top1_accuracy") is not None, e.get("fine_tuned_top1_accuracy") or -1.0,
                            e.get("zero_shot_top1_accuracy") or -1.0)
                else:
                    return (e.get("fine_tuned_accuracy") is not None, e.get("fine_tuned_accuracy") or -1.0,
                            e.get("zero_shot_accuracy") or -1.0)

            best_result = max(all_candidates, key=_score) if all_candidates else {}

        # Extract metrics for reporting
        achieved_macs = _entry_achieved_macs(best_result) or 0.0
        # macs_reduction is fraction reduced if provided; recompute if missing
        macs_reduction = best_result.get("macs_reduction")
        if macs_reduction is None and baseline_macs:
            macs_reduction = 1.0 - (achieved_macs / float(baseline_macs)) if achieved_macs > 0 else 0.0
        macs_reduction = float(macs_reduction or 0.0)

        # Keep ratio-related fields if present (for compatibility), but they no longer drive selection
        achieved_ratio = best_result.get("achieved_ratio", 0.0)
        target_ratio = getattr(args, "ratio", 0.0)

        # Accuracy fields
        if dataset == 'imagenet':
            zero_shot_acc = best_result.get("zero_shot_top1_accuracy", 0.0)
            ft_acc = best_result.get("fine_tuned_top1_accuracy")
            zero_shot_top5 = best_result.get("zero_shot_top5_accuracy", None)
            ft_top5 = best_result.get("fine_tuned_top5_accuracy", None)
            accuracy_label = "Top-1 Accuracy"
        else:
            zero_shot_acc = best_result.get("zero_shot_accuracy", 0.0)
            ft_acc = best_result.get("fine_tuned_accuracy")
            zero_shot_top5 = None
            ft_top5 = None
            accuracy_label = "Accuracy"


        if ft_acc is not None and achieved_macs > 0:
            # Check if the selected model meets both criteria
            passes_accuracy = ft_acc >= accuracy_threshold
            passes_mac_tolerance = _entry_achieved_macs_within_tolerance(
                achieved_macs, target_macs, macs_overshoot_tol, macs_undershoot_tol)
            
            model_success = passes_accuracy and passes_mac_tolerance
            
            print(f"\n[✅] Selected model evaluation:")
            print(f"  Fine-tuned accuracy: {ft_acc:.2f}% (threshold: {accuracy_threshold}%) - {'PASS' if passes_accuracy else 'FAIL'}")
            print(f"  MAC compliance: {achieved_macs/1e9:.3f}G (target: {target_macs/1e9:.3f}G) - {'PASS' if passes_mac_tolerance else 'FAIL'}")
            print(f"  Success criteria met: {model_success}")
        else:
            print(f"\n[❌] Success criteria met: False (no valid model found)")
            model_success = False

        # Fix/propagate MACs if missing in top-level results (optional)
        if macs_reduction == 0 and 'pruning_results' in results:
            macs_reduction = results['pruning_results'].get('macs_reduction', 0.0)

        # Debug information
        # print(f"\n[🔍] DEBUG - Best Result Selection:")
        print(f"   Target MACs: {target_macs/1e9:.3f}G  (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")
        print(f"   Acceptable range: {(target_macs-macs_undershoot_tol)/1e9:.3f}G to {(target_macs+macs_overshoot_tol)/1e9:.3f}G")
        if baseline_macs:
            print(f"   Baseline MACs: {baseline_macs/1e9:.3f}G  → Reduction target ≈ {(1 - target_macs/float(baseline_macs))*100:.2f}%")
        print(f"   Best result achieved MACs: {achieved_macs/1e9:.3f}G")
        print(f"   Best result MACs reduction: {macs_reduction*100:.2f}%")
        if dataset == 'imagenet':
            print(f"   Best result zero-shot Top-1: {zero_shot_acc:.2f}%")
            print(f"   Best result fine-tuned Top-1: {ft_acc if ft_acc is not None else 'N/A'}")
            if zero_shot_top5 is not None:
                print(f"   Best result zero-shot Top-5: {zero_shot_top5:.2f}%")
        else:
            print(f"   Best result zero-shot accuracy: {zero_shot_acc:.2f}%")
            print(f"   Best result fine-tuned accuracy: {ft_acc if ft_acc is not None else 'N/A'}")

        # Log final results to WandB (MACs-first; keep old keys for compatibility)
        final_wandb_log = {
            # MACs-centric
            "results/target_macs": target_macs / 1e9,
            "results/achieved_macs": achieved_macs / 1e9,
            "results/macs_error_pct": (achieved_macs - target_macs) / max(target_macs, 1e-9) * 100.0,
            "results/macs_reduction": macs_reduction,
            "results/macs_reduction_pct": macs_reduction * 100,
            "results/macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
            "results/macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
            # Back-compat (won't drive selection anymore)
            "results/achieved_ratio": achieved_ratio,
            "results/achieved_ratio_pct": achieved_ratio * 100,
            "results/target_ratio": target_ratio,
            "results/target_ratio_pct": target_ratio * 100,
            # Run metadata
            "results/total_iterations": results.get('revision_number', 0),
            "results/systematic_failure": is_systematic_failure,
            "results/candidates_found": len(all_candidates),
        }

        # Compact output option
        if args.compact and 'history' in results:
            print(f"\n{args.dataset.upper()} Compact Summary:")
            for entry in results['history']:
                revision = entry.get('revision', 0)
                baseline_macs = results.get("baseline_macs") or results.get("pruning_results", {}).get("baseline_macs") or results.get("prune", {}).get("pruning_results", {}).get("baseline_macs")
                e_achieved_macs = entry.get("achieved_macs") or entry.get("final_macs") or entry.get("macs_after")
                
                if dataset == 'imagenet':
                    accuracy = entry.get('fine_tuned_top1_accuracy', entry.get('zero_shot_top1_accuracy', entry.get('accuracy', 0)))
                else:
                    accuracy = entry.get('fine_tuned_accuracy', entry.get('zero_shot_accuracy', entry.get('accuracy', 0)))
                    
                strategy = entry.get('strategy_used', {})
                criterion = strategy.get('importance_criterion', 'unknown')
                if e_achieved_macs and baseline_macs:
                    e_macs_red = 1.0 - (float(e_achieved_macs) / float(baseline_macs))
                    achieved_str = f"{float(e_achieved_macs)/1e9:.2f}G ({e_macs_red*100:.1f}%)"
                else:
                    # fallback to ratio if MACs are missing
                    achieved = entry.get('achieved_ratio', 0)
                    achieved_str = f"{achieved*100:.2f}%"
                print(f"{revision}: {achieved_str} {accuracy:.2f}% {criterion}")

        elif results and 'evaluation_results' in results:
            eval_results = results['evaluation_results']

            # Better failure handling in summary
            if is_systematic_failure:
                print("\n" + "="*70)
                print(f"[💥] SYSTEMATIC FAILURE - {args.dataset.upper()}")
                print("="*70)
                # print(f"❌ Model: {args.model}")
                # print(f"❌ Dataset: {args.dataset.upper()}")
                target_macs = results.get("target_macs")
                if target_macs:
                    print(f"❌ Target MACs: {target_macs/1e9:.2f}G (TOO AGGRESSIVE)")
                elif hasattr(args, 'macs_target_ratio') and args.macs_target_ratio:
                    print(f"❌ Target pruning ratio: {args.macs_target_ratio*100:.1f}% (TOO AGGRESSIVE)")
                else:
                    print(f"❌ Pruning target too aggressive")
                print(f"❌ All {len(history)} attempts resulted in <{accuracy_threshold}% accuracy")
                print(f"❌ This indicates the pruning target is unrealistic")
                print("="*70)
                print(f"\n[💡] RECOMMENDATIONS:")
                print(f"1. 🎯 Use a looser MACs budget (e.g., +10–20% MACs)")
                print(f"2. 🔧 Verify model/dataset compatibility")
                print(f"3. 🧪 Test with CIFAR-10 first to validate implementation")
                print(f"4. 📚 Review pruning literature for typical ResNet50 limits on ImageNet")
                
                # Still show the "best" failed result
                print(f"\n[📊] Best Failed Result:")
                print(f"   Achieved pruning ratio: {achieved_ratio*100:.2f}%")
                print(f"   MACs reduction: {macs_reduction*100:.2f}%")
                print(f"   Zero-shot {accuracy_label}: {zero_shot_acc:.2f}% (CATASTROPHIC)")
                if dataset == 'imagenet' and zero_shot_top5:
                    print(f"   Zero-shot Top-5: {zero_shot_top5:.2f}%")
                
            else:
                # Normal summary for successful/partial runs
                print("\n" + "="*70)
                print(f"[📊] FINAL {args.dataset.upper()} PRUNING SUMMARY")
                print("="*70)
                print(f"Model: {args.model}")
                print(f"Dataset: {args.dataset.upper()}")
                target_macs = results.get("target_macs")
                baseline_macs = results.get("baseline_macs") or results.get("pruning_results", {}).get("baseline_macs") or results.get("prune", {}).get("pruning_results", {}).get("baseline_macs")
                achieved_macs = results.get("achieved_macs") or (baseline_macs * (1.0 - achieved_ratio) if (baseline_macs and achieved_ratio is not None) else None)

                if baseline_macs:
                    print(f"Baseline MACs: {float(baseline_macs/1e9)/1e9:.3f}G")
                if target_macs:
                    print(f"Target MACs: {float(target_macs/1e9)/1e9:.3f}G")
                if achieved_macs:
                    print(f"Achieved MACs: {float(achieved_macs/1e9)/1e9:.3f}G")

                print(f"MACs reduction: {macs_reduction*100:.2f}%")


                # Dataset-aware accuracy reporting
                if zero_shot_acc is not None:
                    print(f"Zero-shot {accuracy_label}: {zero_shot_acc:.2f}%")
                if ft_acc is not None:
                    print(f"Fine-tuned {accuracy_label}: {ft_acc:.2f}%")
                    
                # ImageNet Top-5 reporting
                if dataset == 'imagenet':
                    if zero_shot_top5 is not None:
                        print(f"Zero-shot Top-5 Accuracy: {zero_shot_top5:.2f}%")
                    if ft_top5 is not None:
                        print(f"Fine-tuned Top-5 Accuracy: {ft_top5:.2f}%")


                print(f"Success criteria met: {eval_results.get('success', False)}")
                print(f"Total iterations: {results.get('revision_number', 0)}")
                print("="*70)

            print(f"\n[⚙️] {args.dataset.upper()} Parameter Details:")
            last_strategy = None
            if 'history' in results and results['history']:
                last_strategy = results['history'][-1].get('strategy_used', {})

            if last_strategy:
                print(f"Importance criterion: {last_strategy.get('importance_criterion', 'unknown')}")
                print(f"Round-to value: {last_strategy.get('round_to', 'None')}")
                print(f"Global pruning: {last_strategy.get('global_pruning', True)}")
                print(f"Rationale: {last_strategy.get('rationale', 'Not provided')}")

            print(f"Fine-tuning method: {eval_results.get('fine_tuning_method', 'Unknown')}")

            # Dataset-aware history table
            if 'history' in results:
                print(f"\n[📜] {args.dataset.upper()} Pruning History Summary:")
                print("-"*80)
                
                if dataset == 'imagenet':
                    header = f"{'Iter':<5}{'Target MACs':<14}{'Achieved MACs':<16}{'Reduct%':<9}{'ZS Top-1':<10}{'FT Top-1':<10}{'ZS Top-5':<10}{'FT Top-5':<10}{'Criterion':<10}"

                else:
                    header = f"{'Iter':<5}{'Target MACs':<14}{'Achieved MACs':<16}{'Reduct%':<9}{'ZS Acc':<10}{'FT Acc':<10}{'Improve':<10}{'Criterion':<10}"

                
                print(header)
                print("-"*80)

                for i, entry in enumerate(results['history']):
                    baseline_macs = results.get("baseline_macs") or results.get("pruning_results", {}).get("baseline_macs") or results.get("prune", {}).get("pruning_results", {}).get("baseline_macs")
                    t_macs = entry.get('target_macs')
                    a_macs = entry.get('achieved_macs') or entry.get('final_macs') or entry.get('macs_after')
                    criterion = entry.get('strategy_used', {}).get('importance_criterion', 'N/A')

                    red_pct = ((baseline_macs - a_macs) / baseline_macs) * 100 if baseline_macs and a_macs else 0

                    if dataset == 'imagenet':
                        zs_top1 = entry.get('zero_shot_top1_accuracy', entry.get('zero_shot_accuracy', 0))
                        ft_top1 = entry.get('fine_tuned_top1_accuracy', entry.get('accuracy'))
                        zs_top5 = entry.get('zero_shot_top5_accuracy', 0)
                        ft_top5 = entry.get('fine_tuned_top5_accuracy')
                        
                        zs_top1_str = f"{zs_top1:.1f}%" if isinstance(zs_top1, (int, float)) and zs_top1 is not None else "N/A"
                        ft_top1_str = f"{ft_top1:.1f}%" if ft_top1 is not None else "N/A"
                        zs_top5_str = f"{zs_top5:.1f}%" if isinstance(zs_top5, (int, float)) and zs_top5 is not None else "N/A"
                        ft_top5_str = f"{ft_top5:.1f}%" if ft_top5 is not None else "N/A"
                        
                        t_macs_str = f"{float(t_macs)/1e9:>8.2f}G" if t_macs is not None else "    N/A    "
                        a_macs_str = f"{float(a_macs)/1e9:>8.2f}G" if a_macs is not None else "    N/A    "
                        print(f"{i:<5}{t_macs_str}    {a_macs_str}      {red_pct:>6.1f}%  {zs_top1_str:<10}{ft_top1_str:<10}{zs_top5_str:<10}{ft_top5_str:<10}{criterion:<10}")

                        
                    else:  # CIFAR-10
                        zs_acc = entry.get('zero_shot_accuracy', 0)
                        ft_acc = entry.get('fine_tuned_accuracy', entry.get('accuracy'))
                        
                        if ft_acc is None or zs_acc is None:
                            zs_display = f"{zs_acc:.1f}%" if isinstance(zs_acc, (int, float)) and zs_acc is not None else "N/A"
                            t_macs_str = f"{float(t_macs)/1e9:>8.2f}G" if t_macs is not None else "    N/A    "
                            a_macs_str = f"{float(a_macs)/1e9:>8.2f}G" if a_macs is not None else "    N/A    "
                            print(f"{i:<5}{t_macs_str}    {a_macs_str}      {red_pct:>6.1f}%  {zs_display:<10}Skipped  N/A     {criterion:<10}")
                        else:
                            imp = ft_acc - zs_acc if zs_acc is not None else 0
                            t_macs_str = f"{float(t_macs)/1e9:>8.2f}G" if t_macs is not None else "    N/A    "
                            a_macs_str = f"{float(a_macs)/1e9:>8.2f}G" if a_macs is not None else "    N/A    "
                            print(f"{i:<5}{t_macs_str}    {a_macs_str}      {red_pct:>6.1f}%  {zs_acc:.1f}%    {ft_acc:.1f}%  {imp:+.1f}%  {criterion:<10}")

            # Better recommendations based on failure type
            print(f"\n[📝] Final {args.dataset.upper()} Recommendation:")
            if is_systematic_failure:
                print(f"🚨 CRITICAL: Systematic failure detected - all models failed catastrophically")
                print(f"🎯 SOLUTION: Reduce pruning ratio to 10-20% and retry")
            elif eval_results.get('success', False):
                print(f"✅ The {args.dataset} pruning process was successful! The model meets the MACs budget and accuracy requirements.")
            else:
                if eval_results.get('passed_target_macs', eval_results.get('passed_target_ratio', False)):
                    print(f"⚠️ The {args.dataset} pruning achieved the MACs budget but did not meet the accuracy threshold.")
                elif eval_results.get('passed_accuracy_threshold', False):
                    print(f"⚠️ The {args.dataset} model maintains good accuracy but did not achieve the MACs budget.")
                else:
                    print(f"❌ The {args.dataset} pruning process did not meet either the ratio or accuracy requirements.")

                print("Consider:")
                print("1. Using a less aggressive pruning ratio")
                print("2. Trying a different importance criterion")
                print("3. Increasing the fine-tuning epochs")
                print("4. Using a different base model")
                
                if dataset == 'imagenet':
                    print("5. For ImageNet: Consider using pretrained models with stronger baseline performance")
                    print("6. For ImageNet: Try more conservative MACs budgets (only 10–30% MACs reduction at first)")

    except Exception as e:
        print(f"[❌] Error in workflow: {e}")
        wandb.log({"workflow/error": str(e), "workflow/failed": True})
        import traceback
        print(traceback.format_exc())
        raise
        
    finally:
        # Log timing summary to WandB
        profiler.get_summary()  # This prints the summary
        timing_metrics = {}
        for name, times in profiler.timings.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            timing_metrics[f"timing/{name.replace('.', '_')}_avg_seconds"] = avg_time
            timing_metrics[f"timing/{name.replace('.', '_')}_total_seconds"] = total_time
            timing_metrics[f"timing/{name.replace('.', '_')}_call_count"] = len(times)

        # Log overall workflow timing
        total_workflow_time = sum(sum(times) for times in profiler.timings.values())
        timing_metrics["timing/total_workflow_seconds"] = total_workflow_time
        timing_metrics["timing/total_workflow_minutes"] = total_workflow_time / 60

        wandb.log(timing_metrics)

        # Also create a summary table for the timing
        timing_data = []
        for name, times in profiler.timings.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            timing_data.append([name, avg_time, total_time, len(times)])

        timing_table = wandb.Table(
            columns=["Step", "Avg Time (s)", "Total Time (s)", "Call Count"],
            data=timing_data
        )
        wandb.log({"timing_breakdown": timing_table})
        
        # Always finish WandB run
        # print("[WandB] Finishing WandB run...")
        wandb.finish()


