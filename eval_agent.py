from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from llm.provider import get_llm
from data.loaders import get_imagenet_folder_loaders_pbench

from utils.timing import time, time_it, time_it_async
from utils import logging_wandb
import os


class EvaluationAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def _setup_dataset_test_data(self, dataset: str, data_path: str, batch_size=64, num_workers=16):
        """Setup dataset-appropriate test data for evaluation"""
        
        if dataset.lower() == 'imagenet':
            return self._setup_imagenet_test_data(data_path, batch_size, num_workers)
        else:
            return self._setup_cifar10_data(batch_size, num_workers)

    def _setup_cifar10_data(self, batch_size=64, num_workers=16):
        """Setup CIFAR-10 test data for evaluation"""
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((224, 224), antialias=True)
        ])

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return test_loader, test_dataset

    def _setup_imagenet_test_data(self, data_path: str, batch_size=64, num_workers=16):
        """Setup ImageNet test data using the working pbench approach"""
        _, val_loader = get_imagenet_folder_loaders_pbench(data_path, batch_size, num_workers)
        
        # For test dataset, we can use the val dataset
        test_dataset = val_loader.dataset
        return val_loader, test_dataset

    def _evaluate_model(self, model, test_loader, device, dataset: str):
        """Dataset-aware model evaluation"""
        model.eval()
        correct = 0
        total = 0
        
        # For ImageNet, also track Top-5
        if dataset.lower() == 'imagenet':
            correct_top5 = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle different data formats
                if isinstance(batch, dict):
                    # ImageNet arrow format
                    inputs = batch['pixel_values'].to(device)
                    targets = batch['label'].to(device)
                else:
                    # Standard CIFAR-10 format
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Top-5 accuracy for ImageNet
                if dataset.lower() == 'imagenet':
                    _, top5_preds = outputs.topk(5, 1, True, True)
                    correct_top5 += top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).sum().item()
        
        accuracy = 100. * correct / total
        
        if dataset.lower() == 'imagenet':
            top5_accuracy = 100. * correct_top5 / total
            return {
                'accuracy': accuracy,
                'top1_accuracy': accuracy,  # Explicit Top-1
                'top5_accuracy': top5_accuracy
            }
        else:
            return {'accuracy': accuracy}

    def _get_dataset_specific_thresholds(self, dataset: str):
        """Get dataset-specific success thresholds"""
        
        if dataset.lower() == 'imagenet':
            return {
                'accuracy_threshold': 1.0,  # Top-1 accuracy threshold
                'mac_tolerance_pct': 2.0,    # 2% MAC tolerance for ImageNet
                'top5_threshold': 90.0       # Top-5 accuracy threshold
            }
        else:  # CIFAR-10
            return {
                'accuracy_threshold': 85.0,  # Accuracy threshold
                'mac_tolerance_pct': 1.0,    # 1% MAC tolerance for CIFAR-10
                'top5_threshold': None       # Not applicable
            }

    def _extract_zero_shot_accuracy(self, state: Dict, dataset: str):
        """Extract zero-shot accuracy from stored pruning results"""
        
        # ✅ CRITICAL FIX: Get zero-shot results from pruning phase, not evaluation
        pruning_results = state.get('prune', {}).get('pruning_results', {})
        
        if dataset.lower() == 'imagenet':
            # Look for ImageNet-specific accuracy metrics
            zero_shot_top1 = pruning_results.get('zero_shot_top1_accuracy')
            zero_shot_top5 = pruning_results.get('zero_shot_top5_accuracy')
            
            # Fallback to history if not in pruning_results
            if zero_shot_top1 is None:
                history = state.get('history', [{}])
                last_entry = history[-1] if history else {}
                zero_shot_top1 = last_entry.get('zero_shot_top1_accuracy', 
                                              last_entry.get('zero_shot_accuracy', 0))
                zero_shot_top5 = last_entry.get('zero_shot_top5_accuracy', 0)
                
            return zero_shot_top1, zero_shot_top5
        else:
            # CIFAR-10 uses simple accuracy
            zero_shot_acc = pruning_results.get('zero_shot_accuracy')
            
            # Fallback to history if not in pruning_results
            if zero_shot_acc is None:
                history = state.get('history', [{}])
                last_entry = history[-1] if history else {}
                zero_shot_acc = last_entry.get('zero_shot_accuracy', 0)
                
            return zero_shot_acc, None

    def _update_history_with_correct_results(self, state: Dict, eval_results: Dict, dataset: str):
        """
        ENHANCED: Update history with fine-tuned results for candidate models
        REASON: Two-phase search needs fine-tuned accuracy to select best model
        """
        
        if 'history' not in state:
            return
        
        current_revision = state.get('revision_number', 0)
        
        # Find the history entry for current revision
        for entry in state['history']:
            if entry.get('revision') == current_revision:
                
                # Only update if this was a candidate model (within tolerance)
                print(f"[DEBUG] is_candidate_model :  {entry.get('is_candidate_model')}")
                if entry.get('is_candidate_model', False):
                    print(f"[📝] Updating candidate model history with fine-tuned results")
                    
                    # Add fine-tuned results based on dataset
                    if dataset.lower() == 'imagenet':
                        if 'fine_tuned_top1_accuracy' in eval_results:
                            entry['fine_tuned_top1_accuracy'] = eval_results['fine_tuned_top1_accuracy']
                        if 'fine_tuned_top5_accuracy' in eval_results:
                            entry['fine_tuned_top5_accuracy'] = eval_results['fine_tuned_top5_accuracy']
                    else:
                        if 'fine_tuned_accuracy' in eval_results:
                            entry['fine_tuned_accuracy'] = eval_results['fine_tuned_accuracy']
                    
                    # Improvement metrics
                    if 'top1_accuracy_improvement' in eval_results:
                        entry['top1_accuracy_improvement'] = eval_results['top1_accuracy_improvement']
                    if 'accuracy_improvement' in eval_results:
                        entry['accuracy_improvement'] = eval_results['accuracy_improvement']
                    
                    # Store fine-tuned model checkpoint reference
                    if 'fine_tuning_results' in state and 'checkpoint_path' in state['fine_tuning_results']:
                        entry['fine_tuned_checkpoint'] = state['fine_tuning_results']['checkpoint_path']
                    
                    print(f"[✅] Updated candidate history entry for revision {current_revision}")
                else:
                    print(f"[⚠️] Skipping history update - revision {current_revision} was not a candidate model")
                break

    @time_it_async("6. Evaluation Agent")
    async def evaluate(self, state: Dict) -> Dict:
        """Perform comprehensive dataset-aware evaluation of the model"""

        # Extract dataset information
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        data_path = state.get("data_path", "./data")
        
        print(f"\n[📊] Starting comprehensive {dataset.upper()} evaluation...")
        print(f"[📊] Dataset: {dataset} ({num_classes} classes)")

        # Check for pruning success
        pruning_success = (
            state.get("pruning_results", {}).get("success", False) or
            state.get("prune", {}).get("pruning_results", {}).get("success", False)
        )
        
        if not pruning_success:
            print("[⚠️] Skipping evaluation: pruning failed or model is missing.")
            state["evaluation_results"] = {"skipped": True, "dataset": dataset}
            return state

        try:
            # Get fine-tuned model from state (if it exists)
            fine_tuned_model = state.get('fine_tuning_results', {}).get('model')
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[💻] Using device: {device}")

            # Gather MAC-based pruning metrics
            pruning_results = state.get('prune', {}).get('pruning_results', {})
            if not pruning_results:
                pruning_results = state.get('pruning_results', {})
                
            # Extract MAC-based metrics instead of ratio-based
            achieved_macs = float(pruning_results.get('achieved_macs', pruning_results.get('final_macs_g', 0.0)))

            baseline_macs = float(state.get('baseline_macs') or 10.0)   

            def safe_float(value, default):
                """Safely convert a value to float with fallback"""
                if value is None:
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            target_macs = safe_float(state.get('target_macs'), 5.0)
            macs_overshoot_tolerance_pct = safe_float(state.get('macs_overshoot_tolerance_pct'), 1.0)
            macs_undershoot_tolerance_pct = safe_float(state.get('macs_undershoot_tolerance_pct'), 5.0)


            # Calculate MAC efficiency and error
            mac_efficiency = (achieved_macs / baseline_macs) * 100 if baseline_macs > 0 else 0
            mac_error_pct = ((achieved_macs - target_macs) / target_macs) * 100 if target_macs > 0 else 0

            # Get dataset-specific thresholds
            thresholds = self._get_dataset_specific_thresholds(dataset)

            eval_results = {
                'dataset': dataset,
                'num_classes': num_classes,
                'target_macs': target_macs,
                'achieved_macs': achieved_macs,
                'baseline_macs': baseline_macs,
                'mac_efficiency': mac_efficiency,
                'mac_error_pct': mac_error_pct,
                'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
                'accuracy_threshold': thresholds['accuracy_threshold']
            }

            # ✅ CRITICAL FIX: Use stored zero-shot results from pruning phase
            print(f"\n[📊] Extracting stored zero-shot results from pruning phase...")
            
            zero_shot_results = self._extract_zero_shot_accuracy(state, dataset)
            
            if dataset.lower() == 'imagenet':
                zero_shot_top1, zero_shot_top5 = zero_shot_results
                eval_results['zero_shot_top1_accuracy'] = float(zero_shot_top1 or 0)
                eval_results['zero_shot_top5_accuracy'] = float(zero_shot_top5 or 0)
                print(f"[📊] Stored Zero-shot - Top-1: {eval_results['zero_shot_top1_accuracy']:.2f}%, "
                      f"Top-5: {eval_results['zero_shot_top5_accuracy']:.2f}%")
            else:
                zero_shot_acc, _ = zero_shot_results
                eval_results['zero_shot_accuracy'] = float(zero_shot_acc or 0)
                print(f"[📊] Stored Zero-shot accuracy: {eval_results['zero_shot_accuracy']:.2f}%")

            # ✅ CRITICAL FIX: Only evaluate fine-tuned model if it exists and is different
            if fine_tuned_model is not None:
                print(f"\n[🧪] Evaluating fine-tuned model on {dataset}...")
                
                # Load test data for fine-tuned evaluation
                test_loader, test_dataset = self._setup_dataset_test_data(dataset, data_path)
                print(f"[📊] Test dataset size: {len(test_dataset)} samples")
                
                fine_tuned_model = fine_tuned_model.to(device)
                ft_results = self._evaluate_model(fine_tuned_model, test_loader, device, dataset)
                
                if dataset.lower() == 'imagenet':
                    eval_results['fine_tuned_top1_accuracy'] = float(ft_results['top1_accuracy'])
                    eval_results['fine_tuned_top5_accuracy'] = float(ft_results['top5_accuracy'])
                    print(f"[📊] Fine-tuned model - Top-1: {eval_results['fine_tuned_top1_accuracy']:.2f}%, "
                          f"Top-5: {eval_results['fine_tuned_top5_accuracy']:.2f}%")
                else:
                    eval_results['fine_tuned_accuracy'] = float(ft_results['accuracy'])
                    print(f"[📊] Fine-tuned model accuracy: {eval_results['fine_tuned_accuracy']:.2f}%")
            else:
                print(f"[📊] No fine-tuned model available - using zero-shot results only")

            # ✅ CRITICAL FIX: Compute CORRECT improvement using stored zero-shot results
            if dataset.lower() == 'imagenet':
                zero_shot = eval_results.get('zero_shot_top1_accuracy', 0)
                fine_tuned = eval_results.get('fine_tuned_top1_accuracy')
                
                if fine_tuned is not None and zero_shot is not None:
                    top1_improvement = fine_tuned - zero_shot
                    top1_pct = ((fine_tuned / zero_shot) - 1) * 100 if zero_shot > 0 else float('inf')
                    
                    eval_results['top1_accuracy_improvement'] = float(top1_improvement)
                    eval_results['top1_accuracy_improvement_percent'] = float(top1_pct)
                    
                    print(f"[📈] CORRECT Top-1 improvement: {top1_improvement:.2f}% "
                          f"({top1_pct:.1f}% relative)" if top1_pct != float('inf') else f"[📈] CORRECT Top-1 improvement: {top1_improvement:.2f}% (∞% relative)")
                    
                    # Also compute Top-5 improvement
                    zero_shot_top5 = eval_results.get('zero_shot_top5_accuracy', 0)
                    fine_tuned_top5 = eval_results.get('fine_tuned_top5_accuracy')
                    
                    if fine_tuned_top5 is not None and zero_shot_top5 is not None:
                        top5_improvement = fine_tuned_top5 - zero_shot_top5
                        eval_results['top5_accuracy_improvement'] = float(top5_improvement)
                        print(f"[📈] CORRECT Top-5 improvement: {top5_improvement:.2f}%")
                        
            else:  # CIFAR-10
                zero_shot = eval_results.get('zero_shot_accuracy', 0)
                fine_tuned = eval_results.get('fine_tuned_accuracy')
                
                if fine_tuned is not None and zero_shot is not None:
                    improvement = fine_tuned - zero_shot
                    pct = ((fine_tuned / zero_shot) - 1) * 100 if zero_shot > 0 else float('inf')
                    
                    eval_results['accuracy_improvement'] = float(improvement)
                    eval_results['accuracy_improvement_percent'] = float(pct)
                    
                    print(f"[📈] CORRECT Accuracy improvement: {improvement:.2f}% "
                          f"({pct:.1f}% relative)" if pct != float('inf') else f"[📈] CORRECT Accuracy improvement: {improvement:.2f}% (∞% relative)")

            # Check MAC deviation and success flags (dataset-aware)
            mac_overshoot_tolerance_g = target_macs * (macs_overshoot_tolerance_pct / 100.0)
            mac_undershoot_tolerance_g = target_macs * (macs_undershoot_tolerance_pct / 100.0)
            mac_overshoot_tolerance_g = target_macs * (macs_overshoot_tolerance_pct / 100.0)
            mac_undershoot_tolerance_g = target_macs * (macs_undershoot_tolerance_pct / 100.0)
            # Then check against the appropriate tolerance
            mac_error = achieved_macs - target_macs
            mac_within_tolerance = (-mac_undershoot_tolerance_g <= mac_error <= mac_overshoot_tolerance_g)
            thresh = state.get('accuracy_threshold', thresholds['accuracy_threshold'])
            
            # Get final accuracy (primary metric for each dataset)
            if dataset.lower() == 'imagenet':
                final_acc = eval_results.get('fine_tuned_top1_accuracy', eval_results.get('zero_shot_top1_accuracy', 0.0))
                final_top5 = eval_results.get('fine_tuned_top5_accuracy', eval_results.get('zero_shot_top5_accuracy', 0.0))
                passed_acc = final_acc >= thresh
                passed_top5 = final_top5 >= thresholds.get('top5_threshold', 90.0) if final_top5 > 0 else True
            else:
                final_acc = eval_results.get('fine_tuned_accuracy', eval_results.get('zero_shot_accuracy', 0.0))
                passed_acc = final_acc >= thresh
                passed_top5 = True  # Not applicable for CIFAR-10
            
            passed_mac_target = mac_within_tolerance

            eval_results.update({
                'mac_difference_g': float(abs(achieved_macs - target_macs)),
                'mac_error_pct': float(mac_error_pct),
                'passed_accuracy_threshold': passed_acc,
                'passed_mac_target': passed_mac_target,
                'final_accuracy': final_acc,
                'success': passed_acc and passed_mac_target and passed_top5
            })

            # Add ImageNet-specific success criteria
            if dataset.lower() == 'imagenet':
                eval_results['passed_top5_threshold'] = passed_top5
                eval_results['final_top5_accuracy'] = final_top5

            # ✅ CRITICAL FIX: Update history with CORRECT results
            self._update_history_with_correct_results(state, eval_results, dataset)

            # Print comprehensive MAC-aware results
            print(f"\n[✅] {dataset.upper()} Evaluation Complete:")
            print(f"  - Success: {eval_results['success']}")
            print(f"  - MAC Achievement: {achieved_macs/1e9:.3f}G (target: {target_macs/1e9:.3f}G)")
            print(f"  - MAC Efficiency: {mac_efficiency:.1f}% of baseline ({baseline_macs/1e9:.3f}G)")
            print(f"  - MAC Error: {mac_error_pct:+.1f}% (allowed: -{macs_undershoot_tolerance_pct:.1f}%…+{macs_overshoot_tolerance_pct:.1f}%)")

            
            if dataset.lower() == 'imagenet':
                print(f"  - Final Top-1 Accuracy: {final_acc:.2f}% (threshold: {thresh:.1f}%)")
                if final_top5 > 0:
                    print(f"  - Final Top-5 Accuracy: {final_top5:.2f}%")
                    
                # Print correct improvement
                if 'top1_accuracy_improvement' in eval_results:
                    print(f"  - Top-1 Improvement: {eval_results['top1_accuracy_improvement']:.2f}%")
                if 'top5_accuracy_improvement' in eval_results:
                    print(f"  - Top-5 Improvement: {eval_results['top5_accuracy_improvement']:.2f}%")
            else:
                print(f"  - Final Accuracy: {final_acc:.2f}% (threshold: {thresh:.1f}%)")
                if 'accuracy_improvement' in eval_results:
                    print(f"  - Accuracy Improvement: {eval_results['accuracy_improvement']:.2f}%")
            
            print(f"  - Overshoot tolerance: +{mac_overshoot_tolerance_g:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%)")
            print(f"  - Undershoot tolerance: -{mac_undershoot_tolerance_g:.3f}G (-{macs_undershoot_tolerance_pct:.1f}%)")


            eval_wandb_metrics = {
                "final_accuracy": final_acc,
                "success": eval_results['success'],
                "passed_accuracy_threshold": passed_acc,
                "passed_mac_target": passed_mac_target,
                "mac_error_pct": mac_error_pct,
                "target_macs": target_macs,
                "achieved_macs": achieved_macs,
                "mac_efficiency": mac_efficiency,
                "baseline_macs": baseline_macs,
            }

            # Add dataset-specific metrics
            if dataset.lower() == 'imagenet':
                if 'top1_accuracy_improvement' in eval_results:
                    eval_wandb_metrics["top1_improvement"] = eval_results['top1_accuracy_improvement']
                if final_top5 > 0:
                    eval_wandb_metrics["final_top5_accuracy"] = final_top5
            else:
                if 'accuracy_improvement' in eval_results:
                    eval_wandb_metrics["accuracy_improvement"] = eval_results['accuracy_improvement']

            logging_wandb.log_to_wandb(eval_wandb_metrics, step_name="evaluation", dataset=dataset)

            return {**state, 'evaluation_results': eval_results}

        except Exception as e:
            import traceback
            print(f"[❌] Error in {dataset} evaluation: {traceback.format_exc()}")
            return {**state, 'error': str(e)}        

