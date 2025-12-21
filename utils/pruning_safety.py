from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
import copy

class PruningSafetyValidator:
    def validate_and_correct(self, llm_strategy, target_ratio, dataset, history, macs_overshoot_tolerance_pct=None, macs_undershoot_tolerance_pct=None):
        """Validate LLM strategy and apply corrections"""
        
        corrected_strategy = copy.deepcopy(llm_strategy)
        corrections_applied = []
        warnings = []
        
        if 'channel_pruning_ratio' in llm_strategy:
        # This is a CNN strategy - don't look for isomorphic ratios
            corrected_strategy['corrections_applied'] = corrections_applied
            corrected_strategy['safety_validated'] = True
            return corrected_strategy

        # Extract isomorphic ratios
        ratios = llm_strategy.get('isomorphic_group_ratios', {})
        if not ratios:
            warnings.append("No isomorphic ratios provided by LLM")
            return self.create_safe_fallback(dataset, target_ratio)
        
        # Validate each ratio type
        corrected_ratios = self._validate_individual_ratios(
            ratios, target_ratio, dataset, corrections_applied
        )
        
        # Apply historical learning
        corrected_ratios = self._apply_historical_learning(
            corrected_ratios, history, dataset, corrections_applied
        )
        
        # Final safety check
        corrected_ratios = self._final_safety_check(
            corrected_ratios, target_ratio, dataset, corrections_applied
        )
        
        # Update strategy
        corrected_strategy['isomorphic_group_ratios'] = corrected_ratios
        corrected_strategy['corrections_applied'] = corrections_applied
        corrected_strategy['safety_validated'] = True
        
        # Update rationale with corrections
        if corrections_applied:
            original_rationale = corrected_strategy.get('rationale', '')
            correction_summary = "; ".join(corrections_applied)
            corrected_strategy['rationale'] = f"{original_rationale}\n\n[SAFETY CORRECTIONS]: {correction_summary}"
        
        return corrected_strategy
    
    def _validate_individual_ratios(self, ratios, target_ratio, dataset, corrections):
        """CORRECTED: Use actually safe limits"""
        corrected = ratios.copy()
        
        if dataset.lower() == 'imagenet':
            # VERY CONSERVATIVE for ImageNet/DeiT
            max_mlp_pruning = target_ratio * 0.6   # 3% max MLP pruning
            max_qkv_pruning = target_ratio * 0.3  # 2% max attention pruning
        else:
            max_mlp_pruning = target_ratio * 0.2   # 3% max MLP pruning
            max_qkv_pruning = target_ratio * 0.1  
        
        # Check MLP ratio
        mlp_mult = ratios.get('mlp_multiplier', 0)
        actual_mlp_pruning = mlp_mult * target_ratio
        if actual_mlp_pruning > max_mlp_pruning:
            safe_multiplier = max_mlp_pruning / target_ratio
            corrected['mlp_multiplier'] = safe_multiplier
            corrections.append(f"MLP: {mlp_mult:.2f}→{safe_multiplier:.2f} "
                             f"(would cause {actual_mlp_pruning*100:.0f}% pruning, max {max_mlp_pruning*100:.0f}%)")
        
        # Check QKV ratio  
        qkv_mult = ratios.get('qkv_multiplier', 0)
        actual_qkv_pruning = qkv_mult * target_ratio
        if actual_qkv_pruning > max_qkv_pruning:
            safe_multiplier = max_qkv_pruning / target_ratio
            corrected['qkv_multiplier'] = safe_multiplier
            corrections.append(f"QKV: {qkv_mult:.2f}→{safe_multiplier:.2f} "
                             f"(would cause {actual_qkv_pruning*100:.0f}% pruning, max {max_qkv_pruning*100:.0f}%)")
        
        # Ensure critical ratios are 0
        if ratios.get('proj_multiplier', 0) != 0.0:
            corrected['proj_multiplier'] = 0.0
            corrections.append("PROJ: Set to 0.0 (output projections must not be pruned)")
        
        if ratios.get('head_multiplier', 0) != 0.0:
            corrected['head_multiplier'] = 0.0  
            corrections.append("HEAD: Set to 0.0 (attention heads must not be pruned)")
        
        return corrected
    
    def _apply_historical_learning(self, ratios, history, dataset, corrections, accuracy_threshold=1):
        """Learn from previous failures in history"""
        if not history:
            return ratios
        
        # Find patterns of failure
        failed_attempts = []
        for entry in history:
            if dataset.lower() == 'imagenet':
                accuracy = entry.get('fine_tuned_top1_accuracy', entry.get('zero_shot_top1_accuracy', 0))
                threshold = accuracy_threshold
            else:
                accuracy = entry.get('fine_tuned_accuracy', entry.get('zero_shot_accuracy', 0))
                threshold = accuracy_threshold            
            if accuracy < threshold:
                strategy = entry.get('strategy_used', {})
                failed_attempts.append({
                    'mlp_mult': strategy.get('mlp_multiplier', 0),
                    'qkv_mult': strategy.get('qkv_multiplier', 0),
                    'accuracy': accuracy
                })
        
        # Avoid patterns that led to failures
        corrected = ratios.copy()
        for failure in failed_attempts:
            if abs(ratios.get('mlp_multiplier', 0) - failure['mlp_mult']) < 0.1:
                new_mult = max(0.1, failure['mlp_mult'] - 0.2)  # More conservative
                corrected['mlp_multiplier'] = new_mult
                corrections.append(f"MLP: Avoided failed pattern {failure['mlp_mult']:.2f}→{new_mult:.2f} "
                                 f"(previous attempt achieved {failure['accuracy']:.1f}% accuracy)")
        
        return corrected
    
    def _final_safety_check(self, ratios, target_ratio, dataset, corrections):
        """Final comprehensive safety check"""
        
        # Calculate total theoretical parameter reduction
        mlp_reduction = ratios.get('mlp_multiplier', 0) * target_ratio
        qkv_reduction = ratios.get('qkv_multiplier', 0) * target_ratio
        
        # Estimate total impact (rough approximation)
        estimated_total_reduction = (mlp_reduction * 0.6) + (qkv_reduction * 0.4)  # Weighted by typical param distribution
        
        if dataset.lower() == 'imagenet' and estimated_total_reduction > 0.4:
            # Scale back both ratios proportionally
            scale_factor = 0.4 / estimated_total_reduction
            corrected = ratios.copy()
            corrected['mlp_multiplier'] *= scale_factor
            corrected['qkv_multiplier'] *= scale_factor
            corrections.append(f"TOTAL: Scaled ratios by {scale_factor:.2f} to prevent excessive total reduction")
            return corrected
        
        return ratios
    
    def apply_dataset_guardrails(self, strategy, target_ratio, dataset):
        """Apply final dataset-specific guardrails"""
        
        # This is the last line of defense
        ratios = strategy.get('isomorphic_group_ratios', {})
        
        if dataset.lower() == 'imagenet':
            # Absolute limits for ImageNet
            ratios['mlp_multiplier'] = min(ratios.get('mlp_multiplier', 0), 1.0)
            ratios['qkv_multiplier'] = min(ratios.get('qkv_multiplier', 0), 1.0)
        else:
            # More permissive for CIFAR-10
            ratios['mlp_multiplier'] = min(ratios.get('mlp_multiplier', 0), 1.6)
            ratios['qkv_multiplier'] = min(ratios.get('qkv_multiplier', 0), 1.2)
        
        # Always ensure these are 0
        ratios['proj_multiplier'] = 0.0
        ratios['head_multiplier'] = 0.0
        
        strategy['isomorphic_group_ratios'] = ratios
        strategy['final_guardrails_applied'] = True
        
        return strategy
    
    def create_safe_fallback(self, dataset, target_ratio):
        """Create guaranteed safe ratios when all else fails"""
        if dataset.lower() == 'imagenet':
            safe_ratios = {
                'qkv_multiplier': min(0.8, 0.25 / target_ratio),  # Very conservative
                'mlp_multiplier': min(1.0, 0.4 / target_ratio),   # Conservative
                'proj_multiplier': 0.0,
                'head_multiplier': 0.0
            }
        else:
            safe_ratios = {
                'qkv_multiplier': min(1.0, 0.5 / target_ratio),
                'mlp_multiplier': min(1.4, 0.7 / target_ratio),
                'proj_multiplier': 0.0, 
                'head_multiplier': 0.0
            }
        
        return {
            'isomorphic_group_ratios': safe_ratios,
            'importance_criterion': 'taylor',
            'round_to': 2 if dataset.lower() == 'imagenet' else 1,
            'global_pruning': True,
            'rationale': f'Research-based safe fallback for {dataset}: Taylor importance is SOTA with isomorphic pruning',
            'fallback_used': True,
            'safety_validated': True
        }

