import copy

class DatasetAwarePromptEnhancer:
    def create_enhanced_prompt(self, dataset, target_ratio, history, state):
        """Create comprehensive enhanced prompt with MACs-first guidance (keeps signature)"""

        # ---- Read MACs context (preferred) ----
        baseline_macs = (state or {}).get("baseline_macs")
        target_macs = (state or {}).get("target_macs")
        macs_overshoot_tolerance_pct = (state or {}).get("macs_overshoot_tolerance_pct", 1.0)
        macs_undershoot_tolerance_pct = (state or {}).get("macs_undershoot_tolerance_pct", 5.0)

        # ---- Fallback: infer target MACs from legacy ratio if needed ----
        if (target_macs is None) and (baseline_macs is not None) and (target_ratio is not None):
            target_macs = baseline_macs * (1.0 - float(target_ratio))

        # Build MACs-first sections
        constraints = self._get_dataset_constraints(dataset)
        examples = self._get_few_shot_examples(dataset, baseline_macs, target_macs)
        history_insights = self._analyze_history_patterns(history, dataset, baseline_macs)

        # Defensive strings
        base_str = f"{baseline_macs:.3f}G" if baseline_macs else "N/A"
        tgt_str = f"{target_macs:.3f}G" if target_macs else "N/A"

        # Calculate acceptable range
        if target_macs is not None:
            min_range = target_macs * (1 - macs_undershoot_tolerance_pct / 100)
            max_range = target_macs * (1 + macs_overshoot_tolerance_pct / 100)
            acceptable_range = f"{min_range:.3f}G - {max_range:.3f}G"
        else:
            acceptable_range = "N/A - N/A"

        enhanced_prompt = f"""
You are an expert in Vision Transformer pruning with deep knowledge of {dataset} requirements.

{constraints}

{examples}

{history_insights}

TARGET MACS GUIDANCE:
---------------------
Baseline MACs: {base_str}
Target MACs: {tgt_str}
Overshoot tolerance (strict): +{macs_overshoot_tolerance_pct:.1f}%
Undershoot tolerance (lenient): -{macs_undershoot_tolerance_pct:.1f}%
Acceptable range: {acceptable_range}

CALCULATION VALIDATION REQUIRED:
Before suggesting any multipliers, verify:
1. MLP actual pruning ≤ dataset limit (see constraints)
2. QKV actual pruning ≤ dataset limit (see constraints)
3. Your proposed plan is expected to land within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% of the target MACs

STEP-BY-STEP REASONING REQUIRED:
1. Propose a global_scale (0-1) that controls overall pruning pressure
2. Propose per-group multipliers (mlp_multiplier, qkv_multiplier, head_multiplier, proj_multiplier)
3. Explain how these choices should affect MACs (which groups reduce MACs most)
4. Verify within safety limits and MACs tolerance

Your response must include explicit calculation verification in the rationale.

Output format (JSON only, no markdown):
{{
  "importance_criterion": "taylor" or "l1norm" or "l2norm",
  "target_macs": {target_macs if target_macs is not None else "null"},
  "baseline_macs": {baseline_macs if baseline_macs is not None else "null"},
  "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct:.1f},
  "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct:.1f},
  "global_scale": float,
  "round_to": 1 or 2 or null,
  "global_pruning": true,
  "rationale": "Explicitly justify how the proposed global_scale and multipliers will keep achieved MACs within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance.",
  "dataset_optimized": true,
  "isomorphic_group_multipliers": {{
    "qkv_multiplier": float,
    "mlp_multiplier": float,
    "proj_multiplier": 0.0,
    "head_multiplier": 0.0
  }},
  "expected_achieved_macs": float,
  "calculation_verification": {{
    "within_safety_limits": boolean,
    "within_macs_tolerance": boolean  // Must be within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%
  }}
}}
"""
        return enhanced_prompt

    def _get_dataset_constraints(self, dataset):
        """Dataset constraints (unchanged API; now phrased without ratios)"""
        if dataset.lower() == 'imagenet':
            max_mlp = 0.15  # 15% MLP pruning cap (conservative)
            max_qkv = 0.10  # 10% QKV pruning cap (conservative)
            return f"""
CRITICAL IMAGENET CONSTRAINTS (NEVER EXCEED):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 MLP LAYERS: Maximum {max_mlp*100:.0f}% pruning allowed
🚨 QKV ATTENTION: Maximum {max_qkv*100:.0f}% pruning allowed
🚨 OUTPUT PROJECTIONS: 0% pruning (proj_multiplier = 0.0)
🚨 ATTENTION HEADS: 0% pruning (head_multiplier = 0.0)

WHY THESE LIMITS ARE CRITICAL:
- ImageNet requires complex 1000-class discrimination
- Over-pruning MLP/QKV collapses accuracy
- These limits help ensure models stay within MAC tolerance bounds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        else:  # CIFAR-10
            max_mlp = 0.20  # 20%
            max_qkv = 0.10  # 10%
            return f"""
CIFAR-10 CONSTRAINTS (More Flexible):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ MLP LAYERS: Maximum {max_mlp*100:.0f}% pruning allowed
✅ QKV ATTENTION: Maximum {max_qkv*100:.0f}% pruning allowed
✅ Proj/Heads: keep at 0% unless you can justify accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def _get_few_shot_examples(self, dataset, baseline_macs, target_macs):
        """Few-shot examples rewritten to MACs budgets"""
        base = f"{baseline_macs:.2f}G" if baseline_macs else "N/A"
        tgt  = f"{target_macs:.2f}G" if target_macs else "N/A"

        if dataset.lower() == 'imagenet':
            return f"""
REASONING EXAMPLES FOR IMAGENET (MACs-first):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example: Baseline={base}, Target={tgt}
SAFE approach:
- Propose global_scale around 0.85–0.95
- mlp_multiplier ≤ limit; qkv_multiplier ≤ limit
- Expect largest MACs drop from early high-res blocks and attention projections
- Verify expected_achieved_macs ≈ Target within tolerance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            return f"""
REASONING EXAMPLES FOR CIFAR-10 (MACs-first):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example: Baseline={base}, Target={tgt}
MODERATE approach:
- global_scale ~ 0.9–1.0
- mlp_multiplier up to limit, qkv_multiplier up to limit
- Keep proj/head at 0 unless accuracy remains stable
- Verify expected_achieved_macs ≈ Target within tolerance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def _analyze_history_patterns(self, history, dataset, baseline_macs):
        """Analyze history with MACs focus"""
        if not history:
            return "No previous attempts to learn from."

        insights = [f"\nHISTORY ANALYSIS FOR {dataset.upper()}:"]
        insights.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        def _g(x):
            return f"{x/1e9:.2f}G" if isinstance(x, (int, float)) and x else "N/A"

        # Compute typical scaling: requested → achieved MACs
        pairs = []
        for i, entry in enumerate(history):
            target_macs = entry.get('target_macs')
            achieved_macs = entry.get('achieved_macs') or entry.get('final_macs') or entry.get('macs_after')
            if target_macs and achieved_macs:
                pairs.append((float(target_macs), float(achieved_macs)))

            if dataset.lower() == 'imagenet':
                acc = entry.get('fine_tuned_top1_accuracy', entry.get('zero_shot_top1_accuracy', 0))
                acc_label = "Top-1"
            else:
                acc = entry.get('fine_tuned_accuracy', entry.get('zero_shot_accuracy', 0))
                acc_label = "Acc"

            strat = entry.get('strategy_used', {})
            mlp_mult = strat.get('mlp_multiplier', None)
            qkv_mult = strat.get('qkv_multiplier', None)

            insights.append(
                f"Attempt {i+1}: target={_g(target_macs)}, achieved={_g(achieved_macs)}, "
                f"{acc_label}={acc if acc is not None else 'N/A'}%, "
                f"mlp_mult={mlp_mult if mlp_mult is not None else 'N/A'}, "
                f"qkv_mult={qkv_mult if qkv_mult is not None else 'N/A'}"
            )

        if pairs:
            # Typical multiplicative bias: achieved ≈ k * target
            ks = [a / t for (t, a) in pairs if t > 0]
            k_mean = sum(ks) / len(ks) if ks else 1.0
            insights.append(f"Observed target→achieved MACs scaling k≈{k_mean:.3f}. Use this to calibrate expected_achieved_macs.")

        insights.append("Avoid repeating multiplier patterns that produced low accuracy; favor those near budget with higher accuracy.")
        insights.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(insights)

    # 1. FIX SAFETY LIMITS (kept as you had them; values are your call)
    def _get_max_mlp_ratio(self, dataset):
        """CORRECTED: Much more conservative limits"""
        if dataset.lower() == 'imagenet':
            return 0.4  # note: your original comment seems inconsistent with value; keep your chosen caps
        else:
            return 0.2

    def _get_max_qkv_ratio(self, dataset):
        """CORRECTED: Much more conservative attention pruning"""
        if dataset.lower() == 'imagenet':
            return 0.2
        else:
            return 0.1


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
    
    def apply_dataset_guardrails(self, strategy, target_ratio, dataset, macs_overshoot_tolerance_pct=None, macs_undershoot_tolerance_pct=None):
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
            'rationale': f'Safe fallback ratios for {dataset} - guaranteed not to cause catastrophic failure',
            'fallback_used': True,
            'safety_validated': True
        }

    
