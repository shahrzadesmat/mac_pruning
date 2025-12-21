from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import traceback
from langchain_core.messages import SystemMessage, HumanMessage
from llm.provider import get_llm
from llm.prompts import MASTER_AGENT_PROMPT
from utils.analysis_structures import PruningState
import re
from utils.json_utils import parse_llm_json_response
import copy


class MasterAgent:
    # MAC-based tolerance thresholds
    MAC_TOLERANCE_OVERSHOOT_PCT = 1.0   # +1% above target MAC is OK (strict)
    MAC_TOLERANCE_UNDERSHOOT_PCT = 5.0  # -5% below target MAC is OK (lenient)

    def _create_mac_dataset_aware_prompt(self, dataset, num_classes, input_size, accuracy_threshold,
                                    baseline_macs, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct,
                                    profile_results, formatted_history, dataset_content):
        """Create a MAC-aware version of the Master Agent prompt"""
        
        # Convert to gigaflops for display if they're in operations
        if baseline_macs > 1e6:  # Assume it's in operations if > 1M
            baseline_macs_g = baseline_macs / 1e9
            target_macs_g = target_macs / 1e9
        else:  # Assume it's already in gigaflops
            baseline_macs_g = baseline_macs
            target_macs_g = target_macs
        
        # Calculate derived values that the template needs
        mac_efficiency_target = (target_macs_g / baseline_macs_g) * 100
        mac_reduction_needed = ((baseline_macs_g - target_macs_g) / baseline_macs_g) * 100
        
        overshoot_tolerance_value = target_macs_g * macs_overshoot_tolerance_pct / 100
        undershoot_tolerance_value = target_macs_g * macs_undershoot_tolerance_pct / 100
        min_target_macs_g = target_macs_g - undershoot_tolerance_value
        max_target_macs_g = target_macs_g + overshoot_tolerance_value

        # Create format dictionary with EXACT keys that the template expects
        format_dict = {
            'dataset': dataset,
            'num_classes': num_classes,
            'input_size': input_size,
            'accuracy_threshold': accuracy_threshold,
            'baseline_macs': baseline_macs_g,
            'target_macs': target_macs_g,
            'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
            'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
            'mac_reduction_needed': mac_reduction_needed,
            'mac_efficiency_target': mac_efficiency_target,
            'min_target_macs_g': min_target_macs_g,
            'max_target_macs_g': max_target_macs_g,
            'dataset_guidelines': dataset_content['dataset_guidelines'],
            'conservative_factor': dataset_content['recommended_approach'],
            'profile_results': profile_results,
            'pruning_history': formatted_history,
            'previous_strategies': formatted_history
        }

        return MASTER_AGENT_PROMPT.format(**format_dict)

    def _format_mac_history(self, history, dataset='cifar10', target_macs=5.0, baseline_macs=10.0, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Format history with MAC-aware analysis"""
        if not history:
            return f"No previous MAC pruning attempts for {dataset}."
        
        # Dataset-specific metric names
        if dataset.lower() == 'imagenet':
            accuracy_key = 'fine_tuned_top1_accuracy'
            backup_accuracy_key = 'zero_shot_top1_accuracy'
            metric_name = 'Top-1 Accuracy'
        else:
            accuracy_key = 'fine_tuned_accuracy'
            backup_accuracy_key = 'zero_shot_accuracy'
            metric_name = 'Accuracy'
        
        formatted = f"MAC-BASED PRUNING HISTORY WITH DATASET-AWARE ANALYSIS ({dataset.upper()}):\n"
        formatted += f"MAC TARGET: {target_macs:.3f}G from {baseline_macs:.3f}G baseline\n\n"
        
        # Track MAC-specific parameter values and outcomes
        mac_param_values = {
            'target_macs': [],
            'achieved_macs': [],
            'isomorphic_group_ratios': [],
            'channel_pruning_ratio': [],
            'importance_criterion': [],
            'round_to': []
        }
        
        mac_outcomes = {
            'mac_error_pct': [],
            'accuracy': [],
            'mac_efficiency': []
        }
        
        # Collect MAC data with dataset-aware accuracy extraction
        for entry in history:
            strategy = entry.get('strategy_used', {})
            
            # Track MAC parameters
            achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
            target_mac = entry.get('target_macs', target_macs)
            
            mac_param_values['target_macs'].append(target_mac)
            mac_param_values['achieved_macs'].append(achieved_macs)
            mac_param_values['isomorphic_group_ratios'].append(strategy.get('isomorphic_group_ratios', {}))
            mac_param_values['channel_pruning_ratio'].append(strategy.get('channel_pruning_ratio'))
            mac_param_values['importance_criterion'].append(strategy.get('importance_criterion'))
            mac_param_values['round_to'].append(strategy.get('round_to'))
            
            # Track MAC outcomes with dataset-aware accuracy
            accuracy = entry.get(accuracy_key, entry.get(backup_accuracy_key, 0))
            mac_outcomes['accuracy'].append(accuracy)
            
            if achieved_macs and target_mac:
                mac_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
                mac_efficiency = (achieved_macs / baseline_macs) * 100
                mac_outcomes['mac_error_pct'].append(mac_error_pct)
                mac_outcomes['mac_efficiency'].append(mac_efficiency)
            else:
                mac_outcomes['mac_error_pct'].append(None)
                mac_outcomes['mac_efficiency'].append(None)
        
        # Add MAC-specific parameter impact analysis
        formatted += f"MAC PARAMETER EFFECTS ANALYSIS FOR {dataset.upper()}:\n"
        
        # Analyze importance criterion impact with MAC context
        if len(set(mac_param_values['importance_criterion'])) > 1:
            formatted += f"\nIMPORTANCE CRITERION MAC IMPACT ON {dataset.upper()}:\n"
            criteria_results = {}
            
            for criterion in set(mac_param_values['importance_criterion']):
                if criterion:
                    criteria_results[criterion] = {'achieved_macs': [], 'accuracy': [], 'mac_error': []}
            
            for i, entry in enumerate(history):
                criterion = mac_param_values['importance_criterion'][i]
                if criterion and criterion in criteria_results:
                    achieved_macs = mac_param_values['achieved_macs'][i]
                    accuracy = mac_outcomes['accuracy'][i]
                    mac_error = mac_outcomes['mac_error_pct'][i]
                    
                    if achieved_macs:
                        criteria_results[criterion]['achieved_macs'].append(achieved_macs)
                    if accuracy:
                        criteria_results[criterion]['accuracy'].append(accuracy)
                    if mac_error is not None:
                        criteria_results[criterion]['mac_error'].append(mac_error)
            
            for criterion, results in criteria_results.items():
                if results['achieved_macs'] and results['accuracy']:
                    avg_macs = sum(results['achieved_macs']) / len(results['achieved_macs'])
                    avg_accuracy = sum(results['accuracy']) / len(results['accuracy'])
                    avg_error = sum(results['mac_error']) / len(results['mac_error']) if results['mac_error'] else None
                    
                    formatted += f"- {criterion} on {dataset}: avg {metric_name} {avg_accuracy:.2f}%, "
                    formatted += f"avg achieved MACs {avg_macs:.3f}G"
                    if avg_error is not None:
                        formatted += f", avg MAC error {avg_error:+.1f}%"
                    formatted += "\n"
        
        # Add MAC-specific insights
        formatted += f"\nMAC-SPECIFIC INSIGHTS FOR {dataset.upper()}:\n"
        
        if mac_outcomes['accuracy'] and mac_outcomes['mac_efficiency']:
            valid_accuracies = [acc for acc in mac_outcomes['accuracy'] if acc is not None]
            best_accuracy = max(valid_accuracies) if valid_accuracies else 0.0
            min_accuracy = min(valid_accuracies) if valid_accuracies else 0.0
            avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0.0
            best_mac_efficiency = min([eff for eff in mac_outcomes['mac_efficiency'] if eff is not None])
            
            formatted += f"- {metric_name} range: {min_accuracy:.2f}% to {best_accuracy:.2f}% (avg: {avg_accuracy:.2f}%)\n"
            formatted += f"- Best MAC efficiency: {best_mac_efficiency:.1f}% of baseline operations\n"
            formatted += f"- MAC target: {target_macs:.3f}G ({(target_macs/baseline_macs*100):.1f}% efficiency target)\n"
            
            # Dataset-specific MAC performance analysis
            target_efficiency = (target_macs / baseline_macs) * 100
            if dataset.lower() == 'imagenet':
                if best_accuracy >= 3 and best_mac_efficiency <= target_efficiency * 1.1:
                    formatted += "- Strong ImageNet performance at target MAC efficiency, focus on fine-tuning\n"
                elif best_accuracy >= 1:
                    formatted += "- Acceptable ImageNet performance, minor MAC allocation adjustments recommended\n"
                else:
                    formatted += "- ImageNet performance below threshold, consider more conservative MAC allocation\n"
            else:  # CIFAR-10
                if best_accuracy >= 90 and best_mac_efficiency <= target_efficiency * 1.1:
                    formatted += "- Excellent CIFAR-10 performance at MAC target, can try more aggressive MAC reduction\n"
                elif best_accuracy >= 85:
                    formatted += "- Good CIFAR-10 performance, current MAC strategy working well\n"
                else:
                    formatted += "- CIFAR-10 performance needs improvement, try different MAC allocation parameters\n"
        
        # Add detailed MAC entry information
        formatted += f"\nDETAILED {dataset.upper()} MAC HISTORY ENTRIES:\n"
        for i, entry in enumerate(history):
            formatted += f"MAC Attempt {i+1} ({dataset}):\n"
            
            strategy = entry.get('strategy_used', {})
            formatted += "MAC Parameters:\n"
            
            # Isomorphic group ratios for ViT
            isomorphic_group_ratios = strategy.get('isomorphic_group_ratios', {})
            if isomorphic_group_ratios:
                formatted += f"- Isomorphic Ratios: MLP {isomorphic_group_ratios.get('mlp_multiplier', 'N/A')}, "
                formatted += f"QKV {isomorphic_group_ratios.get('qkv_multiplier', 'N/A')}, "
                formatted += f"Proj {isomorphic_group_ratios.get('proj_multiplier', 'N/A')}, "
                formatted += f"Head {isomorphic_group_ratios.get('head_multiplier', 'N/A')}\n"
                        
            # Channel pruning for CNN
            channel_ratio = strategy.get('channel_pruning_ratio')
            if channel_ratio:
                formatted += f"- Channel pruning ratio: {channel_ratio:.3f}\n"
            
            formatted += f"- Importance criterion: {strategy.get('importance_criterion', 'N/A')}\n"
            formatted += f"- Round to: {strategy.get('round_to', 'N/A')}\n"
            
            formatted += "MAC Results:\n"
            target_mac = entry.get('target_macs', target_macs)
            achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
            
            formatted += f"- Target MACs: {target_mac:.3f}G\n"
            formatted += f"- Achieved MACs: {achieved_macs:.3f}G\n" if achieved_macs else "- Achieved MACs: N/A\n"
            
            if achieved_macs and target_mac:
                mac_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
                mac_efficiency = (achieved_macs / baseline_macs) * 100
                formatted += f"- MAC Error: {mac_error_pct:+.1f}%\n"
                formatted += f"- MAC Efficiency: {mac_efficiency:.1f}% of baseline\n"
            
            # Dataset-appropriate accuracy reporting
            accuracy = entry.get(accuracy_key, entry.get(backup_accuracy_key, 0))
            formatted += f"- {metric_name}: {accuracy:.2f}%\n"
            
            if dataset.lower() == 'imagenet':
                top5_acc = entry.get('fine_tuned_top5_accuracy', entry.get('zero_shot_top5_accuracy'))
                if top5_acc:
                    formatted += f"- Top-5 Accuracy: {top5_acc:.2f}%\n"
            
            formatted += "\n"
        
        return formatted
    
    def _get_default_mac_priorities(self, dataset):
        """Get dataset-specific default MAC parameter priorities"""
        if dataset.lower() == 'imagenet':
            return ["importance_criterion", "isomorphic_group_ratios", "round_to"]  # Importance first for accuracy
        else:
            return ["isomorphic_group_ratios", "importance_criterion", "round_to"]  # MAC allocation first for efficiency

    def _validate_mac_dataset_recommendations(self, directives_dict, dataset, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """Validate that LLM MAC recommendations are appropriate for the dataset"""
        validated = directives_dict.copy()
        
        # Validate MAC targets
        suggested_target_macs = directives_dict.get('target_macs', target_macs)
        
        if abs(suggested_target_macs - target_macs) > target_macs * 0.1:  # More than 10% deviation
            print(f"[⚠️] Adjusting MAC target from {suggested_target_macs:.3f}G to {target_macs/1e9:.3f}G")
            validated['target_macs'] = target_macs
            validated['mac_target_adjustment'] = "Restored to original MAC target"
        
        # Validate MAC allocation for ViT models
        isomorphic_group_ratios = directives_dict.get('isomorphic_group_ratios', {})
        if isomorphic_group_ratios:
            total_allocation = sum(isomorphic_group_ratios.values())
            if total_allocation > 100:
                # print(f"[⚠️] MAC allocation totals {total_allocation:.1f}% > 100%, normalizing")
                scale_factor = 100.0 / total_allocation
                for key in isomorphic_group_ratios:
                    isomorphic_group_ratios[key] *= scale_factor
                validated['isomorphic_group_ratios'] = isomorphic_group_ratios
                validated['isomorphic_group_ratios_normalized'] = True
        
        # Validate importance criterion for MAC efficiency
        importance = directives_dict.get('importance_criterion', 'taylor')
        if dataset.lower() == 'imagenet' and importance not in ['taylor', 'l2norm']:
            print(f"[⚠️] Recommending Taylor importance for ImageNet MAC efficiency instead of {importance}")
            validated['importance_criterion'] = 'taylor'
            validated['importance_adjustment'] = "Taylor recommended for ImageNet MAC efficiency"
        
        validated['dataset_adapted'] = True
        validated['mac_budget_focused'] = True
        return validated

    def _create_mac_fallback_response(self, response_content, dataset, state):
        """Create a MAC-aware fallback response when JSON parsing fails"""
        
        # Extract MAC parameters from state
        baseline_macs = state.get('baseline_macs', 10.0)
        target_macs = state.get('target_macs', 5.0)
        macs_overshoot_tolerance_pct=state.get('macs_overshoot_tolerance_pct', 2.0),
        macs_undershoot_tolerance_pct=state.get('macs_undershoot_tolerance_pct', 1.0),

        
        # Dataset-aware MAC fallback strategy
        if dataset.lower() == 'imagenet':
            exploration_strategy = f"Use conservative MAC allocation with Taylor importance for ImageNet complexity. Target: {target_macs:.3f}G MAC operations."
            isomorphic_group_ratios_priorities = ["importance_criterion", "isomorphic_group_ratios", "round_to"]
        else:
            exploration_strategy = f"Explore aggressive MAC allocation suitable for CIFAR-10. Target: {target_macs:.3f}G MAC operations."
            isomorphic_group_ratios_priorities = ["isomorphic_group_ratios", "importance_criterion", "round_to"]
        
        should_continue = True
        stop_reason = None
        
        # Try to extract MAC directives using regex
        directives_match = re.search(r'"directives"[:\s]+"([^"]+)"', response_content)
        if directives_match:
            exploration_strategy = directives_match.group(1).strip()
        
        # Try to extract MAC allocation priorities
        params_match = re.search(r'"isomorphic_group_ratios_tuning_order"[:\s]+\[(.*?)\]', response_content, re.DOTALL)
        if not params_match:
            params_match = re.search(r'"parameter_priorities"[:\s]+\[(.*?)\]', response_content, re.DOTALL)
        
        if params_match:
            params_str = params_match.group(1)
            priority_matches = re.findall(r'["\'](.*?)["\']', params_str)
            if priority_matches:
                isomorphic_group_ratios_priorities = priority_matches
        
        # Check for continue/stop with MAC context
        if re.search(r'"continue"[:\s]+(false|False)', response_content):
            should_continue = False
            reason_match = re.search(r'"stop_reason"[:\s]+["\'](.*?)["\']', response_content)
            if reason_match:
                stop_reason = reason_match.group(1).strip()
            else:
                stop_reason = f"MAC parsing failed for {dataset}, but stop detected in response."
        
        print(f"[⚠️] Using {dataset}-specific MAC fallback strategy: {exploration_strategy}")
        
        return {
            'master_results': {
                'directives': exploration_strategy,
                'isomorphic_group_ratios_priorities': isomorphic_group_ratios_priorities,
                'parameter_priorities': isomorphic_group_ratios_priorities,  # Legacy compatibility
                'exploration_strategy': exploration_strategy,
                'should_continue': should_continue,
                'stop_reason': stop_reason,
                'dataset': dataset,
                'fallback_used': True,
                'dataset_adapted': True,
                'mac_budget_focused': True,
                'target_macs': target_macs,
                'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
            }
        }
        
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    # ----------------------------------------------------------------------
    # Utility: dataset-aware "high-accuracy" cut-off
    # ----------------------------------------------------------------------
    def _get_high_accuracy_threshold(self, dataset: str) -> float:
        """
        Return the default accuracy threshold that defines a
        "high-accuracy" attempt when the user didn't supply one.

        • ImageNet  →  1 %   (new rule)
        • CIFAR-10 → 30 %
        """
        return 1.0 if dataset.lower() == "imagenet" else 30.0

    async def analyze_and_direct(self, state: PruningState) -> Dict:
        """MAC-aware analysis and directive generation for next pruning steps"""
        
        # Extract dataset and MAC information
        dataset = state.get('dataset', 'cifar10')
        num_classes = state.get('num_classes', 10)
        input_size = state.get('input_size', 224)
        accuracy_threshold = state.get('accuracy_threshold', 85.0)

        # MAC-specific parameters
        baseline_macs = state.get('baseline_macs')
        target_macs = state.get('target_macs')
        macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)
        
        baseline_macs = state.get('baseline_macs')

        if target_macs is None:
            raise ValueError("target_macs must be specified in state. Please provide target MAC operations (e.g., target_macs=8.8e9 for 8.8G operations)")

        # Add these debug lines RIGHT BEFORE the print statement:
        # print(f"[DEBUG] baseline_macs type: {type(baseline_macs)}, value: {baseline_macs}")
        # print(f"[DEBUG] target_macs type: {type(target_macs)}, value: {target_macs}")
        # print(f"[DEBUG] macs_overshoot_tolerance_pct type: {type(macs_overshoot_tolerance_pct)}, value: {macs_overshoot_tolerance_pct}")
        # print(f"[DEBUG] macs_undershoot_tolerance_pct type: {type(macs_undershoot_tolerance_pct)}, value: {macs_undershoot_tolerance_pct}")

        # print(f"[🧠] MAC-aware Master Agent analyzing {dataset} pruning strategy")
        # print(f"[🧠] Dataset context: {num_classes} classes, {input_size}x{input_size}, threshold: {accuracy_threshold}%")
        
        if baseline_macs is None:
            # Check if we can get it from profile_results
            profile_baseline = state.get('profile_results', {}).get('baseline_macs')
            if profile_baseline is not None:
                baseline_macs = profile_baseline
                # print(f"[DEBUG] Retrieved baseline_macs from profile_results: {baseline_macs/1e9:.3f}G")
            else:
                raise ValueError("baseline_macs not found in state or profile_results. Profiling may have failed.")

        print(f"[🧠] MAC context: {baseline_macs/1e9:.3f}G → {target_macs/1e9:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")

        
        # Get MAC-aware dataset-specific guidance
        dataset_content = self.get_dataset_specific_guidance(
            dataset, num_classes, accuracy_threshold, baseline_macs, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
        )
        
        # Get history and targets
        history = state.get('history', [])     
        # ========================================
        # MAC-BASED CATASTROPHIC FAILURE HANDLING
        # ========================================
        
        catastrophic_recovery = self._handle_mac_catastrophic_failures_correctly(
            history,
            target_macs,
            baseline_macs,
            macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct,
            state["dataset"],
        )

        if catastrophic_recovery:
            return {
                'master_results': {
                    'should_continue': True,
                    'recommended_strategy': catastrophic_recovery,
                    'strategic_guidance': catastrophic_recovery.get('strategic_guidance'),
                    'failure_recovery': True,
                    'failure_type': 'mac_config_catastrophic'
                }
            }
        
        # ========================================
        # NORMAL FLOW - NO FAILURE DETECTED
        # ========================================
        
        # Format MAC-aware pruning history with quality evaluations
        formatted_history = self._format_mac_history(history, dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
        
        # Get profile results
        profile_results = state.get('profile_results', {}).get('analysis', 'No profile yet.')
        
        # Create MAC-aware dataset prompt with JSON instructions
        prompt_text = self._create_mac_dataset_aware_prompt(
            dataset=dataset,
            num_classes=num_classes,
            input_size=input_size,
            accuracy_threshold=accuracy_threshold,
            baseline_macs=baseline_macs,
            target_macs=target_macs,
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
            profile_results=profile_results,
            formatted_history=formatted_history,
            dataset_content=dataset_content
        )
        
        # Add strict JSON formatting instructions
        json_instructions = """

CRITICAL JSON OUTPUT REQUIREMENTS:
================================
- Output ONLY valid JSON with NO comments
- Do NOT use # or // comments anywhere in the JSON
- Use double quotes for all strings
- No trailing commas
- No explanatory text before or after JSON
- No markdown code blocks (no ```)
- Example format:
{
"isomorphic_group_ratios_tuning_order": ["channel_pruning_ratio", "importance_criterion", "round_to"],
"target_macs": 5.000,
"macs_overshoot_tolerance_pct": 10.0,
"macs_undershoot_tolerance_pct": 5.0,
"rationale": "MAC-based detailed explanation here without any comments",
"dataset_adapted": true,
"mac_budget_focused": true,
"continue": true,
"stop_reason": null
}
"""
        
        final_prompt = prompt_text + json_instructions
        
        try:
            # Call LLM with the MAC-aware Master Agent prompt
            messages = [
                SystemMessage(content=final_prompt),
                HumanMessage(content=state.get('query', 'Provide MAC-based pruning directives.'))
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Debug print the raw response
            # print(f"[DEBUG] Raw LLM response from MAC Master Agent:\n{response.content}")
            
            # Use the robust JSON parsing function
            directives_dict = parse_llm_json_response(response.content)
            
            if not directives_dict:
                # print(f"[❌] MAC Master Agent JSON parsing failed - using fallback")
                return self._create_mac_fallback_response(response.content, dataset, state)
            
            # Extract MAC-specific fields with safe defaults
            should_continue = directives_dict.get('continue', True)
            stop_reason = directives_dict.get('stop_reason')
            exploration_strategy = directives_dict.get('directives', 
                            directives_dict.get('exploration_strategy',
                            f"Explore different MAC allocation configurations for {dataset}."))
            isomorphic_group_ratios_priorities = directives_dict.get('isomorphic_group_ratios_tuning_order',
                            directives_dict.get('parameter_priorities', 
                            self._get_default_mac_priorities(dataset)))
            
            # Validate MAC-specific recommendations
            validated_dict = self._validate_mac_dataset_recommendations(directives_dict, dataset, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
            
            print(f"[🧠] MAC Master Agent exploration strategy: {exploration_strategy}")
            print(f"[🧠] MAC allocation tuning priorities: {isomorphic_group_ratios_priorities}")
            print(f"[🧠] MAC budget focused: {validated_dict.get('mac_budget_focused', False)}")
            print(f"[🧠] Continue: {should_continue}")
            if not should_continue:
                print(f"[🛑] Stop reason: {stop_reason}")
            
            return {
                'master_results': {
                    'directives': exploration_strategy,
                    'isomorphic_group_ratios_priorities': isomorphic_group_ratios_priorities,
                    'parameter_priorities': isomorphic_group_ratios_priorities,  # Legacy compatibility
                    'exploration_strategy': exploration_strategy,
                    'should_continue': should_continue,
                    'stop_reason': stop_reason,
                    'dataset': dataset,
                    'dataset_adapted': validated_dict.get('dataset_adapted', False),
                    'mac_budget_focused': validated_dict.get('mac_budget_focused', True),
                    'target_macs': target_macs,
                    "macs_overshoot_tolerance_pct": 1.0,
                    "macs_undershoot_tolerance_pct": 5.0,
                    'llm_response_parsed': True
                }
            }
            
        except Exception as e:
            print(f"[❌] Error in MAC Master Agent LLM call: {e}")
            import traceback
            # print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # Use MAC fallback response
            return self._create_mac_fallback_response("LLM call failed", dataset, state)

    # ========================================
    # MAC-BASED CATASTROPHIC FAILURE LOGIC
    # ========================================
    def _handle_mac_catastrophic_failures_correctly(self, history, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, dataset, accuracy_threshold=None):
        
        accuracy_threshold = self._get_high_accuracy_threshold(dataset)

        # ---------- (1) MAC local-optimum escape ----------
        local_optimum_trap = self._detect_mac_local_optimum_trap(
            history, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, dataset
        )
        if local_optimum_trap:
            return self._generate_mac_exploration_strategy(
                local_optimum_trap, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
            )

        # ---------- (2) high-accuracy MAC near-misses ----------
        high_acc_near = []
        for entry in history:
            achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
            target_mac = entry.get('target_macs', target_macs)
            acc = entry.get(
                'zero_shot_top1_accuracy',
                entry.get('zero_shot_accuracy', 0)
            ) or 0
            
            if achieved_macs is not None and target_mac is not None:
                mac_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
                
                if (
                    -macs_undershoot_tolerance_pct <= mac_error_pct <= macs_overshoot_tolerance_pct
                    and acc > accuracy_threshold
                ):
                    high_acc_near.append(
                        dict(entry=entry, accuracy=acc,
                             mac_error_pct=mac_error_pct, achieved_macs=achieved_macs)
                    )

        if high_acc_near:
            best = max(high_acc_near, key=lambda x: x['accuracy'])
            return self._generate_mac_high_accuracy_preservation_guidance(
                best, target_macs, baseline_macs
            )

        # ---------- (3) MAC catastrophic-config collection ----------
        failed_configs = []
        for entry in history:
            achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
            target_mac = entry.get('target_macs', target_macs)
            
            if achieved_macs is not None and target_mac is not None:
                mac_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
                
                if mac_error_pct > macs_overshoot_tolerance_pct:
                    failure_type = "mac_overshoot"
                elif mac_error_pct < -macs_undershoot_tolerance_pct:
                    failure_type = "mac_undershoot"
                else:
                    failure_type = "acceptable"

                if failure_type != "acceptable":
                    strat = entry.get('strategy_used', {})
                    failed_configs.append({
                        'importance_criterion': strat.get('importance_criterion'),
                        'round_to': strat.get('round_to'),
                        'isomorphic_group_ratios': strat.get('isomorphic_group_ratios', {}),
                        'channel_pruning_ratio': strat.get('channel_pruning_ratio'),
                        'achieved_macs': achieved_macs,
                        'target_macs': target_mac,
                        'mac_error_pct': mac_error_pct,
                        'failure_type': failure_type,
                        'mac_distance_from_tolerance': max(mac_error_pct - macs_overshoot_tolerance_pct, -macs_undershoot_tolerance_pct - mac_error_pct, 0)
                    })

                    # print(f"[🚫] MAC CATASTROPHIC CONFIG ({failure_type}):")
                    # print(f"   Importance: {strat.get('importance_criterion')}")
                    print(f"   Round_to: {strat.get('round_to')}")
                    isomorphic_group_ratios = strat.get('isomorphic_group_ratios', {})
                    channel_ratio = strat.get('channel_pruning_ratio')
                    if isomorphic_group_ratios:  # ViT case
                        pass
                        # print(f"   QKV: {isomorphic_group_ratios.get('qkv_multiplier', 'N/A')}, MLP: {isomorphic_group_ratios.get('mlp_multiplier', 'N/A')}")
                    elif channel_ratio is not None:  # CNN case
                        pass
                        # print(f"   Channel_ratio: {channel_ratio}")
                    # print(f"   Target: {target_mac/1e9:.3f}G → Achieved: {achieved_macs/1e9:.3f}G")
                    # print(f"   MAC Error: {mac_error_pct:+.2f}% (tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")

        if failed_configs:
            print(f"[💡] Found {len(failed_configs)} MAC catastrophic configs")
            unders = [c for c in failed_configs if c['failure_type'] == 'mac_undershoot']
            overs = [c for c in failed_configs if c['failure_type'] == 'mac_overshoot']
            print(f"   MAC undershoots (insufficient pruning): {len(unders)}")
            print(f"   MAC overshoots (excessive pruning): {len(overs)}")

            return self._generate_mac_strategic_guidance_from_failures(
                failed_configs, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
            )

        return None
    
    def _detect_mac_local_optimum_trap(self, history, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, dataset, accuracy_threshold=None):
        """
        Detect if we're stuck trying to improve one MAC "excellent" result that keeps failing
        """
        if accuracy_threshold is None:
            accuracy_threshold = self._get_high_accuracy_threshold(dataset)
        if len(history) < 5:
            return None
        
        # Look for pattern: high accuracy result with good MAC achievement followed by repeated failures
        excellent_results = []
        recent_catastrophic = 0
        
        for i, entry in enumerate(history):
            accuracy = entry.get('zero_shot_top1_accuracy', entry.get('zero_shot_accuracy', 0)) or 0
            achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
            
            # Excellent result: high accuracy, close to MAC target
            if achieved_macs is not None and accuracy > accuracy_threshold:
                mac_error_pct = abs((achieved_macs - target_macs) / target_macs * 100)
                if mac_error_pct <= max(macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct) * 2:  # Within 2x max tolerance
                    excellent_results.append({
                        'index': i,
                        'accuracy': accuracy,
                        'achieved_macs': achieved_macs,
                        'mac_error_pct': mac_error_pct,
                        'entry': entry
                    })
            
            # Recent catastrophic failure
            if i >= len(history) - 5 and accuracy < 1:  # Last 5 attempts
                recent_catastrophic += 1
        
        # TRAP DETECTED: Have excellent MAC result(s) but recent attempts are catastrophic
        if excellent_results and recent_catastrophic >= 3:
            best_excellent = max(excellent_results, key=lambda x: x['accuracy'])
            
            print(f"[🕳️] MAC LOCAL OPTIMUM TRAP DETECTED:")
            print(f"   Excellent result: {best_excellent['accuracy']:.2f}% accuracy at {best_excellent['achieved_macs']/1e9:.3f}G MAC")
            print(f"   MAC target: {target_macs/1e9:.3f}G (error: {best_excellent['mac_error_pct']:+.1f}%)")
            print(f"   Recent failures: {recent_catastrophic}/5 attempts had <1% accuracy")
            print(f"   Problem: Stuck trying to improve one good MAC result, but small changes cause collapse")
            
            return {
                'trap_type': 'mac_local_optimum',
                'excellent_result': best_excellent,
                'recent_failures': recent_catastrophic,
                'pattern': 'excellent_mac_baseline_with_catastrophic_adjustments'
            }
        
        return None


    def _generate_mac_exploration_strategy(self, trap_info, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """
        Generate strategy to escape MAC local optimum trap through diversified exploration
        """
        excellent_result = trap_info['excellent_result']
        best_entry = excellent_result['entry']
        best_strategy = best_entry.get('strategy_used', {})
        
        # Get the isomorphic strategy that worked well
        isomorphic_group_ratios = best_strategy.get('isomorphic_group_ratios', {})
        working_mlp_multiplier = isomorphic_group_ratios.get('mlp_multiplier', 0.6)
        working_qkv_multiplier = isomorphic_group_ratios.get('qkv_multiplier', 0.4)
        working_channel_ratio = best_strategy.get('channel_pruning_ratio', 0.5)
        working_importance = best_strategy.get('importance_criterion', 'taylor')
        working_round_to = best_strategy.get('round_to', 2)
        
        overshoot_tolerance = macs_overshoot_tolerance_pct
        undershoot_tolerance = macs_undershoot_tolerance_pct
        
        strategic_guidance = {
            'guidance_type': 'escape_mac_local_optimum',
            'trap_detected': True,
            'excellent_baseline': {
                'accuracy': excellent_result['accuracy'],
                'achieved_macs': excellent_result['achieved_macs'],
                'mac_error_pct': excellent_result['mac_error_pct'],
                'working_strategy': best_strategy
            },
            'exploration_strategy': 'diversified_mac_search_around_working_baseline',
            'strategic_recommendations': [
                f"MAC LOCAL OPTIMUM TRAP: Excellent result {excellent_result['accuracy']:.1f}% accuracy at {excellent_result['achieved_macs']:.3f}G exists but adjustments fail",
                f"AVOID: Tiny adjustments to working multipliers {working_mlp_multiplier:.2f}/{working_qkv_multiplier:.2f}",
                f"STRATEGY: Try meaningfully different MAC approaches while staying conservative",
                f"PRESERVE: Working importance_criterion='{working_importance}' and round_to={working_round_to}",
                f"EXPLORE: Different multiplier combinations, not micro-adjustments"
            ],
            'exploration_directions': [
                {
                    'direction': 'more_conservative_multipliers',
                    'suggested_mlp_multiplier_range': [working_mlp_multiplier * 0.7, working_mlp_multiplier * 0.9],
                    'suggested_qkv_multiplier_range': [working_qkv_multiplier * 0.8, working_qkv_multiplier * 1.0],
                    'rationale': 'Reduce multipliers to preserve accuracy while staying near target'
                },
                {
                    'direction': 'shift_multiplier_balance',
                    'suggested_mlp_multiplier_range': [working_mlp_multiplier * 0.6, working_mlp_multiplier * 0.8],
                    'suggested_qkv_multiplier_range': [working_qkv_multiplier * 1.1, working_qkv_multiplier * 1.3],
                    'rationale': 'Shift emphasis from MLP to QKV layers'
                },
                {
                    'direction': 'try_different_importance_same_multipliers',
                    'preserve_multipliers': True,
                    'suggested_importance': ['l1norm', 'l2norm'] if working_importance == 'taylor' else ['taylor'],
                    'rationale': 'Same multipliers but different importance criterion'
                }
            ],
            'forbidden_actions': [
                'micro_adjustments_to_excellent_multiplier_baseline',
                f'multipliers_within_0.05_of_{working_mlp_multiplier:.2f}_{working_qkv_multiplier:.2f}',
                'repeating_any_failed_recent_multiplier_combination'
            ],
            'target_macs': target_macs,
            'baseline_macs': baseline_macs,
            'acceptable_mac_range': {
                'min_acceptable_macs_g': target_macs * (1 - undershoot_tolerance/100),
                'max_acceptable_macs_g': target_macs * (1 + overshoot_tolerance/100),
                'overshoot_tolerance_pct': overshoot_tolerance,
                'undershoot_tolerance_pct': undershoot_tolerance
            }
        }
        
        return {
            'strategic_guidance': strategic_guidance,
            'continue': True,
            'rationale': f'Escape MAC local optimum: Diversify multiplier search instead of micro-adjusting {excellent_result["accuracy"]:.1f}% baseline at {excellent_result["achieved_macs"]:.3f}G',
            'guidance_type': 'escape_mac_local_optimum'
        }


    def _generate_mac_high_accuracy_preservation_guidance(self, best_result, target_macs, baseline_macs):
        """
        NATURAL LEARNING: Let LLM analyze the MAC pattern and determine adjustment
        Remove all hardcoded magic numbers - just provide data for LLM analysis
        """
        
        entry = best_result['entry']
        accuracy = best_result['accuracy']
        achieved_macs = best_result['achieved_macs']
        mac_error_pct = best_result['mac_error_pct']
        
        # Get the successful MAC strategy details for LLM analysis
        strategy_used = entry.get('strategy_used', {})
        isomorphic_group_ratios = strategy_used.get('isomorphic_group_ratios', {})
        
        # ✅ NO MAGIC NUMBERS: Just provide data for LLM to analyze
        strategic_guidance = {
            'guidance_type': 'high_accuracy_mac_learning',  # MAC-specific learning
            'learning_data': {
                'successful_accuracy': accuracy,
                'successful_achieved_macs': achieved_macs,
                'target_macs': target_macs,
                'baseline_macs': baseline_macs,
                'mac_error_pct': mac_error_pct,
                'mac_deviation_from_target': achieved_macs - target_macs,
                'successful_strategy': strategy_used,
                'successful_isomorphic_group_ratios': {
                    'mlp_multiplier': isomorphic_group_ratios.get('mlp_multiplier'),
                    'qkv_multiplier': isomorphic_group_ratios.get('qkv_multiplier'),
                    'proj_multiplier': isomorphic_group_ratios.get('proj_multiplier'),
                    'head_multiplier': isomorphic_group_ratios.get('head_multiplier')
                }
            },
            'strategic_recommendations': [
                f"EXCELLENT MAC STRATEGY FOUND: {accuracy:.1f}% accuracy achieved",
                f"Strategy achieved {achieved_macs/1e9:.3f}G vs target {target_macs/1e9:.3f}G MAC",
                f"MAC Error: {mac_error_pct:+.1f}% ({'undershoot' if mac_error_pct < 0 else 'overshoot'})",
                f"LLM should analyze this MAC data and calculate optimal allocation adjustment",
                f"Preserve the importance_criterion and round_to that worked well",
                f"Calculate new MAC allocation percentages based on observed MAC reduction relationship"
            ],
            'learning_instructions': {
                'task': 'analyze_successful_mac_strategy_and_adjust',
                'preserve_base_strategy': True,
                'calculate_isomorphic_group_ratios_adjustment': True,
                'use_mathematical_mac_relationship': True,
                'avoid_magic_numbers': True
            },
            'mathematical_context': {
                'observed_relationship': f"MAC allocation {isomorphic_group_ratios.get('mlp_multiplier', 'unknown')}%/{isomorphic_group_ratios.get('qkv_multiplier', 'unknown')}% → {achieved_macs/1e9:.3f}G MAC operations",
                'target_relationship': f"need MAC allocation X%/Y% → {target_macs/1e9:.3f}G MAC operations",
                'mac_efficiency_observed': f"{(achieved_macs/baseline_macs)*100:.1f}% of baseline",
                'mac_efficiency_target': f"{(target_macs/baseline_macs)*100:.1f}% of baseline",
                'calculation_needed': True
            }
        }
        
        return {
            'strategic_guidance': strategic_guidance,
            'continue': True,
            'rationale': f'High-accuracy MAC learning mode: Let LLM analyze {accuracy:.1f}% accuracy strategy and calculate MAC allocation adjustment to hit {target_macs/1e9:.3f}G target',
            'guidance_type': 'high_accuracy_mac_learning'
        }


    def _generate_mac_strategic_guidance_from_failures(self, failed_configs, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """
        Generate STRATEGIC GUIDANCE for MAC-based failures
        """
        
        # ✅ NEW: Extract COMPLETE failure signatures including multipliers
        complete_failed_signatures = []
        failed_multiplier_patterns = {
            'undershoot_patterns': [],
            'overshoot_patterns': []
        }
        
        # Track unique configs to avoid duplicates
        seen_configs = set()

        for config in failed_configs:
            # Create a unique key for deduplication
            isomorphic_group_ratios = config.get('isomorphic_group_ratios', {})
            config_key = (
                config['importance_criterion'],
                config['round_to'],
                isomorphic_group_ratios.get('mlp_multiplier'),
                isomorphic_group_ratios.get('qkv_multiplier'),
                config['failure_type']
            )
            
            # Skip if we've already seen this exact configuration
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)
            
            complete_signature = {
                'importance_criterion': config['importance_criterion'],
                'round_to': config['round_to'],
                'mlp_multiplier': isomorphic_group_ratios.get('mlp_multiplier'),
                'qkv_multiplier': isomorphic_group_ratios.get('qkv_multiplier'),
                'proj_multiplier': isomorphic_group_ratios.get('proj_multiplier', 0.0),
                'head_multiplier': isomorphic_group_ratios.get('head_multiplier', 0.0),
                'failure_type': config['failure_type'],
                'achieved_macs': config['achieved_macs'],
                'target_macs': config['target_macs'],
                'mac_error_pct': config['mac_error_pct'],
                'mac_distance_from_tolerance': config.get('mac_distance_from_tolerance', 0)
            }
            
            # Only add if we have multiplier data
            if complete_signature['mlp_multiplier'] is not None and complete_signature['qkv_multiplier'] is not None:
                complete_failed_signatures.append(complete_signature)
                
                # ✅ NEW: Create danger zone patterns around failed multipliers
                tolerance = 0.05  # ±0.05 danger zone around failed multipliers
                
                danger_pattern = {
                    'mlp_range': (
                        max(0.0, complete_signature['mlp_multiplier'] - tolerance),
                        complete_signature['mlp_multiplier'] + tolerance
                    ),
                    'qkv_range': (
                        max(0.0, complete_signature['qkv_multiplier'] - tolerance),
                        complete_signature['qkv_multiplier'] + tolerance
                    ),
                    'importance_criterion': complete_signature['importance_criterion'],
                    'round_to': complete_signature['round_to'],
                    'achieved_macs_g': complete_signature['achieved_macs'] / 1e9,
                    'target_macs_g': complete_signature['target_macs'] / 1e9,
                    'mac_error_pct': complete_signature['mac_error_pct'],
                    'reason': f"Caused {complete_signature['mac_error_pct']:+.1f}% MAC error",
                    'severity': abs(complete_signature['mac_error_pct'])
                }
                
                if config['failure_type'] == 'mac_undershoot':
                    failed_multiplier_patterns['undershoot_patterns'].append(danger_pattern)
                elif config['failure_type'] == 'mac_overshoot':
                    failed_multiplier_patterns['overshoot_patterns'].append(danger_pattern)
        
        # ✅ NEW: Sort patterns by severity (worst failures first)
        failed_multiplier_patterns['undershoot_patterns'].sort(key=lambda x: x['severity'], reverse=True)
        failed_multiplier_patterns['overshoot_patterns'].sort(key=lambda x: x['severity'], reverse=True)
        
        # ✅ NEW: Print complete failure signature summary
        print(f"[🔍] Complete failure signatures found: {len(complete_failed_signatures)}")
        for i, sig in enumerate(complete_failed_signatures[:3]):  # Show first 3
            print(f"   Failure {i+1}: ({sig['importance_criterion']}, {sig['round_to']}, {sig['mlp_multiplier']:.2f}, {sig['qkv_multiplier']:.2f}) → {sig['mac_error_pct']:+.1f}% MAC error")
        
        print(f"   Undershoot danger zones: {len(failed_multiplier_patterns['undershoot_patterns'])}")
        print(f"   Overshoot danger zones: {len(failed_multiplier_patterns['overshoot_patterns'])}")
        
        # Show most severe failures for debugging
        if failed_multiplier_patterns['undershoot_patterns']:
            worst_undershoot = failed_multiplier_patterns['undershoot_patterns'][0]
            print(f"   Worst undershoot: mlp={worst_undershoot['mlp_range'][1]:.3f}, qkv={worst_undershoot['qkv_range'][1]:.3f} → {worst_undershoot['mac_error_pct']:+.1f}% error")
        
        # Generate MAC-STRATEGIC RECOMMENDATIONS (enhanced)
        recommendations = []
        failure_analysis = {}
        
        undershoot_patterns = failed_multiplier_patterns['undershoot_patterns']
        overshoot_patterns = failed_multiplier_patterns['overshoot_patterns']
        
        if undershoot_patterns:
            recommendations.append(f"Previous attempts achieved too many MACs (insufficient pruning) - need more aggressive MAC reduction to reach {target_macs/1e9:.3f}G")
            failure_analysis['mac_undershoot_pattern'] = True
            
            # ✅ NEW: Add multiplier-specific guidance for undershoots
            worst_patterns = undershoot_patterns[:3]  # Top 3 worst
            recommendations.append(f"DANGEROUS MULTIPLIER ZONES identified - avoid these ranges:")
            for i, pattern in enumerate(worst_patterns):
                mlp_min, mlp_max = pattern['mlp_range']
                qkv_min, qkv_max = pattern['qkv_range']
                recommendations.append(f"  Zone {i+1}: mlp_multiplier {mlp_min:.3f}-{mlp_max:.3f}, qkv_multiplier {qkv_min:.3f}-{qkv_max:.3f} (caused {pattern['mac_error_pct']:+.1f}% MAC error)")
            
            # Suggest safe ranges
            if len(worst_patterns) > 0:
                safe_mlp_max = min(p['mlp_range'][0] for p in worst_patterns) - 0.05
                safe_qkv_max = min(p['qkv_range'][0] for p in worst_patterns) - 0.05
                recommendations.append(f"SAFE MULTIPLIER SUGGESTION: Try mlp_multiplier ≤ {max(0.1, safe_mlp_max):.3f}, qkv_multiplier ≤ {max(0.1, safe_qkv_max):.3f}")
        
        if overshoot_patterns:
            recommendations.append(f"Some attempts achieved too few MACs (excessive pruning) - balance MAC reduction carefully around {target_macs/1e9:.3f}G target")
            failure_analysis['mac_overshoot_pattern'] = True
            
            # ✅ NEW: Add multiplier-specific guidance for overshoots
            recommendations.append("OVERLY AGGRESSIVE MULTIPLIER ZONES identified - these caused too much pruning")
        
        # Add MAC-strategic direction based on patterns
        if len(complete_failed_signatures) >= 3:
            recommendations.append("Multiple complete MAC configurations failed - try fundamentally different parameter combinations")
            failure_analysis['multiple_complete_failures'] = True
        
        # Add MAC-specific strategic guidance
        mac_overshoot_tolerance_g = target_macs * (macs_overshoot_tolerance_pct / 100)
        mac_undershoot_tolerance_g = target_macs * (macs_undershoot_tolerance_pct / 100)
        recommendations.extend([
            f"MAC target: {target_macs/1e9:.3f}G +{mac_overshoot_tolerance_g/1e9:.3f}G/-{mac_undershoot_tolerance_g/1e9:.3f}G (tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)",
            f"Acceptable MAC range: {(target_macs - mac_undershoot_tolerance_g)/1e9:.3f}G to {(target_macs + mac_overshoot_tolerance_g)/1e9:.3f}G",
            "Focus on MAC budget achievement over parameter reduction percentages",
            "MAC efficiency target: {:.1f}% of baseline operations".format((target_macs / baseline_macs) * 100)
        ])
        
        # ✅ ENHANCED: Generate MAC-STRATEGIC GUIDANCE structure with complete signatures only
        strategic_guidance = {
            # ✅ NEW: Complete failure tracking only
            'avoid_complete_signatures': complete_failed_signatures,
            'failed_multiplier_patterns': failed_multiplier_patterns,
            
            # MAC context
            'target_macs': target_macs,
            'baseline_macs': baseline_macs,
            'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
            'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
            'acceptable_mac_range': {
                'min_acceptable_macs_g': target_macs - mac_undershoot_tolerance_g,
                'max_acceptable_macs_g': target_macs + mac_overshoot_tolerance_g,
                'overshoot_tolerance_g': mac_overshoot_tolerance_g,
                'undershoot_tolerance_g': mac_undershoot_tolerance_g
            },
            
            # Guidance
            'strategic_recommendations': recommendations,
            'failure_analysis': failure_analysis,
            'guidance_type': 'mac_catastrophic_recovery',
            'priority_direction': 'achieve_mac_budget_within_tolerance',
            
            # ✅ NEW: Statistics for debugging
            'failure_statistics': {
                'total_failures': len(complete_failed_signatures),
                'undershoot_failures': len(failed_multiplier_patterns['undershoot_patterns']),
                'overshoot_failures': len(failed_multiplier_patterns['overshoot_patterns'])
            }
        }
        
        return {
            'strategic_guidance': strategic_guidance,
            'continue': True,
            'rationale': f'MAC strategic guidance: avoid {len(complete_failed_signatures)} complete failure signatures, prioritize achieving {target_macs/1e9:.3f}G MAC target',
            'guidance_type': 'mac_catastrophic_recovery',
            'failure_recovery': True
        }

    def get_dataset_specific_guidance(self, dataset, num_classes, accuracy_threshold, baseline_macs, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
            """Get MAC-aware dataset-specific guidance for the Master Agent"""
            
            mac_efficiency_target = (target_macs / baseline_macs) * 100
            mac_reduction_needed = ((baseline_macs - target_macs) / baseline_macs) * 100
            
            if dataset.lower() == 'imagenet':
                return {
                    'complexity_level': 'high',
                    'recommended_approach': 'conservative',
                    'typical_mac_efficiency_range': '60-80%',
                    'expected_accuracy_drop': '1-3%',
                    'fine_tuning_epochs': '5-10',
                    'dataset_guidelines': f"""
                    ImageNet MAC Master Agent Guidelines:
                    - Target MAC budget: {target_macs:.3f}G ({mac_efficiency_target:.1f}% efficiency)
                    - Consider conservative MAC allocation approaches for large models
                    - Expect 1-3% Top-1 accuracy drop as acceptable trade-off at {mac_efficiency_target:.1f}% MAC efficiency
                    - Pretrained weights provide strong starting point for MAC-efficient pruning
                    - MAC tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% around {target_macs:.3f}G target
                    """,
                    'early_stopping_criteria': [
                        'No MAC improvement in 3 consecutive iterations',
                        f'MAC operations consistently outside {target_macs:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) range'
                    ]
                }
            else:  # CIFAR-10
                return {
                    'complexity_level': 'moderate',
                    'recommended_approach': 'moderate to aggressive',
                    'typical_mac_efficiency_range': '20-50%',
                    'expected_accuracy_drop': '<1%',
                    'fine_tuning_epochs': '1-3',
                    'dataset_guidelines': f"""
                    CIFAR-10 MAC Master Agent Guidelines:
                    - Target MAC budget: {target_macs:.3f}G ({mac_efficiency_target:.1f}% efficiency)
                    - Can use aggressive MAC reduction (down to 20-50% of baseline)
                    - Expect <1% accuracy drop as target at {mac_efficiency_target:.1f}% MAC efficiency
                    - Short fine-tuning (1-3 epochs) typically sufficient for MAC-efficient models
                    - Training from scratch is feasible if MAC targets are very aggressive
                    - MAC tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% around {target_macs:.3f}G target
                    """,
                    'early_stopping_criteria': [
                        'No MAC improvement in 3 consecutive iterations',
                        f'MAC operations consistently outside {target_macs:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) range'
                    ]
                }

    def _create_dataset_aware_prompt(self, dataset, num_classes, input_size, accuracy_threshold,
                                   profile_results, formatted_history, target_ratio, dataset_content):
        """Create a dataset-aware version of the Master Agent prompt (legacy function for compatibility)"""
        
        return MASTER_AGENT_PROMPT.format(
            dataset=dataset,
            num_classes=num_classes,
            input_size=input_size,
            accuracy_threshold=accuracy_threshold,
            dataset_guidelines=dataset_content['dataset_guidelines'],
            conservative_factor=dataset_content['recommended_approach'],
            profile_results=profile_results,
            pruning_history=formatted_history,
            previous_strategies=formatted_history,
            target_ratio=target_ratio
        )

    def _format_history(self, history, dataset='cifar10'):
        """Format history with dataset-aware analysis (legacy function for compatibility)"""
        if not history:
            return f"No previous pruning attempts for {dataset}."
        
        # Dataset-specific metric names
        if dataset.lower() == 'imagenet':
            accuracy_key = 'fine_tuned_top1_accuracy'
            backup_accuracy_key = 'zero_shot_top1_accuracy'
            metric_name = 'Top-1 Accuracy'
        else:
            accuracy_key = 'fine_tuned_accuracy'
            backup_accuracy_key = 'zero_shot_accuracy'
            metric_name = 'Accuracy'
        
        formatted = f"PRUNING HISTORY WITH DATASET-AWARE ANALYSIS ({dataset.upper()}):\n\n"
        
        # Track parameter values and outcomes
        param_values = {
            'pruning_ratio': [],
            'importance_criterion': [],
            'round_to': []
        }
        
        outcomes = {
            'achieved_ratio': [],
            'accuracy': [],
            'target_deviation': []
        }
        
        # Collect data with dataset-aware accuracy extraction
        for entry in history:
            strategy = entry.get('strategy_used', {})
            
            # Track parameters
            param_values['pruning_ratio'].append(
                strategy.get('pruning_ratio_requested', strategy.get('pruning_ratio', None))
            )
            param_values['importance_criterion'].append(strategy.get('importance_criterion'))
            param_values['round_to'].append(strategy.get('round_to'))
            
            # Track outcomes with dataset-aware accuracy
            target = entry.get('target_ratio')
            achieved = entry.get('achieved_ratio')
            accuracy = entry.get(accuracy_key, entry.get(backup_accuracy_key, 0))
            
            outcomes['achieved_ratio'].append(achieved)
            outcomes['accuracy'].append(accuracy)
            
            if target and achieved:
                outcomes['target_deviation'].append(abs(target - achieved))
        
        # Add dataset-specific parameter impact analysis
        formatted += f"PARAMETER EFFECTS ANALYSIS FOR {dataset.upper()}:\n"
        
        # Analyze importance criterion impact with dataset context
        if len(set(param_values['importance_criterion'])) > 1:
            formatted += f"\nIMPORTANCE CRITERION IMPACT ON {dataset.upper()}:\n"
            criteria_results = {}
            
            for criterion in set(param_values['importance_criterion']):
                if criterion:
                    criteria_results[criterion] = {'achieved': [], 'accuracy': [], 'deviation': []}
            
            for i, entry in enumerate(history):
                criterion = param_values['importance_criterion'][i]
                if criterion and criterion in criteria_results:
                    achieved = outcomes['achieved_ratio'][i]
                    accuracy = outcomes['accuracy'][i]
                    deviation = outcomes['target_deviation'][i]
                    
                    if achieved:
                        criteria_results[criterion]['achieved'].append(achieved)
                    if accuracy:
                        criteria_results[criterion]['accuracy'].append(accuracy)
                    if deviation:
                        criteria_results[criterion]['deviation'].append(deviation)
            
            for criterion, results in criteria_results.items():
                if results['achieved'] and results['accuracy']:
                    avg_achieved = sum(results['achieved']) / len(results['achieved'])
                    avg_accuracy = sum(results['accuracy']) / len(results['accuracy'])
                    avg_deviation = sum(results['deviation']) / len(results['deviation']) if results['deviation'] else None
                    
                    formatted += f"- {criterion} on {dataset}: avg {metric_name} {avg_accuracy:.2f}%, "
                    formatted += f"avg achieved ratio {avg_achieved:.4f}"
                    if avg_deviation:
                        formatted += f", avg deviation {avg_deviation:.4f}"
                    formatted += "\n"
        
        # Add dataset-specific recommendations based on history
        formatted += f"\nDATASET-SPECIFIC INSIGHTS FOR {dataset.upper()}:\n"
        
        if outcomes['accuracy']:
            best_accuracy = max(outcomes['accuracy'])
            worst_accuracy = min(outcomes['accuracy'])
            avg_accuracy = sum(outcomes['accuracy']) / len(outcomes['accuracy'])
            
            formatted += f"- {metric_name} range: {worst_accuracy:.2f}% to {best_accuracy:.2f}% (avg: {avg_accuracy:.2f}%)\n"
            
            if dataset.lower() == 'imagenet':
                if best_accuracy >= 3:
                    formatted += "- Strong ImageNet performance achieved, focus on efficiency optimization\n"
                elif best_accuracy >= 1:
                    formatted += "- Acceptable ImageNet performance, minor adjustments recommended\n"
                else:
                    formatted += "- ImageNet performance below threshold, consider more conservative pruning\n"
            else:  # CIFAR-10
                if best_accuracy >= 90:
                    formatted += "- Excellent CIFAR-10 performance, can try more aggressive pruning\n"
                elif best_accuracy >= 85:
                    formatted += "- Good CIFAR-10 performance, current strategy working well\n"
                else:
                    formatted += "- CIFAR-10 performance needs improvement, try different parameters\n"
        
        # Add detailed entry information with dataset context
        formatted += f"\nDETAILED {dataset.upper()} HISTORY ENTRIES:\n"
        for i, entry in enumerate(history):
            formatted += f"Attempt {i+1} ({dataset}):\n"
            
            strategy = entry.get('strategy_used', {})
            formatted += "Parameters:\n"
            formatted += f"- Pruning ratio: {strategy.get('pruning_ratio_requested', strategy.get('pruning_ratio', 'N/A'))}\n"
            formatted += f"- Importance criterion: {strategy.get('importance_criterion', 'N/A')}\n"
            formatted += f"- Round to: {strategy.get('round_to', 'N/A')}\n"
            
            formatted += "Results:\n"
            formatted += f"- Target ratio: {entry.get('target_ratio', 'N/A')}\n"
            formatted += f"- Achieved ratio: {entry.get('achieved_ratio', 'N/A')}\n"
            
            # Dataset-appropriate accuracy reporting
            accuracy = entry.get(accuracy_key, entry.get(backup_accuracy_key, 0))
            formatted += f"- {metric_name}: {accuracy:.2f}%\n"
            
            if dataset.lower() == 'imagenet':
                top5_acc = entry.get('fine_tuned_top5_accuracy', entry.get('zero_shot_top5_accuracy'))
                if top5_acc:
                    formatted += f"- Top-5 Accuracy: {top5_acc:.2f}%\n"
            
            formatted += f"- MACs reduction: {entry.get('macs_reduction', 'N/A')}\n"
            formatted += "\n"
        
        return formatted
    
    def _get_default_priorities(self, dataset):
        """Get dataset-specific default parameter priorities (legacy function for compatibility)"""
        if dataset.lower() == 'imagenet':
            return ["importance_criterion", "pruning_ratio", "round_to"]  # Importance first for accuracy
        else:
            return ["pruning_ratio", "importance_criterion", "round_to"]  # Ratio first for efficiency

    def _validate_dataset_recommendations(self, directives_dict, dataset):
        """Validate that LLM recommendations are appropriate for the dataset (legacy function for compatibility)"""
        validated = directives_dict.copy()
        
        # Validate pruning ratio ranges
        pruning_ratio = directives_dict.get('pruning_ratio', 0.5)
        
        if dataset.lower() == 'imagenet':
            # ImageNet should be conservative (60-80% max)
            if pruning_ratio > 0.8:
                print(f"[⚠️] Adjusting aggressive ImageNet pruning ratio from {pruning_ratio:.3f} to 0.8")
                validated['pruning_ratio'] = 0.8
                validated['adjustment_reason'] = "Capped for ImageNet complexity"
        else:  # CIFAR-10
            # CIFAR-10 can be more aggressive (up to 95%)
            if pruning_ratio > 0.95:
                print(f"[⚠️] Adjusting extreme CIFAR-10 pruning ratio from {pruning_ratio:.3f} to 0.95")
                validated['pruning_ratio'] = 0.95
                validated['adjustment_reason'] = "Capped at maximum reasonable level"
        
        # Validate importance criterion
        importance = directives_dict.get('importance_criterion', 'taylor')
        if dataset.lower() == 'imagenet' and importance not in ['taylor', 'l2norm']:
            print(f"[⚠️] Recommending Taylor importance for ImageNet instead of {importance}")
            validated['importance_criterion'] = 'taylor'
            validated['importance_adjustment'] = "Taylor recommended for ImageNet accuracy"
        
        validated['dataset_adapted'] = True
        return validated

    def _create_fallback_response(self, response_content, dataset, state):
        """Create a dataset-aware fallback response when JSON parsing fails (legacy function for compatibility)"""
        
        # Dataset-aware fallback strategy
        if dataset.lower() == 'imagenet':
            exploration_strategy = "Use conservative pruning with Taylor importance for ImageNet complexity"
            parameter_priorities = ["importance_criterion", "pruning_ratio", "round_to"]
        else:
            exploration_strategy = "Explore aggressive pruning ratios suitable for CIFAR-10"
            parameter_priorities = ["pruning_ratio", "importance_criterion", "round_to"]
        
        should_continue = True
        stop_reason = None
        
        # Try to extract directives using regex
        directives_match = re.search(r'"directives"[:\s]+"([^"]+)"', response_content)
        if directives_match:
            exploration_strategy = directives_match.group(1).strip()
        
        # Try to extract parameter priorities
        params_match = re.search(r'"parameter_priorities"[:\s]+\[(.*?)\]', response_content, re.DOTALL)
        if params_match:
            params_str = params_match.group(1)
            priority_matches = re.findall(r'["\'](.*?)["\']', params_str)
            if priority_matches:
                parameter_priorities = priority_matches
        
        # Check for continue/stop with dataset context
        if re.search(r'"continue"[:\s]+(false|False)', response_content):
            should_continue = False
            reason_match = re.search(r'"stop_reason"[:\s]+["\'](.*?)["\']', response_content)
            if reason_match:
                stop_reason = reason_match.group(1).strip()
            else:
                stop_reason = f"Parsing failed for {dataset}, but stop detected in response."
        
        print(f"[⚠️] Using {dataset}-specific fallback strategy: {exploration_strategy}")
        
        return {
            'master_results': {
                'directives': exploration_strategy,
                'parameter_priorities': parameter_priorities,
                'exploration_strategy': exploration_strategy,
                'should_continue': should_continue,
                'stop_reason': stop_reason,
                'dataset': dataset,
                'fallback_used': True,
                'dataset_adapted': True,
            }
        }
