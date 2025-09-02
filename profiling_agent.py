from __future__ import annotations
from typing import Dict
import traceback
from langchain_core.messages import SystemMessage, HumanMessage
from llm.provider import get_llm
from llm.prompts import PROFILING_PROMPT
from utils.timing import time_it, time_it_async
from utils.analysis_structures import PruningState
from utils.misc import _to_g
import torch
from torch import nn
from data.dataset_content import get_dataset_specific_content
import timm
import pbench
pbench.forward_patch.patch_timm_forward()
from ptflops import get_model_complexity_info
import re


class ProfilingAgent:
    def __init__(self, llm=None):
        # Use provided LLM or default to ChatOpenAI
        self.llm = llm or get_llm()

    @time_it_async("1. MAC-aware Profiling Agent")
    async def profile_model(self, state: PruningState) -> Dict:
        """MAC-aware model structure analysis and pruning sensitivity identification."""
        print(f"[DEBUG] State keys in profile_model: {state.keys()}")
        print(f"[DEBUG] State type: {type(state)}")

        baseline_macs = state.get("baseline_macs")
        target_macs = state.get("target_macs")

        # Extract MAC-aware and dataset information
        model_name = state.get("model_name", "Unknown Model")
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        input_size = state.get("input_size", 224)
        is_subsequent = state.get('revision_number', 0) > 0
        
        macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)

        overshoot_upper_bound = (target_macs * (1 + macs_overshoot_tolerance_pct/100)) / 1e9
        undershoot_lower_bound = (target_macs * (1 - macs_undershoot_tolerance_pct/100)) / 1e9

        print(f"[🔍] MAC-aware profiling of {model_name} for {dataset} ({num_classes} classes, {input_size}x{input_size})")
        baseline_str = f"{baseline_macs/1e9:.3f}G" if baseline_macs is not None else "N/A"
        target_str = f"{target_macs/1e9:.3f}G" if target_macs is not None else "N/A"
        print(f"[🔍] MAC Context: {baseline_str} → {target_str} (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")


        def safe_float(value, default):
            """Safely convert a value to float with fallback"""
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        # Apply safe handling to baseline_macs and target_macs at the module level
        original_baseline_macs = baseline_macs
        original_target_macs = target_macs
        baseline_macs_for_calc = safe_float(baseline_macs, float('nan'))
        target_macs_for_calc = safe_float(target_macs, float('nan'))

        # Prepare subsequent info if this is a re-profiling
        subsequent_info = ""
        if is_subsequent:
            model_type = state.get('model_type', 'pruned')
            
            # Get MAC results from previous attempts
            achieved_macs = state.get('pruning_results', {}).get('achieved_macs', 
                            state.get('evaluation_results', {}).get('achieved_macs', 0))
            
            # Apply safe handling to achieved_macs
            achieved_macs = safe_float(achieved_macs, 0.0)
            
            # Calculate efficiency safely (baseline_macs is already safe)
            mac_efficiency = (achieved_macs / baseline_macs_for_calc) * 100 if baseline_macs_for_calc > 0 else 0
            
            # Get accuracy based on dataset
            if dataset.lower() == 'imagenet':
                accuracy = state.get('evaluation_results', {}).get('fine_tuned_top1_accuracy', 
                        state.get('evaluation_results', {}).get('zero_shot_top1_accuracy', 0))
                accuracy_type = "Top-1"
            else:
                accuracy = state.get('evaluation_results', {}).get('accuracy', 0)
                accuracy_type = "Accuracy"

            subsequent_info = f"""
            This is a subsequent MAC-aware profile of a {model_type} model on {dataset}.
            MAC Results: Achieved {achieved_macs/1e9:.3f}G from {baseline_macs_for_calc/1e9:.3f}G baseline (efficiency: {mac_efficiency:.1f}%)
            Target MAC: {target_macs_for_calc/1e9:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
            Current {accuracy_type}: {accuracy:.2f}%
            Revision number: {state.get('revision_number', 0)}
            Dataset complexity: {dataset} ({num_classes} classes)
            Focus: Optimize MAC allocation for {target_macs/1e9:.3f}G target
            """

        # Create a sample model to analyze its architecture or use the provided model
        try:
            device = torch.device("cpu")  # Use CPU for profiling
            
            if is_subsequent and 'current_model' in state:
                print("[🔍] MAC profiling the current model from state")
                model = state['current_model']
            else:
                print(f"[🔍] Creating new model instance for MAC profiling: {model_name}")
                
                # Dataset-aware model creation
                if dataset.lower() == 'imagenet':
                    # Use pretrained for ImageNet, with correct number of classes
                    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
                    print(f"[🔍] Created ImageNet model with pretrained weights and {num_classes} classes")
                else:
                    # CIFAR-10 or other datasets - no pretrained weights needed
                    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                    print(f"[🔍] Created {dataset} model with {num_classes} classes")
                
            # Dataset-aware input size
            example_inputs = (torch.randn(1, 3, input_size, input_size),)
            print(f"[🔍] Using input size: {input_size}x{input_size}")

            # Calculate actual baseline MACs if not provided
            if original_baseline_macs is None:
                model.eval()
                with torch.no_grad():
                    # TODO: Replace with fvcore/ptflops when available
                    # For now, we'll calculate it from layer analysis below
                    measured_baseline_macs = None  # Will be set after layer analysis
                    print(f"[🔍] Will measure baseline MACs from layer analysis")
            else:
                measured_baseline_macs = original_baseline_macs
                print(f"[🔍] Using provided baseline MACs: {measured_baseline_macs/1e9:.3f}G")
            
            # Extract MAC-aware layer information
            layer_info = []
            mac_distribution = {}
            dependencies = []
            constraints = []
            sensitivity = []
            critical_layers = []  # Layers that absolutely should not be pruned
            mac_critical_layers = []  # Layers critical for MAC efficiency
            
            # Use ptflops to get accurate model-wide MAC measurement
            flops, params_str = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True)

            # Extract numeric FLOP value
            if 'GMac' in flops:
                flops_numeric = float(flops.replace(' GMac', '')) * 1e9
            elif 'MMac' in flops:
                flops_numeric = float(flops.replace(' MMac', '')) * 1e6
            elif 'KMac' in flops:
                flops_numeric = float(flops.replace(' KMac', '')) * 1e3
            else:
                numbers = re.findall(r'[\d.]+', flops)
                flops_numeric = float(numbers[0]) * 1e9 if numbers else 10e9

            layer_macs = flops_numeric
            print(f"[✅] ptflops measured: {flops}, converted to MACs: {layer_macs/1e9:.3f}G")

            # Build simplified layer_info for compatibility
            layer_info = [{
                "name": "total_model", 
                "type": "ptflops_measurement", 
                "estimated_macs": layer_macs, 
                "mac_percentage": 100.0
            }]

            # Set baseline MAC variables
            estimated_total_macs = layer_macs

            # Set measured baseline if it wasn't provided
            if original_baseline_macs is None:
                measured_baseline_macs = layer_macs
                print(f"[✅] Measured baseline MACs from layers: {measured_baseline_macs/1e9:.3f}G")

            mac_distribution = {
                'estimated_baseline_macs': estimated_total_macs / 1e9 if estimated_total_macs else 0,
                'mac_reduction_needed_pct': ((baseline_macs_for_calc - target_macs_for_calc) / baseline_macs_for_calc) * 100
            }
            
            # Get model summary
            model_summary = str(model)
            
            # Generate MAC-aware architecture-specific insights
            if "resnet" in model_name.lower():
                dependencies.extend([
                    "Residual connections create MAC dependencies between blocks",
                    "Shortcut connections create strong MAC efficiency dependencies between input and output channels"
                ])
                constraints.extend([
                    "Channel dimensions must match at residual connections for MAC efficiency",
                    "Downsample layers must maintain proper MAC/dimension reduction ratios"
                ])
                sensitivity.extend([
                    "Early layers are more MAC-sensitive to pruning",
                    f"Target {target_macs_for_calc/1e9:.3f}G requires strategic conv layer MAC reduction"
                ])
                
                # Dataset-specific ResNet considerations
                if dataset.lower() == 'imagenet':
                    sensitivity.extend([
                        f"Pretrained ImageNet features should be preserved while achieving {target_macs_for_calc/1e9:.3f}G MAC target",
                        "Early conv layers critical for low-level feature extraction at scale and MAC efficiency"
                    ])
                    constraints.append("Pretrained weight structure should be maintained for MAC efficiency")
                else:
                    sensitivity.append(f"Can be more aggressive with MAC reduction for {target_macs_for_calc/1e9:.3f}G target due to simpler dataset")
                
            elif "vit" in model_name.lower() or "swin" in model_name.lower():
                dependencies.extend([
                    "Attention mechanisms create complex MAC dependencies",
                    "Multi-head attention requires consistent head dimensions for MAC efficiency"
                ])
                constraints.extend([
                    "Head dimensions must be maintained for attention MAC calculations",
                    "Embedding dimensions must be consistent for MAC efficiency across layers"
                ])
                sensitivity.extend([
                    "Head pruning generally preferred over full layer pruning for MAC optimization",
                    f"MLP and QKV blocks are primary targets for {target_macs_for_calc/1e9:.3f}G MAC reduction"
                ])
                
                # Dataset-specific ViT considerations  
                if dataset.lower() == 'imagenet':
                    sensitivity.extend([
                        f"Pretrained attention patterns valuable for ImageNet at {target_macs_for_calc/1e9:.3f}G MAC target",
                        "Patch embedding layer critical for image tokenization and MAC efficiency"
                    ])
                    constraints.append("Position embeddings should be preserved for MAC-efficient processing")
                else:
                    sensitivity.append(f"Attention layers can handle more aggressive MAC reduction for {target_macs_for_calc/1e9:.3f}G target on simpler datasets")
                
                if "swin" in model_name.lower():
                    constraints.append("Window partition mechanisms must be preserved for MAC efficiency")
                    
            else:
                dependencies.append("Standard feed-forward MAC dependencies between layers")
                sensitivity.extend([
                    "Deeper layers typically less MAC-sensitive to pruning",
                    f"Target {target_macs_for_calc:.3f}G requires systematic MAC reduction strategy"
                ])
                
                # Dataset-specific general considerations
                if dataset.lower() == 'imagenet':
                    sensitivity.append(f"Complex feature hierarchies require conservative MAC reduction to {target_macs_for_calc/1e9:.3f}G")
                else:
                    sensitivity.append(f"Simple feature requirements allow aggressive MAC reduction to {target_macs_for_calc/1e9:.3f}G")
            

            mac_efficiency_target = (
                (target_macs_for_calc / baseline_macs_for_calc) * 100
                if isinstance(baseline_macs_for_calc, (int, float)) and baseline_macs_for_calc > 0
                and isinstance(target_macs_for_calc, (int, float))
                else None
            )

            if dataset.lower() == 'imagenet':
                constraints.extend([
                    f"1000-class classifier requires substantial MAC capacity at {target_macs_for_calc:.3f}G target",
                    f"Complex feature extraction needs sufficient MAC budget ({mac_efficiency_target:.1f}% efficiency)" if mac_efficiency_target is not None else "Complex feature extraction needs sufficient MAC budget",
                    "Pretrained weights contain valuable learned representations for MAC-efficient processing"
                ])
                sensitivity.extend([
                    f"Early layers extract critical low-level features for complex images at {mac_efficiency_target:.1f}% MAC efficiency",
                    f"Final classifier MAC-sensitive due to 1000-way classification at {target_macs_for_calc:.3f}G budget",
                    "Middle layers can be MAC-pruned more aggressively than early/late layers"
                ])
            else:  # CIFAR-10 or similar
                constraints.extend([
                    f"{num_classes}-class classifier can handle significant MAC reduction",
                    f"Simpler images require less complex feature extraction at {target_macs_for_calc/1e9:.3f}G target"
                ])
                sensitivity.extend([
                    f"Final classifier less MAC-sensitive due to fewer classes at {target_macs_for_calc:.3f}G target",
                    f"Can use more aggressive MAC reduction ratios to achieve {mac_efficiency_target:.1f}% efficiency",
                    f"Less risk of feature degradation with aggressive MAC pruning to {target_macs_for_calc:.3f}G"
                ])
            
            # Compile MAC-aware profile results with dataset information
            profile_results = {
                "model_summary": model_summary,
                "layer_info": layer_info,
                "mac_distribution": mac_distribution,
                "dependencies": dependencies,
                "constraints": constraints,
                "sensitivity": sensitivity,
                "critical_layers": critical_layers,
                "mac_critical_layers": mac_critical_layers,
                "is_subsequent_profile": is_subsequent,
                "dataset": dataset,
                "num_classes": num_classes,
                "input_size": input_size,
                "model_complexity": "high" if dataset.lower() == 'imagenet' else "moderate",
                "baseline_macs": measured_baseline_macs or original_baseline_macs or layer_macs,
                "target_macs": original_target_macs,
                "measured_layer_macs": layer_macs,
                "macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
                "macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
                "mac_efficiency_target": mac_efficiency_target
            }
            
            # For subsequent profiles, add MAC analysis of changes since initial profile
            if is_subsequent:
                profile_results["changes_since_initial"] = f"Model has been MAC-pruned and fine-tuned for {dataset} targeting {target_macs:.3f}G operations"
            

            # Get MAC-aware dataset-specific content for the prompt
            dataset_content = get_dataset_specific_content(
                dataset, num_classes, input_size, baseline_macs, target_macs, 
                macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct,
                state.get('accuracy_threshold', 85.0)
            )
            

            mac_reduction_needed = ((baseline_macs_for_calc - target_macs_for_calc) / baseline_macs_for_calc) * 100
            
            prompt_text = PROFILING_PROMPT.format(
                model_arch=str(model_name),
                dataset=dataset,
                num_classes=num_classes,
                input_size=input_size,
                baseline_macs=baseline_macs_for_calc/1e9,
                target_macs=target_macs_for_calc,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                overshoot_upper_bound=overshoot_upper_bound,
                undershoot_lower_bound=undershoot_lower_bound,
                mac_reduction_needed=mac_reduction_needed,
                dataset_considerations=dataset_content['dataset_guidance'],
                is_subsequent=is_subsequent,
                subsequent_info=subsequent_info
            )
            
            messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=state['query'])
            ]
            response = await self.llm.ainvoke(messages)
            
            # Combine automated and LLM analysis
            profile_results["analysis"] = response.content
            
            print(f"[✅] MAC-aware profiling complete for {dataset} model with {len(layer_info)} analyzable layers")
            eff_str = f"{mac_efficiency_target:.1f}%" if mac_efficiency_target is not None else "N/A"
            print(f"[✅] MAC Target: {target_macs_for_calc/1e9:.3f}G ({eff_str} efficiency)")

            
            return {'profile_results': profile_results}
            
        except Exception as e:
            print(f"Error in MAC-aware profiling: {str(e)}")
            import traceback
            print(f"[⚠️] MAC profiling error traceback: {traceback.format_exc()}")
            
            # Fallback to LLM-based profiling with MAC context
            try:
                dataset_content = get_dataset_specific_content(
                    dataset, num_classes, input_size, baseline_macs, target_macs,
                    macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct,
                    state.get('accuracy_threshold', 85.0)
                )
                
                mac_reduction_needed = ((baseline_macs_for_calc - target_macs_for_calc) / baseline_macs_for_calc) * 100

                
                prompt_text = PROFILING_PROMPT.format(
                    model_arch=str(model_name),
                    dataset=dataset,
                    num_classes=num_classes,
                    input_size=input_size,
                    baseline_macs=baseline_macs_for_calc/1e9,
                    target_macs=target_macs_for_calc,
                    macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                    macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                    overshoot_upper_bound=overshoot_upper_bound,
                    undershoot_lower_bound=undershoot_lower_bound,
                    mac_reduction_needed=mac_reduction_needed,
                    dataset_considerations=dataset_content['dataset_guidance'],
                    is_subsequent=is_subsequent,
                    subsequent_info=subsequent_info
                )
                
                messages = [
                    SystemMessage(content=prompt_text),
                    HumanMessage(content=state['query'])
                ]
                response = await self.llm.ainvoke(messages)
                
                # Return minimal profile with LLM analysis
                fallback_profile = {
                    "analysis": response.content,
                    "dataset": dataset,
                    "num_classes": num_classes,
                    "input_size": input_size,
                    "model_complexity": "high" if dataset.lower() == 'imagenet' else "moderate",
                    "baseline_macs": baseline_macs,
                    "target_macs": target_macs,
                    "macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
                    "macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
                    "error_fallback": True
                }
                
                return {'profile_results': fallback_profile}

                
            except Exception as fallback_error:
                print(f"[❌] Fallback MAC profiling also failed: {fallback_error}")
                # Ultimate fallback - ensure all values are not None
                safe_target_macs = target_macs/1e9 if target_macs is not None else 5.0
                return {
                    'profile_results': {
                        'analysis': f"Basic MAC profile for {model_name} on {dataset}. Target: {safe_target_macs:.3f}G. Error during detailed analysis.",
                        'dataset': dataset,
                        'num_classes': num_classes,
                        'input_size': input_size,
                        'baseline_macs': baseline_macs,
                        'target_macs': target_macs,
                        'critical_failure': True
                    }
                }

