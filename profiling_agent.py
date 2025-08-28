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


        print(f"[🔍] MAC-aware profiling of {model_name} for {dataset} ({num_classes} classes, {input_size}x{input_size})")
        baseline_str = f"{baseline_macs/1e9:.3f}G" if baseline_macs is not None else "N/A"
        target_str = f"{target_macs/1e9:.3f}G" if target_macs is not None else "N/A"
        print(f"[🔍] MAC Context: {baseline_str} → {target_str} (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")

        
                
        # Prepare subsequent info if this is a re-profiling
        subsequent_info = ""
        if is_subsequent:
            model_type = state.get('model_type', 'pruned')
            
            # Get MAC results from previous attempts
            achieved_macs = state.get('pruning_results', {}).get('achieved_macs', 
                             state.get('evaluation_results', {}).get('achieved_macs', 0))
            mac_efficiency = (achieved_macs / baseline_macs * 100) if achieved_macs and baseline_macs else 0
            
            # Legacy fallback
            achieved_ratio = state.get('pruning_results', {}).get('achieved_ratio', 0)
            
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
            MAC Results: Achieved {achieved_macs/1e9:.3f}G from {baseline_macs/1e9:.3f}G baseline (efficiency: {mac_efficiency:.1f}%)
            Target MAC: {target_macs/1e9:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
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
            
            # Extract MAC-aware layer information
            layer_info = []
            mac_distribution = {}
            dependencies = []
            constraints = []
            sensitivity = []
            critical_layers = []  # Layers that absolutely should not be pruned
            mac_critical_layers = []  # Layers critical for MAC efficiency
            
            # Estimate MAC distribution across layers (simplified)
            total_params = sum(p.numel() for p in model.parameters())
            estimated_total_macs = baseline_macs  # Convert to operations
            
            # Extract layer information with MAC awareness
            layer_macs = 0
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    params = sum(p.numel() for p in module.parameters())
                    
                    # Estimate MAC operations for this layer (simplified)
                    if isinstance(module, nn.Conv2d):
                        # Rough MAC estimation for conv layers
                        output_size = input_size // (2 ** name.count('downsample'))  # Simplified
                        estimated_macs = params * (output_size ** 2)
                    elif isinstance(module, nn.Linear):
                        estimated_macs = params
                    else:
                        estimated_macs = params * 0.1  # BatchNorm has fewer operations
                    
                    layer_macs += estimated_macs

                    if estimated_total_macs is not None and estimated_total_macs > 0:
                        mac_percentage = (estimated_macs / estimated_total_macs) * 100
                    else:
                        mac_percentage = 0
                    
                    layer_info.append({
                        "name": name,
                        "type": module.__class__.__name__,
                        "params": params,
                        "estimated_macs": estimated_macs,
                        "mac_percentage": mac_percentage
                    })
                    
                    # MAC-aware critical layer identification
                    if any(x in name for x in ['fc', 'head', 'classifier']) and isinstance(module, nn.Linear):
                        # Check if this is the final classifier with correct number of classes
                        if module.out_features == num_classes:
                            critical_layers.append(f"{name} (final {num_classes}-class classifier)")
                            if mac_percentage > 5:  # High MAC contribution
                                mac_critical_layers.append(f"{name} (high MAC classifier: {mac_percentage:.1f}%)")
                    elif 'conv1' == name or 'patch_embed' in name:
                        critical_layers.append(f"{name} (initial feature extraction)")
                        if mac_percentage > 2:
                            mac_critical_layers.append(f"{name} (MAC-critical input processing: {mac_percentage:.1f}%)")
                    elif 'downsample' in name:
                        critical_layers.append(f"{name} (architectural downsample)")
                    elif 'pos_embed' in name or 'positional_embedding' in name:
                        critical_layers.append(f"{name} (positional encoding)")
                    
                    # Identify high MAC contribution layers
                    if mac_percentage > 10:  # Layers contributing >10% of MAC operations
                        mac_critical_layers.append(f"{name} (high MAC contribution: {mac_percentage:.1f}%)")
            
            # Build MAC distribution summary
            conv_macs = sum(layer['estimated_macs'] for layer in layer_info if layer['type'] == 'Conv2d')
            linear_macs = sum(layer['estimated_macs'] for layer in layer_info if layer['type'] == 'Linear')
            

            if estimated_total_macs is not None and estimated_total_macs > 0:
                conv_mac_pct = (conv_macs / estimated_total_macs) * 100
                linear_mac_pct = (linear_macs / estimated_total_macs) * 100
                mac_percentage = (estimated_macs / estimated_total_macs) * 100
            else:
                conv_mac_pct = 0
                linear_mac_pct = 0
                mac_reduction_needed_pct = 0

            mac_distribution = {
                'conv_layers_mac_pct': conv_mac_pct,
                'linear_layers_mac_pct': linear_mac_pct,
                'estimated_baseline_macs': estimated_total_macs / 1e9 if estimated_total_macs else 0,
                'mac_reduction_needed_pct': mac_reduction_needed_pct
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
                    f"Target {target_macs:.3f}G requires strategic conv layer MAC reduction"
                ])
                
                # Dataset-specific ResNet considerations
                if dataset.lower() == 'imagenet':
                    sensitivity.extend([
                        f"Pretrained ImageNet features should be preserved while achieving {target_macs:.3f}G MAC target",
                        "Early conv layers critical for low-level feature extraction at scale and MAC efficiency"
                    ])
                    constraints.append("Pretrained weight structure should be maintained for MAC efficiency")
                else:
                    sensitivity.append(f"Can be more aggressive with MAC reduction for {target_macs:.3f}G target due to simpler dataset")
                
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
                    f"MLP and QKV blocks are primary targets for {target_macs:.3f}G MAC reduction"
                ])
                
                # Dataset-specific ViT considerations  
                if dataset.lower() == 'imagenet':
                    sensitivity.extend([
                        f"Pretrained attention patterns valuable for ImageNet at {target_macs:.3f}G MAC target",
                        "Patch embedding layer critical for image tokenization and MAC efficiency"
                    ])
                    constraints.append("Position embeddings should be preserved for MAC-efficient processing")
                else:
                    sensitivity.append(f"Attention layers can handle more aggressive MAC reduction for {target_macs:.3f}G target on simpler datasets")
                
                if "swin" in model_name.lower():
                    constraints.append("Window partition mechanisms must be preserved for MAC efficiency")
                    
            else:
                dependencies.append("Standard feed-forward MAC dependencies between layers")
                sensitivity.extend([
                    "Deeper layers typically less MAC-sensitive to pruning",
                    f"Target {target_macs:.3f}G requires systematic MAC reduction strategy"
                ])
                
                # Dataset-specific general considerations
                if dataset.lower() == 'imagenet':
                    sensitivity.append(f"Complex feature hierarchies require conservative MAC reduction to {target_macs:.3f}G")
                else:
                    sensitivity.append(f"Simple feature requirements allow aggressive MAC reduction to {target_macs:.3f}G")
            

            # MAC-aware dataset-specific constraints and sensitivities
            if baseline_macs is not None and baseline_macs > 0:
                mac_efficiency_target = (target_macs / baseline_macs) * 100
            else:
                mac_efficiency_target = 50.0
            if dataset.lower() == 'imagenet':
                constraints.extend([
                    f"1000-class classifier requires substantial MAC capacity at {target_macs:.3f}G target",
                    f"Complex feature extraction needs sufficient MAC budget ({mac_efficiency_target:.1f}% efficiency)",
                    "Pretrained weights contain valuable learned representations for MAC-efficient processing"
                ])
                sensitivity.extend([
                    f"Early layers extract critical low-level features for complex images at {mac_efficiency_target:.1f}% MAC efficiency",
                    f"Final classifier MAC-sensitive due to 1000-way classification at {target_macs:.3f}G budget",
                    "Middle layers can be MAC-pruned more aggressively than early/late layers"
                ])
            else:  # CIFAR-10 or similar
                constraints.extend([
                    f"{num_classes}-class classifier can handle significant MAC reduction",
                    f"Simpler images require less complex feature extraction at {target_macs:.3f}G target"
                ])
                sensitivity.extend([
                    f"Final classifier less MAC-sensitive due to fewer classes at {target_macs:.3f}G target",
                    f"Can use more aggressive MAC reduction ratios to achieve {mac_efficiency_target:.1f}% efficiency",
                    f"Less risk of feature degradation with aggressive MAC pruning to {target_macs:.3f}G"
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
                "baseline_macs": baseline_macs,
                "target_macs": target_macs,
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
            
            # Calculate MAC reduction metrics for prompt
            if baseline_macs is not None and target_macs is not None and baseline_macs > 0:
                mac_reduction_needed = ((baseline_macs - target_macs) / baseline_macs) * 100
            else:
                mac_reduction_needed = 50.0  # Default 50% reduction
            
            # Call LLM for deeper insights with MAC context
            prompt_text = PROFILING_PROMPT.format(
                model_arch=str(model_name),
                dataset=dataset,
                num_classes=num_classes,
                input_size=input_size,
                baseline_macs=baseline_macs/1e9,  # Convert to G for LLM
                target_macs=target_macs/1e9,      # Convert to G for LLM
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
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
            print(f"[✅] MAC Target: {target_macs/1e9:.3f}G ({mac_efficiency_target:.1f}% efficiency)")
            
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
                
                if baseline_macs is not None and baseline_macs > 0:
                    mac_reduction_needed = ((baseline_macs - target_macs) / baseline_macs) * 100
                else:
                    mac_reduction_needed = 50.0

                
                prompt_text = PROFILING_PROMPT.format(
                    model_arch=str(model_name),
                    dataset=dataset,
                    num_classes=num_classes,
                    input_size=input_size,
                    baseline_macs=baseline_macs/1e9,  # Convert to G for LLM
                    target_macs=target_macs/1e9,      # Convert to G for LLM
                    macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                    macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
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

