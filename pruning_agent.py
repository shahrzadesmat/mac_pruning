import os
import re
import copy
from datetime import datetime
from typing import Dict
import torch
from torch import nn
import torch_pruning as tp
import timm
import pbench
from utils.timing import time, time_it, time_it_async
from utils.analysis_structures import PruningState
from utils.analysis_isomorphism import ViTIsomorphicAnalyzer
from utils.pruning_safety import PruningSafetyValidator
from utils.pruning_math import (
    calculate_mac_targets,
    extract_pruning_ratio,
    extract_mac_target,
)
from utils.json_utils import parse_llm_json_response, deep_merge
import pbench  # Your custom pruning benchmarks
from data.loaders import (
    get_cifar10_loaders_pbench, 
    get_imagenet_folder_loaders_pbench
)

from utils.analysis_structures import (
    PruningState,
    MLPCouple,
    AttentionCouple,
)
from utils.analysis_isomorphism import IsomorphicGroup
from utils.logging_wandb import log_to_wandb


class PruningAgent:
    def __init__(self):
        pass

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_model(self, model_name: str, device: torch.device, dataset: str = "cifar10", num_classes: int = 10):
        """Enhanced model preparation with support for new architectures"""
        
        # Handle different model name formats
        model_name_normalized = self._normalize_model_name(model_name)
        
        if dataset.lower() == 'imagenet':
            try:
                model = timm.create_model(model_name_normalized, pretrained=True, num_classes=num_classes)
                print(f"[🔧] Created ImageNet model: {model_name_normalized} with pretrained weights")
            except Exception as e:
                print(f"[⚠️] Failed to create {model_name_normalized} with pretrained weights: {e}")
                # Try without pretrained
                model = timm.create_model(model_name_normalized, pretrained=False, num_classes=num_classes)
                print(f"[🔧] Created ImageNet model: {model_name_normalized} without pretrained weights")
        else:
            model = timm.create_model(model_name_normalized, pretrained=False, num_classes=num_classes)
            print(f"[🔧] Created {dataset} model: {model_name_normalized}")
        
        for param in model.parameters():
            param.requires_grad = True
        return model.to(device)
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model names to match timm conventions"""
        name_mapping = {
            'convnext-s': 'convnext_small',
            'convnext_s': 'convnext_small',
            'nvit-s': 'vit_small_patch16_224',  # Map to closest equivalent
            'nvit_s': 'vit_small_patch16_224',
            'upop': 'vit_base_patch16_224',     # Map to base ViT if Upop not available
            'vit-slim': 'vit_small_patch16_224', # Map to small ViT variant
            'vit_slim': 'vit_small_patch16_224',
            'resnet-101': 'resnet101',
            'resnet-50': 'resnet50',
            'resnet_101': 'resnet101',
            'resnet_50': 'resnet50',
        }
        
        model_name_lower = model_name.lower().replace('-', '_')
        return name_mapping.get(model_name_lower, model_name)

    def _calculate_importance(self, model, loader, criterion, device, importance_type="taylor"):
        """Calculate importance based on the specified importance criterion"""
        if importance_type == "taylor":
            # For taylor importance, we need gradients
            model.train()
            model.zero_grad()
            
            print(f"[🧮] Calculating Taylor importance using gradient accumulation...")
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= 100:  # Limit batches for efficiency
                    break
                
                # Handle different data formats (CIFAR-10 vs ImageNet arrow format)
                if isinstance(batch, dict):
                    # ImageNet arrow format
                    inputs = batch['pixel_values'].to(device)
                    targets = batch['label'].to(device)
                else:
                    # Standard CIFAR-10 format
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
            print(f"[🧮] Taylor importance calculation complete")
            return tp.importance.GroupTaylorImportance()
            
        elif importance_type == "l1norm":
            print(f"[🧮] Using L1 norm importance (no gradient computation needed)")
            return tp.importance.GroupNormImportance(p=1)
        elif importance_type == "l2norm":
            print(f"[🧮] Using L2 norm importance (no gradient computation needed)")
            return tp.importance.GroupNormImportance(p=2)
        else:
            print(f"[⚠️] Unknown importance type: {importance_type}, falling back to taylor")
            return tp.importance.GroupTaylorImportance()

    def _setup_dataset_loader(self, dataset: str, data_path: str, batch_size: int = 64):
        """Setup dataset-appropriate data loader with fixed ImageNet folder handling"""
        
        if dataset.lower() == 'imagenet':
            # Get subset fraction from state
            imagenet_subset = getattr(self, '_current_state', {}).get('imagenet_subset', 1.0)
            train_loader, val_loader = get_imagenet_folder_loaders_pbench(
                data_path, batch_size, num_workers=16, subset_fraction=imagenet_subset
            )
            return train_loader, val_loader
        else:
            train_loader, val_loader = get_cifar10_loaders_pbench(batch_size, num_workers=16)
            return train_loader, val_loader



    def _setup_ignored_layers(self, model, num_classes: int, dataset: str):
        """Enhanced ignored layers setup for new architectures"""
        ignored_layers = []
        num_heads = {}
        
        for name, m in model.named_modules():
            # Preserve final classifiers
            if (('fc' in name) or ('classifier' in name) or ('head' in name)) and isinstance(m, nn.Linear):
                if m.out_features == num_classes:
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving final classifier: {name} ({num_classes} classes)")
            
            # ConvNext-specific layer preservation
            if 'convnext' in self.model_name.lower():
                # Preserve layer normalization layers
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving ConvNext normalization: {name}")
                
                # Preserve depthwise convolutions (critical for ConvNext)
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving ConvNext depthwise conv: {name}")
            
            # Standard attention handling
            if isinstance(m, timm.models.swin_transformer.WindowAttention):
                num_heads[m.qkv] = m.num_heads
                ignored_layers.append(m.relative_position_bias_table)
            
            if isinstance(m, timm.models.vision_transformer.Attention):
                num_heads[m.qkv] = m.num_heads
        
        return ignored_layers, num_heads
    


    def _evaluate_model(self, model, loader, device, dataset: str):
        """Dataset-aware model evaluation"""
        model.eval()
        correct = 0
        total = 0

        # For ImageNet, also track Top-5
        if dataset.lower() == 'imagenet':
            correct_top5 = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # 🛠️ Removed the batch cap here

                # Handle different data formats
                if isinstance(batch, dict):
                    # ImageNet arrow format
                    inputs = batch['pixel_values'].to(device)
                    labels = batch['label'].to(device)
                else:
                    # Standard CIFAR-10 format
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                # Top-1 accuracy
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

                # Top-5 accuracy for ImageNet
                if dataset.lower() == 'imagenet':
                    _, top5_preds = outputs.topk(5, 1, True, True)
                    correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

        accuracy = 100. * correct / total

        if dataset.lower() == 'imagenet':
            top5_accuracy = 100. * correct_top5 / total
            print(f"[🎯] Zero-shot accuracy: Top-1: {accuracy:.2f}%, Top-5: {top5_accuracy:.2f}%")
            return accuracy, top5_accuracy
        else:
            print(f"[🎯] Zero-shot accuracy: {accuracy:.2f}%")
            return accuracy, None


    def _is_vision_transformer(self, model: nn.Module, model_name: str) -> bool:
        """Detect if model is a Vision Transformer."""
        vit_indicators = ['vit', 'deit', 'beit', 'swin', 'nvit', 'upop']
        
        if any(indicator in model_name.lower() for indicator in vit_indicators):
            return True
        
        # Check for attention mechanisms
        has_attention = any(
            hasattr(m, 'qkv') or hasattr(m, 'num_heads')
            for m in model.modules()
        )
        return has_attention
    
    def _validate_model_integrity(self, model: nn.Module):
        """Validate model can still do forward pass"""
        try:
            device = next(model.parameters()).device
            test_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(test_input)
            print("[✅] Model integrity validated")
        except Exception as e:
            raise RuntimeError(f"Model broken after pruning: {e}")


    def _prune_attention_heads(self, group: IsomorphicGroup):
        """Prune attention heads by reducing num_heads."""
        for layer in group.layers:
            if hasattr(layer, 'num_heads') and hasattr(layer, 'qkv'):
                original_heads = layer.num_heads
                heads_to_remove = max(1, int(original_heads * group.pruning_ratio))
                new_heads = max(1, original_heads - heads_to_remove)
                
                # Update attention module
                layer.num_heads = new_heads
                if hasattr(layer, 'qkv'):
                    embed_dim = layer.qkv.out_features // 3
                    layer.head_dim = embed_dim // new_heads
                    layer.scale = layer.head_dim ** -0.5
                
                print(f"[🔧] Attention heads: {original_heads} -> {new_heads}")

    def _fix_model_structure(self, model: nn.Module):
        """Fix any structural issues after direct pruning."""
        print(f"[🔧] Fixing model structure after direct pruning...")
        
        # Update any cached dimension attributes
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                try:
                    qkv_out = module.qkv.out_features
                    embed_dim = qkv_out // 3
                    
                    # Ensure num_heads divides embed_dim
                    if embed_dim % module.num_heads != 0:
                        # Find the largest factor of embed_dim that's <= num_heads
                        for new_heads in range(module.num_heads, 0, -1):
                            if embed_dim % new_heads == 0:
                                print(f"[🔧] Adjusting {name}: heads {module.num_heads} -> {new_heads}")
                                module.num_heads = new_heads
                                break
                    
                    # Update head_dim and scale
                    module.head_dim = embed_dim // module.num_heads
                    module.scale = module.head_dim ** -0.5
                    
                except Exception as e:
                    print(f"[⚠️] Could not fix {name}: {e}")

    def _fix_vit_attention_dimensions(self, model: nn.Module):
        """Fix ViT attention dimensions after pruning."""
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                try:
                    qkv_out = module.qkv.out_features
                    embed_dim = qkv_out // 3
                    
                    # Ensure divisibility
                    if embed_dim % module.num_heads != 0:
                        # Find compatible number of heads
                        for heads in range(module.num_heads, 0, -1):
                            if embed_dim % heads == 0:
                                module.num_heads = heads
                                break
                    
                    module.head_dim = embed_dim // module.num_heads
                    module.scale = module.head_dim ** -0.5
                    
                except Exception as e:
                    print(f"[⚠️] Could not fix {name}: {e}")


    def _coordinate_layer_dimensions(self, model: nn.Module):
        """Coordinate dimensions between connected layers after pruning."""
        print(f"[🔧] Coordinating layer dimensions after pruning...")
        
        for name, module in model.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'proj'):
                try:
                    # Get QKV output dimensions
                    qkv_out = module.qkv.out_features
                    embed_dim = qkv_out // 3  # Q, K, V each get embed_dim channels
                    
                    # Current projection layer expects this input
                    current_proj_in = module.proj.in_features
                    
                    if embed_dim != current_proj_in:
                        print(f"[🔧] Fixing {name}: QKV outputs {embed_dim}, but proj expects {current_proj_in}")
                        
                        # Option 1: Adjust projection layer to match QKV output
                        if embed_dim < current_proj_in:
                            # Shrink projection layer input
                            self._resize_linear_input(module.proj, embed_dim)
                            print(f"[✂️] Resized {name}.proj input: {current_proj_in} -> {embed_dim}")
                        
                        # Update attention module attributes
                        if hasattr(module, 'num_heads'):
                            # Ensure divisibility
                            while embed_dim % module.num_heads != 0 and module.num_heads > 1:
                                module.num_heads -= 1
                            
                            module.head_dim = embed_dim // module.num_heads
                            module.scale = module.head_dim ** -0.5
                            
                            print(f"[🔧] Updated {name}: embed_dim={embed_dim}, heads={module.num_heads}, head_dim={module.head_dim}")
                            
                except Exception as e:
                    print(f"[⚠️] Could not coordinate {name}: {e}")

    def _resize_linear_input(self, layer: nn.Linear, new_input_dim: int):
        """Resize a linear layer's input dimension."""
        if layer.in_features <= new_input_dim:
            return  # No need to resize
            
        print(f"[✂️] Resizing linear layer input: {layer.in_features} -> {new_input_dim}")
        
        # Select most important input channels
        with torch.no_grad():
            # Calculate importance of input channels
            importance = torch.norm(layer.weight, dim=0)  # Norm per input channel
            _, top_indices = torch.topk(importance, new_input_dim, largest=True)
            top_indices = torch.sort(top_indices)[0]
            
            # Create new smaller weight matrix
            new_weight = layer.weight.data[:, top_indices]
            
            # Update layer
            layer.weight = nn.Parameter(new_weight)
            layer.in_features = new_input_dim

    def _coordinate_mlp_dimensions(self, model: nn.Module):
        """Fix MLP layer dimension mismatches."""
        print(f"[🔧] Coordinating MLP dimensions...")
        
        for name, module in model.named_modules():
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                try:
                    fc1_out = module.fc1.out_features
                    fc2_in = module.fc2.in_features
                    
                    if fc1_out != fc2_in:
                        print(f"[🔧] MLP mismatch in {name}: fc1 outputs {fc1_out}, fc2 expects {fc2_in}")
                        
                        # Adjust fc2 input to match fc1 output
                        if fc1_out < fc2_in:
                            self._resize_linear_input(module.fc2, fc1_out)
                            print(f"[✂️] Resized {name}.fc2 input: {fc2_in} -> {fc1_out}")
                            
                except Exception as e:
                    print(f"[⚠️] Could not coordinate MLP {name}: {e}")    

    def _detect_model_architecture(self, model_name, model=None):
        """Enhanced model detection for new architectures"""
        model_name_lower = model_name.lower()
        
        # Check for transformer/attention-based models (ViT family)
        vit_indicators = ['vit', 'deit', 'beit', 'swin', 'nvit', 'upop']
        
        # Only check model.modules() if model is not None
        has_attention = False
        if model is not None:
            has_attention = any(isinstance(m, timm.models.vision_transformer.Attention) for m in model.modules())
        
        if any(indicator in model_name_lower for indicator in vit_indicators) or has_attention:
            return 'vit'
        
        # Check for ConvNext models
        if 'convnext' in model_name_lower or 'convneXt' in model_name:
            return 'convnext'
        
        # Check for ResNet variants
        if 'resnet' in model_name_lower:
            return 'resnet'
        
        # Default to CNN for unknown models
        return 'cnn'


    @time_it_async("4. Pruning Agent")
    async def execute_pruning(self, state: Dict) -> Dict:
        """Enhanced pruning with Analysis Agent recommendations"""
        print("\n[🔧] Executing dataset-aware pruning with Analysis Agent recommendations...")

        self._current_state = state

        # Extract dataset information
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        input_size = state.get("input_size", 224)
        data_path = state.get("data_path", "./data")
        
        print(f"[🔧] Pruning for {dataset}: {num_classes} classes, {input_size}x{input_size}")

        # Extract model name
        model_name = state.get("model_name")
        if not model_name:
            query = state.get('query', '')
            match = re.search(r'Prune\s+(\w+(?:_\w+)*)\s+model', query)
            if match:
                model_name = match.group(1)
                print(f"[📋] Extracted model name from query: {model_name}")
            else:
                raise ValueError("Model name is missing and couldn't be parsed from query.")
                
        # ONLY CHANGE: Enhanced CNN detection and routing
        model_name = state.get('model_name', 'unknown')

        # Detect model architecture type  
        arch_type = self._detect_model_architecture(model_name)

        cnn_types = ['resnet', 'efficientnet', 'mobilenet', 'densenet', 'resnext', 'convnext']
        vit_types = ['vit', 'deit', 'beit', 'swin', 'nvit', 'upop']

        is_vit = any(vit_type in model_name.lower() for vit_type in vit_types)
        is_cnn = (not is_vit and 
                (arch_type in ['resnet', 'convnext', 'cnn'] or 
                any(cnn_type in model_name.lower() for cnn_type in cnn_types)))

        if is_cnn:
            print(f"[🔍] Detected CNN/ConvNext architecture: {model_name}")
            print(f"[🔄] Routing to enhanced CNN pruning method...")
            return await self._execute_cnn_pruning(state, model_name)
        else:
            print(f"[🔍] Detected ViT/Transformer architecture: {model_name}")
            print(f"[🔄] Using existing ViT pruning method...")


        analysis_results = state.get("analysis_results", {})
        
        # Store analysis results for access in isomorphic pruning
        self._current_analysis_results = analysis_results

        target_macs = state.get("target_macs")
        target_ratio = None

        if target_macs is None:
            # Fallback to ratio-based approach
            target_ratio = analysis_results.get("pruning_ratio")
            if target_ratio is None:
                target_ratio = analysis_results.get("suggested_pruning_ratio")
                if target_ratio is None:
                    target_ratio = state.get("target_pruning_ratio")

            
            # Get base MACs to calculate target
            device = self.get_device()
            temp_model = self._prepare_model(model_name, device, dataset, num_classes)
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            base_macs, _ = tp.utils.count_ops_and_params(temp_model, example_inputs)
            target_macs = float(base_macs) * (1.0 - float(target_ratio))
            del temp_model  # Clean up
        else:
            target_ratio = None  # Will be derived from MACs later
                
        print(f"[🎯] Target pruning ratio: {target_ratio:.4f}" if target_ratio is not None else "[🎯] Using MACs-first approach, target ratio will be derived")
        
        # Extract importance criterion with FIXED logic
        importance_type = None
        
        # 1. First try direct from analysis results
        importance_type = analysis_results.get("importance_criterion")
        
        # 2. Then try from strategy_dict  
        if importance_type is None:
            strategy_dict = analysis_results.get("strategy_dict", {})
            importance_type = strategy_dict.get("importance_criterion")
        
        # 3. Then try from master results (the LLM's original suggestion!)
        if importance_type is None:
            master_results = state.get("master_results", {})
            recommended_strategy = master_results.get("recommended_strategy", {})
            importance_type = recommended_strategy.get("importance_criterion")

        if importance_type is None:
            if dataset.lower() == 'imagenet':
                importance_type = "taylor"  # Always use Taylor for ImageNet
            else:
                importance_type = "l1norm"   # L1 norm OK for CIFAR-10
                
        print(f"[🧮] FIXED: Using importance criterion: {importance_type}")
        
        # Extract round-to value
        round_to_value = analysis_results.get("round_to_value")
        if round_to_value is None:
            strategy_dict = analysis_results.get("strategy_dict", {})
            round_to_value = strategy_dict.get("round_to", None)
        print(f"[📏] Using round-to value: {round_to_value}")
        
        # Extract isomorphic group ratios
        isomorphic_group_ratios = analysis_results.get("isomorphic_group_ratios")
        if isomorphic_group_ratios is None:
            strategy_dict = analysis_results.get("strategy_dict", {})
            isomorphic_group_ratios = strategy_dict.get("isomorphic_group_ratios", {})

        current_revision = state.get('revision_number', 0)

        
        if (dataset.lower() == 'imagenet' and 
            ('deit' in model_name.lower() or 'vit' in model_name.lower()) and 
            current_revision == 0):
            
            print(f"[🛡️] ENFORCING Option 2: Skip attention pruning for ImageNet ViT")
            isomorphic_group_ratios = {
                'qkv_multiplier': 0.0,      # Force NO attention pruning
                'mlp_multiplier': 1.5,      # Aggressive MLP only
                'proj_multiplier': 0.0,     # Protect projections
                'head_multiplier': 0.0      # Protect heads
            }
            print(f"[🔧] Using SAFE ImageNet ViT ratios: {isomorphic_group_ratios}")

        # ✅ Apply conservative bias ONLY for first attempt on ImageNet ViTs, then let Analysis Agent learn
        if dataset.lower() == "imagenet" and any(vit_type in model_name.lower() for vit_type in ['vit', 'deit', 'beit', 'swin']) and current_revision == 0:
            mlp_bias = max(isomorphic_group_ratios.get("mlp_multiplier", 0), 1.65)
            isomorphic_group_ratios["mlp_multiplier"] = mlp_bias
            print(f"[🎯] First attempt: Biasing {model_name} MLP multiplier to {mlp_bias:.2f} for ImageNet stability")
        elif current_revision > 0:
            print(f"[🧠] Revision {current_revision}: Using Analysis Agent's learned ratios")
            print(f"[🔧] Analysis Agent suggested ratios: {isomorphic_group_ratios}")
        else:
            print(f"[🔧] Using LLM-suggested isomorphic ratios: {isomorphic_group_ratios}")


        # If still no isomorphic ratios, use default values based on dataset
        if not isomorphic_group_ratios:
            if dataset.lower() == 'imagenet':
                # Conservative ratios for ImageNet
                isomorphic_group_ratios = {
                    'qkv_multiplier': 0.0,      # Conservative for attention
                    'mlp_multiplier': 1.5,      # Moderate for MLP
                    'proj_multiplier': 0.0,     # Don't prune projections
                    'head_multiplier': 0.0      # Don't prune heads
                }
            else:  # CIFAR-10
                # More aggressive ratios for CIFAR-10
                isomorphic_group_ratios = {
                    'qkv_multiplier': 0.5,      # Moderate for attention
                    'mlp_multiplier': 1.2,      # Aggressive for MLP
                    'proj_multiplier': 0.0,     # Don't prune projections
                    'head_multiplier': 0.0      # Don't prune heads
                }
            print(f"[🔧] Using default {dataset} isomorphic ratios: {isomorphic_group_ratios}")
        else:
            print(f"[🔧] Using LLM-suggested isomorphic ratios: {isomorphic_group_ratios}")

        # Get rationale if available
        rationale = analysis_results.get("rationale")
        if rationale:
            print(f"[💡] Rationale: {rationale}")

        # Setup pruning parameters
        tolerance = 0.02  # Acceptable deviation from target MACs
        device = self.get_device()
        print(f"[💻] Using device: {device}")

        # Set global pruning to True for isomorphic pruning
        global_flag = True

        # Setup dataset-appropriate data loader
        try:
            # Adjust batch size based on dataset and available memory
            if dataset.lower() == 'imagenet':
                batch_size = 64  # Smaller batch size for ImageNet
            else:
                batch_size = 64  # Standard for CIFAR-10
                
            train_loader, val_loader = self._setup_dataset_loader(dataset, data_path, batch_size)
            criterion = nn.CrossEntropyLoss().to(device)
            print(f"[📊] Setup {dataset} data loader with batch size {batch_size}")

        except Exception as e:
            print(f"[❌] Failed to setup data loader: {e}")
            return {**state, 'error': f'Data loader setup failed: {str(e)}'}

        # Tracking variables
        best_model = None
        best_accuracy = -float('inf')
        best_ratio = None
        best_state_dict = None
        best_top5_accuracy = None
        attempted_ratios = state.get('attempted_pruning_ratios', [])


        if target_ratio is None:
            # We're in MACs-first mode, need to calculate ratio
            device = self.get_device()
            temp_model = self._prepare_model(model_name, device, dataset, num_classes)
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            base_macs, _ = tp.utils.count_ops_and_params(temp_model, example_inputs)
            target_ratio = 1.0 - (target_macs / base_macs)
            del temp_model
            print(f"[🔄] Calculated target ratio from MACs: {target_ratio:.4f}")

        # Now target_ratio is guaranteed to exist
        current_ratio = target_ratio
        if current_ratio in attempted_ratios:
            print(f"[⚠️] Ratio {current_ratio:.4f} was previously attempted, but Analysis Agent suggested it again.")

        # Record this attempt
        attempted_ratios.append(current_ratio)

        # Attempt pruning with the ratio suggested by Analysis Agent
        print(f"\n[🔁] Attempting pruning with ratio = {current_ratio:.4f}")

        try:
            # Prepare model with dataset awareness
            model = self._prepare_model(model_name, device, dataset, num_classes)
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            base_params = sum(p.numel() for p in model.parameters())
            print(f"[📊] Base model parameters: {base_params:,}")

            # Execute dependency-aware pruning
            print(f"[🧠] Using dependency-aware isomorphic pruning")
            
            # Create isomorphic groups with LLM ratios
            analyzer = ViTIsomorphicAnalyzer(model)

            # Calculate base MACs if not already done
            if 'base_macs' not in locals():
                base_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)

            # Always derive ratio from MACs (MACs-first approach)
            target_ratio = 1.0 - (target_macs / base_macs)
            groups = analyzer.create_isomorphic_groups(target_macs, isomorphic_group_ratios, base_macs)
            
            # Execute dependency-aware pruning
            pruning_result = self._execute_dependency_aware_pruning(model, groups)

            if not pruning_result['success']:
                print(f"[❌] Pruning failed: {pruning_result.get('error', 'Unknown error')}")
                return {**state, 'error': f"Pruning failed: {pruning_result.get('error', 'Unknown error')}"}

            # Continue with evaluation
            achieved_ratio = pruning_result['actual_ratio']
            
            # ✅ CRITICAL FIX: Evaluate IMMEDIATELY after pruning for TRUE zero-shot results
            print(f"[🧪] Evaluating pruned model (TRUE zero-shot) on {dataset}...")
            evaluation_result = self._evaluate_model(model, val_loader, device, dataset)
            
            if dataset.lower() == 'imagenet':
                zero_shot_top1, zero_shot_top5 = evaluation_result
                print(f"[📊] TRUE Zero-shot - Top-1: {zero_shot_top1:.2f}%, Top-5: {zero_shot_top5:.2f}%")
                best_top5_accuracy = zero_shot_top5
                zero_shot_accuracy = zero_shot_top1  # For compatibility
            else:
                zero_shot_accuracy = evaluation_result[0]  # Only top-1 for CIFAR-10
                zero_shot_top1 = zero_shot_accuracy
                zero_shot_top5 = None
                print(f"[📊] TRUE Zero-shot accuracy: {zero_shot_accuracy:.2f}%")



            # For ImageNet, add Top-5 metrics
            if dataset.lower() == 'imagenet':
                log_to_wandb({
                    "zero_shot_top1_accuracy": zero_shot_top1,
                    "zero_shot_top5_accuracy": zero_shot_top5,
                }, step_name="pruning", dataset=dataset)

            # Store the pruned model state BEFORE any potential modifications
            pruned_model_state = copy.deepcopy(model.state_dict())

            # Set as best model
            best_accuracy = zero_shot_accuracy
            best_model = model
            best_state_dict = model.state_dict()
            best_ratio = achieved_ratio

        except Exception as e:
            import traceback
            print(f"[❌] Error in pruning attempt: {e}")
            print(traceback.format_exc())
            
            # If pruning failed, report failure
            state["pruning_results"] = {
                "success": False,
                "error": f"Pruning failed with ratio {current_ratio:.4f}: {str(e)}",
                "achieved_ratio": 0.0,
                "zero_shot_accuracy": 0.0,
                "dataset": dataset
            }
            state['attempted_pruning_ratios'] = attempted_ratios
            return state

        # Pruning succeeded - save model and calculate metrics
        if not best_model:
            print(f"[❌] Pruning failed with ratio {current_ratio:.4f}")
            state["pruning_results"] = {
                "success": False,
                "error": f"Pruning failed with ratio {current_ratio:.4f}",
                "achieved_ratio": 0.0,
                "zero_shot_accuracy": 0.0,
                "dataset": dataset
            }
            state['attempted_pruning_ratios'] = attempted_ratios
            return state

        # Save pruned model checkpoint
        checkpoint_dir = state.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filename = f"pruned_{model_name}_{dataset}_rev{state.get('revision_number', 0)}_ratio{best_ratio:.3f}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        torch.save({
            'complete_model': best_model,
            'state_dict': best_state_dict,
            'pruning_ratio': best_ratio,
            'dataset': dataset,
            'num_classes': num_classes,
            'model_name': model_name,
            'zero_shot_accuracy': zero_shot_accuracy,
            'revision_number': state.get('revision_number', 0),
            'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)

        print(f"[💾] Saved pruned checkpoint: {checkpoint_path}")

        # Calculate MACs reduction using already established values
        print(f"[📊] Calculating MACs reduction...")
        final_macs, _ = tp.utils.count_ops_and_params(best_model, example_inputs)
        macs_reduction = float((base_macs - final_macs) / base_macs)


        state['target_macs'] = float(target_macs)

        macs_error_pct = (
            100.0 * (float(final_macs) - float(target_macs)) / max(float(target_macs), 1e-9)
            if target_macs is not None else None
            )


        log_payload = {
            "achieved_ratio": best_ratio,
            "achieved_ratio_pct": best_ratio * 100,
            "macs_reduction": macs_reduction,
            "macs_reduction_pct": macs_reduction * 100,
            "zero_shot_accuracy": zero_shot_accuracy,
            "parameters_before": base_params,
            "parameters_after": sum(p.numel() for p in best_model.parameters()),
            "revision_number": state.get('revision_number', 0),
            "baseline_macs": float(base_macs),
            "achieved_macs": float(final_macs),
        }
        if target_macs is not None:
            log_payload["target_macs"] = float(target_macs)
            log_payload["macs_error_pct"] = float(macs_error_pct)

        log_to_wandb(log_payload, step_name="pruning", dataset=dataset)


        pruning_results = {
            'success': True,

            # --- MACs-first (NEW) ---
            'baseline_macs': float(base_macs),
            'target_macs': float(target_macs) if target_macs is not None else None,
            'achieved_macs': float(final_macs),
            'macs_reduction': float(macs_reduction),
            'macs_error_pct': float(macs_error_pct) if macs_error_pct is not None else None,

            # --- Legacy (kept for back-compat) ---
            'achieved_ratio': float(best_ratio),

            # --- Existing fields unchanged ---
            'checkpoint_path': checkpoint_path,
            'dataset': dataset,
            'num_classes': num_classes,
            'pruned_model_state': pruned_model_state,
        }



        # Add TRUE zero-shot metrics based on dataset
        if dataset.lower() == 'imagenet':
            pruning_results.update({
                'zero_shot_top1_accuracy': float(zero_shot_top1),
                'zero_shot_top5_accuracy': float(zero_shot_top5),
                'zero_shot_accuracy': float(zero_shot_top1)  # For compatibility
            })
        else:
            pruning_results['zero_shot_accuracy'] = float(zero_shot_accuracy)

        # ✅ CRITICAL FIX: Update history with TRUE zero-shot results
        strategy_dict = analysis_results.get("strategy_dict", {})

        # Create strategy_used with all the existing fields
        strategy_used = {
            'importance_criterion': importance_type,
            'round_to': round_to_value,
            'global_pruning': global_flag,
            'pruning_ratio': current_ratio,
            'pruning_ratio_requested': current_ratio,
            'rationale': rationale if rationale else "No rationale provided"
        }

        if 'isomorphic_group_ratios' in strategy_dict:
            strategy_used['isomorphic_group_ratios'] = strategy_dict['isomorphic_group_ratios']
            print(f"[📝] Storing isomorphic ratios in history: {strategy_dict['isomorphic_group_ratios']}")

        # ✅ ADD: Copy channel_pruning_ratio if it exists (for CNN models)
        if 'channel_pruning_ratio' in strategy_dict:
            strategy_used['channel_pruning_ratio'] = strategy_dict['channel_pruning_ratio']

        history_entry = {
            'revision': state.get('revision_number', 0),
            'target_ratio': target_ratio,
            'achieved_ratio': best_ratio,    
            'target_macs': target_macs,
            'achieved_macs': final_macs,
            'baseline_macs': base_macs,
            'macs_error_pct': (float(final_macs) - float(target_macs)) / max(float(target_macs), 1e-9) * 100.0,
            'macs_reduction': macs_reduction,
            'dataset': dataset,
            'strategy_used': strategy_used
        }


        # Add TRUE zero-shot results to history (these should NEVER be overwritten)
        if dataset.lower() == 'imagenet':
            history_entry.update({
                'zero_shot_top1_accuracy': float(zero_shot_top1),
                'zero_shot_top5_accuracy': float(zero_shot_top5),
                'zero_shot_accuracy': float(zero_shot_top1)  # For compatibility
            })
        else:
            history_entry['zero_shot_accuracy'] = float(zero_shot_accuracy)

        # Add to history
        state['history'] = state.get('history', [])
        state['history'].append(history_entry)

        # Print comprehensive results summary
        print(f"\n[📊] {dataset.upper()} Pruning Results Summary:")
        print(f"Parameters reduction: {best_ratio*100:.2f}%")
        print(f"MACs reduction: {macs_reduction*100:.2f}%")
        
        if dataset.lower() == 'imagenet':
            print(f"TRUE Zero-shot Top-1 accuracy: {zero_shot_top1:.2f}%")
            if zero_shot_top5 is not None:
                print(f"TRUE Zero-shot Top-5 accuracy: {zero_shot_top5:.2f}%")
        else:
            print(f"TRUE Zero-shot accuracy: {zero_shot_accuracy:.2f}%")
        
        print(f"Model saved to: {checkpoint_path}")

        # Update attempted ratios in state
        state['attempted_pruning_ratios'] = attempted_ratios

        return {
            **state,
            'pruning_results': pruning_results,
            'prune': {
                'model': best_model,
                'pruning_results': pruning_results
            },
            'revision_number': state.get('revision_number', 0) + 1,
            'model_name': model_name
        }
    
    async def _execute_cnn_pruning(self, state: Dict, model_name: str) -> Dict:
        """CNN pruning with Analysis Agent learning (no mathematical assumptions)"""
        
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        input_size = state.get("input_size", 224)
        data_path = state.get("data_path", "./data")
        target_ratio = state.get("target_pruning_ratio")
        
        print(f"[🔧] Enhanced CNN Pruning: {model_name} on {dataset}")
        if target_ratio is not None:
            print(f"[🎯] Target PARAMETER reduction: {target_ratio*100:.1f}%")
        else:
            print(f"[🎯] MAC-first mode: Target parameter reduction will be derived from MAC constraints")
        
        # Get analysis results from Analysis Agent (with historical learning)
        analysis_results = state.get("analysis_results", {})
        strategy_dict = analysis_results.get("strategy_dict", {})
        
        # Extract ALL parameters from Analysis Agent
        channel_pruning_ratio = strategy_dict.get('channel_pruning_ratio')
        suggested_round_to = strategy_dict.get('round_to', 8)
        importance_criterion = strategy_dict.get('importance_criterion', 'taylor')
        
        # LEARNING-FIRST APPROACH: Use Analysis Agent or conservative fallback
        if channel_pruning_ratio is not None:
            print(f"[✅] Using Analysis Agent's learned channel ratio: {channel_pruning_ratio:.4f}")
            print(f"[✅] This includes historical learning corrections")
        else:
            # Simple conservative fallback - let Analysis Agent learn the real relationship
            if target_ratio is not None:
                channel_pruning_ratio = target_ratio * 0.8  # Conservative starting point
                print(f"[⚠️] No learned ratio - using conservative starting point: {channel_pruning_ratio:.4f}")
            else:
                # MAC-first mode fallback
                channel_pruning_ratio = 0.3  # Conservative default for MAC-first mode
                print(f"[⚠️] MAC-first mode - using conservative fallback: {channel_pruning_ratio:.4f}")
            print(f"[🧠] Analysis Agent will learn actual {model_name} relationship from results")        
        print(f"[🧮] Using Analysis Agent parameters:")
        print(f"   Channel ratio: {channel_pruning_ratio:.4f}")
        print(f"   Round_to: {suggested_round_to}")
        print(f"   Importance: {importance_criterion}")

        try:
            device = self.get_device()
            model = self._prepare_model(model_name, device, dataset, num_classes)
            original_params = sum(p.numel() for p in model.parameters())
            
            print(f"[📊] Base CNN parameters: {original_params:,}")
            
            # Setup data loader for importance calculation
            batch_size = 64 if dataset.lower() == 'imagenet' else 64
            train_loader, val_loader = self._setup_dataset_loader(dataset, data_path, batch_size)
            criterion = nn.CrossEntropyLoss().to(device)
            
            # RESPECT Analysis Agent's importance choice (no override!)
            if importance_criterion == 'taylor':
                imp = self._calculate_importance(model, train_loader, criterion, device, 'taylor')
                print(f"[🧮] Using Taylor importance (Analysis Agent choice)")
            elif importance_criterion == 'l1norm':
                imp = tp.importance.GroupNormImportance(p=1)
                print(f"[🧮] Using L1 importance (Analysis Agent choice)")
            elif importance_criterion == 'l2norm':
                imp = tp.importance.GroupNormImportance(p=2)
                print(f"[🧮] Using L2 importance (Analysis Agent choice)")
            elif importance_criterion == 'random':
                imp = tp.importance.GroupNormImportance(p=1)  # Use L1 as fallback for random
                print(f"[🧮] Using L1 importance (random fallback - GroupRandomImportance not available)")
            elif importance_criterion == 'magnitude':
                imp = tp.importance.GroupNormImportance(p=1)  # Magnitude is similar to L1
                print(f"[🧮] Using L1 importance (magnitude equivalent)")
            else:
                # Fallback - but warn about unknown criterion
                print(f"[⚠️] Unknown importance criterion: {importance_criterion}, falling back to L1")
                imp = tp.importance.GroupNormImportance(p=1)
                print(f"[🧮] Using L1 importance (fallback)")
            
            # Setup ignored layers (preserve classifier and first conv)
            ignored_layers = []
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear) and m.out_features == num_classes:
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving classifier: {name}")
                elif name == 'conv1' and isinstance(m, nn.Conv2d):
                    ignored_layers.append(m)
                    print(f"[🔒] Preserving first conv: {name}")
            
            # Example inputs
            example_inputs = (torch.randn(1, 3, input_size, input_size).to(device),)
            
            print(f"[🔧] Creating MetaPruner with Analysis Agent ratio: {channel_pruning_ratio:.4f}")
            
            pruner = tp.pruner.MetaPruner(
                model,
                example_inputs=example_inputs,
                global_pruning=True,
                importance=imp,
                isomorphic=False,
                pruning_ratio=channel_pruning_ratio,
                ignored_layers=ignored_layers,
                num_heads={},
                prune_head_dims=False,
                prune_num_heads=False,
                customized_pruners=pbench.extension.EXTENDED_PRUNERS,
                round_to=suggested_round_to,
            )
            
            # Execute pruning
            print(f"[✂️] Executing CNN structured pruning...")
            for i, g in enumerate(pruner.step(interactive=True)):
                print(f"[✂️] Pruning group {i+1}")
                g.prune()
            
            # Calculate results
            final_params = sum(p.numel() for p in model.parameters())
            achieved_ratio = 1 - (final_params / original_params)
            
            print(f"[📊] CNN Pruning Results:")
            print(f"  - Original: {original_params:,} params")
            print(f"  - Final: {final_params:,} params")
            
            print(f"  - Achieved reduction: {achieved_ratio*100:.2f}%")

            if target_ratio is not None:
                print(f"  - Target reduction: {target_ratio*100:.2f}%")
                print(f"  - Difference: {abs(achieved_ratio - target_ratio)*100:.2f}%")
            
                # Success check
                tolerance = 0.02
                ratio_diff = achieved_ratio - target_ratio
                success = (0 <= ratio_diff <= tolerance) or (-0.01 <= ratio_diff < 0)
                print(f"  - Meets target (≤{tolerance*100:.0f}% overshoot): {success}")
            else:
                # MAC-first mode - success determined by MAC constraints elsewhere
                print(f"  - MAC-first mode: Target reduction derived from MAC constraints")
                print(f"  - Achieved reduction: {achieved_ratio*100:.2f}%")
                success = True  # Will be determined by MAC tolerance in routing logic
            
            # Evaluate and log accuracy (but don't make decisions based on it)
            evaluation_result = self._evaluate_model(model, val_loader, device, dataset)

            if dataset.lower() == 'imagenet':
                zero_shot_top1, zero_shot_top5 = evaluation_result
                zero_shot_accuracy = zero_shot_top1
                print(f"[📊] Zero-shot - Top-1: {zero_shot_top1:.2f}%, Top-5: {zero_shot_top5:.2f}%")
            else:
                zero_shot_accuracy = evaluation_result[0]
                zero_shot_top5 = None
                print(f"[📊] Zero-shot accuracy: {zero_shot_accuracy:.2f}%")
            
            # Calculate MACs
            base_model = self._prepare_model(model_name, device, dataset, num_classes)
            base_macs, _ = tp.utils.count_ops_and_params(base_model, example_inputs)
            final_macs, _ = tp.utils.count_ops_and_params(model, example_inputs)
            macs_reduction = (base_macs - final_macs) / base_macs
            
            # Save checkpoint
            checkpoint_path = f"pruned_{model_name}_{dataset}_cnn_learning.pth"
            torch.save({
                'complete_model': model,
                'state_dict': model.state_dict(),
                'pruning_ratio': achieved_ratio,
                'dataset': dataset,
                'model_name': model_name,
                'approach': 'cnn_learning_based',
                'channel_ratio_used': channel_pruning_ratio,
                'target_param_reduction': target_ratio
            }, checkpoint_path)
            
            # MACs-first + keep ratio fields for back-compat
            pruning_results = {
                'success': True,

                'baseline_macs': float(base_macs),  # absolute MACs of the unpruned model
                'target_macs': float(state.get('target_macs')) if state.get('target_macs') is not None else None,
                'achieved_macs': float(final_macs), # absolute MACs after pruning
                'macs_reduction': float(macs_reduction),  # fraction reduced (e.g., 0.35 = 35%)
                'macs_error_pct': (
                    (float(final_macs) - float(state.get('target_macs', 0))) / max(float(state.get('target_macs', 1)), 1e-9) * 100.0
                    if state.get('target_macs') is not None else None
                ),

                # --- Legacy ratio fields (kept so nothing else breaks) ---
                'achieved_ratio': float(achieved_ratio) if 'achieved_ratio' in locals() else None,
                'channel_ratio_used': channel_pruning_ratio,

                # --- Unchanged fields ---
                'checkpoint_path': checkpoint_path,
                'dataset': dataset,
                'num_classes': num_classes,
                'approach': 'cnn_learning_based',
                'learning_based': True
            }

            
            # Add accuracy results
            if dataset.lower() == 'imagenet':
                pruning_results.update({
                    'zero_shot_top1_accuracy': float(zero_shot_top1),
                    'zero_shot_top5_accuracy': float(zero_shot_top5),
                    'zero_shot_accuracy': float(zero_shot_top1)
                })
            else:
                pruning_results['zero_shot_accuracy'] = float(zero_shot_accuracy)
            
            # Create history entry
            history_entry = {
                'revision': state.get('revision_number', 0),
                'target_ratio': target_ratio,
                'achieved_ratio': achieved_ratio,
                'macs_reduction': macs_reduction,
                'dataset': dataset,
                'strategy_used': {
                    'importance_criterion': importance_criterion,
                    'approach': 'cnn_learning_based',
                    'isomorphic': False,
                    'round_to': suggested_round_to,
                    'channel_ratio_used': channel_pruning_ratio,
                    'learning_based': True
                }
            }
            
            # Add accuracy to history
            if dataset.lower() == 'imagenet':
                history_entry.update({
                    'zero_shot_top1_accuracy': float(zero_shot_top1),
                    'zero_shot_top5_accuracy': float(zero_shot_top5),
                    'zero_shot_accuracy': float(zero_shot_top1)
                })
            else:
                history_entry['zero_shot_accuracy'] = float(zero_shot_accuracy)
            
            # Update state
            state['history'] = state.get('history', [])
            state['history'].append(history_entry)
            
            return {
                **state,
                'pruning_results': pruning_results,
                'prune': {
                    'model': model,
                    'pruning_results': pruning_results
                },
                'revision_number': state.get('revision_number', 0) + 1,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"[❌] CNN pruning failed: {e}")
            import traceback
            print(traceback.format_exc())
            
            return {
                **state,
                'pruning_results': {
                    'success': False,
                    'error': str(e),
                    'achieved_ratio': 0.0,
                    'dataset': dataset
                }
            }

    def _execute_dependency_aware_pruning(self, model: nn.Module, groups: Dict[str, IsomorphicGroup]) -> Dict:
        """Execute pruning while respecting dependencies"""
        
        # Store model reference for other methods to use
        self.model = model
        
        original_params = sum(p.numel() for p in model.parameters())
        
        try:
            for group_name, group in groups.items():
                # ✅ FIX: Check for empty layers first
                if len(group.layers) == 0:
                    print(f"[⚠️] Skipping {group_name}: no layers found")
                    continue
                    
                # ✅ FIX: Use the pre-calculated pruning_ratio directly (don't recalculate from MAC)
                pruning_ratio = group.pruning_ratio
                
                # ✅ FIX: Only skip if ratio is actually zero or negative
                if pruning_ratio <= 0.0:
                    print(f"[⚠️] Skipping {group_name}: pruning ratio is {pruning_ratio:.3f}")
                    continue
                    
                # ✅ FIX: Add detailed debug info
                print(f"[✂️] Pruning {group.name} with ratio {pruning_ratio:.3f} ({len(group.layers)} couples)")
                print(f"    MAC count: {group.mac_count:.3f}G, Target MAC: {group.target_macs:.3f}G")
                
                if group_name == 'mlp_blocks':
                    self._prune_mlp_couples(group)
                elif group_name == 'attention_blocks':
                    self._prune_attention_couples(group)
                # Skip output_projections as intended
                
            # Validate model integrity after pruning
            self._validate_model_integrity(model)
            
            final_params = sum(p.numel() for p in model.parameters())
            actual_ratio = 1 - (final_params / original_params)
            
            # ✅ FIX: Add detailed parameter reduction info
            print(f"[📊] Parameter reduction: {original_params:,} → {final_params:,} ({actual_ratio:.3f})")
            
            return {
                'success': True,
                'approach': 'dependency_aware_isomorphic',
                'original_params': original_params,
                'final_params': final_params,
                'actual_ratio': actual_ratio
            }
            
        except Exception as e:
            print(f"[❌] Dependency-aware pruning failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {'success': False, 'error': str(e)}
        finally:
            # Clean up the model reference
            if hasattr(self, 'model'):
                delattr(self, 'model')

    def _prune_mlp_couples(self, group: IsomorphicGroup):
        """Prune MLP couples while maintaining fc1.out == fc2.in"""
        
        for mlp_couple in group.layers:
            if not isinstance(mlp_couple, MLPCouple):
                continue
                
            # Validate coupling before pruning
            if not mlp_couple.validate_coupling():
                print(f"[⚠️] MLP couple invalid: {mlp_couple.fc1_name}")
                continue
            
            original_hidden = mlp_couple.get_hidden_dim()
            target_hidden = max(1, int(original_hidden * (1 - group.pruning_ratio)))
            
            print(f"[🔧] MLP {original_hidden} -> {target_hidden}")
            
            # Compute joint importance for both layers
            importance = self._compute_mlp_couple_importance(mlp_couple)
            
            # Select channels to keep
            _, keep_indices = torch.topk(importance, target_hidden, largest=True)
            keep_indices = torch.sort(keep_indices)[0]
            
            # Update both layers atomically
            self._update_mlp_couple(mlp_couple, keep_indices)
            
            # Validate after update
            if not mlp_couple.validate_coupling():
                raise RuntimeError(f"MLP couple broken after pruning: {mlp_couple.fc1_name}")

    def _compute_mlp_couple_importance(self, mlp_couple: MLPCouple):
        """Compute joint importance for fc1+fc2 pair"""
        with torch.no_grad():
            # fc1 importance: norm of each output channel
            fc1_importance = torch.norm(mlp_couple.fc1.weight, dim=1)
            
            # fc2 importance: norm of each input channel  
            fc2_importance = torch.norm(mlp_couple.fc2.weight, dim=0)
            
            # Joint importance: channels that are important for BOTH layers
            joint_importance = fc1_importance * fc2_importance
            
            return joint_importance

    def _update_mlp_couple(self, mlp_couple: MLPCouple, keep_indices: torch.Tensor):
        """Update both fc1 and fc2 layers with the same channel indices"""
        with torch.no_grad():
            # Update fc1 (prune output channels - rows)
            mlp_couple.fc1.weight = nn.Parameter(
                mlp_couple.fc1.weight[keep_indices, :]
            )
            if mlp_couple.fc1.bias is not None:
                mlp_couple.fc1.bias = nn.Parameter(
                    mlp_couple.fc1.bias[keep_indices]
                )
            mlp_couple.fc1.out_features = len(keep_indices)
            
            # Update fc2 (prune input channels - columns)
            mlp_couple.fc2.weight = nn.Parameter(
                mlp_couple.fc2.weight[:, keep_indices]
            )
            mlp_couple.fc2.in_features = len(keep_indices)
            # fc2.bias unchanged (output channels not pruned)

    def _validate_model_integrity(self, model: nn.Module):
        """Validate model can still do forward pass"""
        try:
            device = next(model.parameters()).device
            test_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(test_input)
            print("[✅] Model integrity validated")
        except Exception as e:
            raise RuntimeError(f"Model broken after pruning: {e}")

    def _prune_attention_couples(self, group: IsomorphicGroup):
        """Prune attention couples while maintaining qkv.out//3 == proj.in"""
        
        print(f"[🔧] Pruning {len(group.layers)} attention couples")
        
        for attn_couple in group.layers:
            if not isinstance(attn_couple, AttentionCouple):
                continue
                
            # Validate coupling before pruning
            if not attn_couple.validate_coupling():
                print(f"[⚠️] Attention couple invalid: {attn_couple.qkv_name}")
                continue
            
            original_embed = attn_couple.get_embed_dim()
            target_embed = max(1, int(original_embed * (1 - group.pruning_ratio)))
            
            # Ensure target_embed is divisible by num_heads (if available)
            target_embed = self._adjust_embed_dim_for_heads(attn_couple, target_embed)
            
            print(f"[🔧] Attention {original_embed} -> {target_embed}")
            
            # Compute joint importance for QKV and projection
            importance = self._compute_attention_couple_importance(attn_couple)
            
            # Select embedding dimensions to keep
            _, keep_indices = torch.topk(importance, target_embed, largest=True)
            keep_indices = torch.sort(keep_indices)[0]
            
            # Update both QKV and projection layers atomically
            self._update_attention_couple(attn_couple, keep_indices)
            
            # Validate after update
            if not attn_couple.validate_coupling():
                raise RuntimeError(f"Attention couple broken after pruning: {attn_couple.qkv_name}")

    def _adjust_embed_dim_for_heads(self, attn_couple: AttentionCouple, target_embed: int):
        """Adjust target embedding dimension to be compatible with attention heads"""
        
        # Try to find the attention module that contains this QKV layer
        for name, module in self.model.named_modules():
            if hasattr(module, 'qkv') and module.qkv is attn_couple.qkv:
                if hasattr(module, 'num_heads'):
                    num_heads = module.num_heads
                    # Ensure target_embed is divisible by num_heads
                    adjusted_embed = (target_embed // num_heads) * num_heads
                    if adjusted_embed < num_heads:
                        adjusted_embed = num_heads  # At least one dimension per head
                    
                    if adjusted_embed != target_embed:
                        print(f"[🔧] Adjusted embed_dim {target_embed} -> {adjusted_embed} for {num_heads} heads")
                    
                    return adjusted_embed
        
        # If no attention module found, return original target
        return target_embed

    def _compute_attention_couple_importance(self, attn_couple: AttentionCouple):
        """Compute joint importance for QKV+Proj pair"""
        with torch.no_grad():
            # QKV layer has shape [embed_dim, 3*embed_dim]
            # We need to compute importance for each embedding dimension
            
            # QKV importance: compute per embedding dimension
            # QKV weight shape: [3*embed_dim, embed_dim]
            # Each embedding dimension corresponds to 3 consecutive output channels (Q, K, V)
            embed_dim = attn_couple.get_embed_dim()
            
            qkv_importance = torch.zeros(embed_dim, device=attn_couple.qkv.weight.device)
            
            for i in range(embed_dim):
                # For each embedding dimension, look at Q, K, V channels
                q_channel = i
                k_channel = i + embed_dim
                v_channel = i + 2 * embed_dim
                
                # Importance is the sum of norms for Q, K, V channels
                q_norm = torch.norm(attn_couple.qkv.weight[q_channel, :])
                k_norm = torch.norm(attn_couple.qkv.weight[k_channel, :])
                v_norm = torch.norm(attn_couple.qkv.weight[v_channel, :])
                
                qkv_importance[i] = q_norm + k_norm + v_norm
            
            # Projection importance: norm of each input channel
            # Proj weight shape: [output_dim, embed_dim]
            proj_importance = torch.norm(attn_couple.proj.weight, dim=0)
            
            # Joint importance: channels that are important for BOTH QKV and projection
            joint_importance = qkv_importance * proj_importance
            
            return joint_importance

    def _update_attention_couple(self, attn_couple: AttentionCouple, keep_indices: torch.Tensor):
        """Update both QKV and projection layers with the same embedding indices"""
        with torch.no_grad():
            embed_dim = attn_couple.get_embed_dim()
            
            # Create indices for QKV layer (Q, K, V for each kept embedding dimension)
            qkv_keep_indices = []
            for idx in keep_indices:
                qkv_keep_indices.extend([
                    idx,                    # Q channel
                    idx + embed_dim,        # K channel  
                    idx + 2 * embed_dim     # V channel
                ])
            qkv_keep_indices = torch.tensor(qkv_keep_indices, device=keep_indices.device)
            
            # Update QKV layer (prune output channels - rows)
            attn_couple.qkv.weight = nn.Parameter(
                attn_couple.qkv.weight[qkv_keep_indices, :]
            )
            if attn_couple.qkv.bias is not None:
                attn_couple.qkv.bias = nn.Parameter(
                    attn_couple.qkv.bias[qkv_keep_indices]
                )
            attn_couple.qkv.out_features = len(qkv_keep_indices)
            
            # Update projection layer (prune input channels - columns)
            attn_couple.proj.weight = nn.Parameter(
                attn_couple.proj.weight[:, keep_indices]
            )
            attn_couple.proj.in_features = len(keep_indices)
            # proj.bias unchanged (output channels not pruned)
            
            # Update attention module attributes if found
            self._update_attention_module_attributes(attn_couple, len(keep_indices))

    def _update_attention_module_attributes(self, attn_couple: AttentionCouple, new_embed_dim: int):
        """Update attention module attributes after pruning"""
        
        # Find the attention module that contains this QKV layer
        for name, module in self.model.named_modules():
            if hasattr(module, 'qkv') and module.qkv is attn_couple.qkv:
                if hasattr(module, 'num_heads'):
                    # Update head_dim and scale
                    module.head_dim = new_embed_dim // module.num_heads
                    if hasattr(module, 'scale'):
                        module.scale = module.head_dim ** -0.5
                    
                    print(f"[🔧] Updated {name}: embed_dim={new_embed_dim}, "
                        f"heads={module.num_heads}, head_dim={module.head_dim}")
                    break

