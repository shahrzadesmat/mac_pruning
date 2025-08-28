from dataclasses import dataclass
import json, re
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
from utils.analysis_dependency import DependencyAnalyzer
from utils.analysis_structures import MLPCouple, AttentionCouple
import torch_pruning as tp
import torch


@dataclass
class IsomorphicGroup:
    """Represents an isomorphic group of layers with similar structure."""
    name: str
    layers: List[nn.Module]
    layer_names: List[str]
    dimensions: List[int]
    mac_count: float  # Current MAC operations for this group
    target_macs: float  # Target MAC operations after pruning
    pruning_ratio: float
    constraints: List[str]

class ViTIsomorphicAnalyzer:
    """Enhanced analyzer with dependency awareness"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.dependency_analyzer = DependencyAnalyzer(model)
            
    def create_isomorphic_groups(self, target_macs: float, isomorphic_group_ratios: Dict[str, float] = None, baseline_macs: float = None) -> Dict[str, IsomorphicGroup]:
        """Create dependency-aware isomorphic groups based on MAC constraints"""
        
        # Default MAC allocation percentages for different layer types
        default_isomorphic_group_ratios = {
            'qkv_multiplier': 0.4,      # 40% of target MAC for QKV layers
            'mlp_multiplier': 0.5,      # 50% of target MAC for MLP layers
            'proj_multiplier': 0.08,    # 8% of target MAC for projection layers
            'head_multiplier': 0.02     # 2% of target MAC for head layers
        }
        
        isomorphic_group_ratios = isomorphic_group_ratios or default_isomorphic_group_ratios
        
        groups = self._create_dependency_aware_groups(target_macs, isomorphic_group_ratios, baseline_macs)
        
        return groups

    def _create_dependency_aware_groups(self, target_macs, isomorphic_group_ratios, baseline_macs):
        """Create groups that respect layer dependencies based on MAC constraints"""
        
        # Find all transformer blocks first
        transformer_blocks = self._find_transformer_blocks()
        
        print(f"[🔍] Found {len(transformer_blocks)} transformer blocks")
        
        target_ratio = 1.0 - (target_macs / baseline_macs)
        
        groups = {
            'mlp_blocks': IsomorphicGroup(
                name='MLP Blocks (Coupled)',
                layers=[],
                layer_names=[],
                dimensions=[],
                mac_count=0,
                target_macs=0,  # Will be calculated
                pruning_ratio=min(0.99, isomorphic_group_ratios.get("mlp_multiplier", 0.5)),
                constraints=['fc1.out_features == fc2.in_features']
            ),
            'attention_blocks': IsomorphicGroup(
                name='Attention Blocks (Coupled)', 
                layers=[],
                layer_names=[],
                dimensions=[],
                mac_count=0,
                target_macs=0,  # Will be calculated
                pruning_ratio=min(0.99, isomorphic_group_ratios.get("qkv_multiplier", 0.4)),
                constraints=['qkv.out_features == proj.in_features']
            ),
            'output_projections': IsomorphicGroup(
                name='Output Projections',
                layers=[],
                layer_names=[],
                dimensions=[],
                mac_count=0,
                target_macs=0,  # ✅ Set to 0 initially
                pruning_ratio=isomorphic_group_ratios.get("proj_multiplier", 0.0) * target_ratio,
                constraints=['Preserve residual connections']
            )
        }
        
        # ✅ ADD: Calculate total MACs and estimate distribution
        example_inputs = (torch.randn(1, 3, 224, 224).to(next(self.model.parameters()).device),)
        total_macs, _ = tp.utils.count_ops_and_params(self.model, example_inputs)
        
        print(f"[📊] Total model MACs: {total_macs/1e9:.3f}G")
        print(f"[📊] Target MACs: {target_macs/1e9:.3f}G")
        print(f"[📊] Target ratio: {target_ratio:.3f}")

        # Estimate MAC distribution
        mlp_mac_fraction = self._estimate_mlp_mac_fraction()
        attention_mac_fraction = self._estimate_attention_mac_fraction()

        groups['mlp_blocks'].mac_count = total_macs * mlp_mac_fraction
        groups['attention_blocks'].mac_count = total_macs * attention_mac_fraction

        # Calculate target MACs based on pruning ratios
        groups['mlp_blocks'].target_macs = groups['mlp_blocks'].mac_count * (1 - groups['mlp_blocks'].pruning_ratio)
        groups['attention_blocks'].target_macs = groups['attention_blocks'].mac_count * (1 - groups['attention_blocks'].pruning_ratio)
        
        print(f"[📊] MLP: {groups['mlp_blocks'].mac_count/1e9:.3f}G → {groups['mlp_blocks'].target_macs/1e9:.3f}G (ratio: {groups['mlp_blocks'].pruning_ratio:.3f})")
        print(f"[📊] Attention: {groups['attention_blocks'].mac_count/1e9:.3f}G → {groups['attention_blocks'].target_macs/1e9:.3f}G (ratio: {groups['attention_blocks'].pruning_ratio:.3f})")
        
        # ✅ ADD: Group layers by their coupled relationships with debug info
        for block_idx, block_layers in transformer_blocks.items():
            print(f"[🔍] Block {block_idx} layers: {list(block_layers.keys())}")
            
            # Add MLP block (fc1 + fc2 together)
            if 'fc1' in block_layers and 'fc2' in block_layers:
                mlp_couple = MLPCouple(
                    fc1=block_layers['fc1']['layer'],
                    fc2=block_layers['fc2']['layer'],
                    fc1_name=block_layers['fc1']['name'],
                    fc2_name=block_layers['fc2']['name']
                )
                groups['mlp_blocks'].layers.append(mlp_couple)
                groups['mlp_blocks'].layer_names.append(f"block_{block_idx}_mlp")
                print(f"[✅] Added MLP couple for block {block_idx}")
            else:
                print(f"[❌] Missing MLP layers in block {block_idx}")
            
            # Add Attention block (qkv + proj together) 
            if 'qkv' in block_layers and 'proj' in block_layers:
                attn_couple = AttentionCouple(
                    qkv=block_layers['qkv']['layer'],
                    proj=block_layers['proj']['layer'], 
                    qkv_name=block_layers['qkv']['name'],
                    proj_name=block_layers['proj']['name']
                )
                groups['attention_blocks'].layers.append(attn_couple)
                groups['attention_blocks'].layer_names.append(f"block_{block_idx}_attn")
                print(f"[✅] Added Attention couple for block {block_idx}")
            else:
                print(f"[❌] Missing Attention layers in block {block_idx}")
        
        # ✅ ADD: Final summary
        print(f"[📊] Final group summary:")
        print(f"  - MLP blocks: {len(groups['mlp_blocks'].layers)} couples")
        print(f"  - Attention blocks: {len(groups['attention_blocks'].layers)} couples")
        print(f"  - Output projections: {len(groups['output_projections'].layers)} layers")
        
        return groups

    def _find_transformer_blocks(self):
        """Find and organize transformer blocks"""
        blocks = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'blocks.' in name:
                # Parse: "blocks.0.mlp.fc1" -> block=0, component=mlp, layer=fc1
                parts = name.split('.')
                block_idx = int(parts[1])
                component = parts[2]  # 'mlp' or 'attn'
                layer_type = parts[3]  # 'fc1', 'fc2', 'qkv', 'proj'
                
                if block_idx not in blocks:
                    blocks[block_idx] = {}
                
                blocks[block_idx][layer_type] = {
                    'layer': module,
                    'name': name,
                    'component': component
                }
        
        return blocks


    def _estimate_mlp_mac_fraction(self):
        """Estimate what fraction of total MACs come from MLP layers"""
        mlp_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.weight.numel()
                total_params += param_count
                if 'mlp' in name or 'fc1' in name or 'fc2' in name:
                    mlp_params += param_count
                    print(f"[🔍] Found MLP layer: {name} ({param_count:,} params)")
        
        fraction = mlp_params / max(total_params, 1) if total_params > 0 else 0.6
        print(f"[📊] MLP MAC fraction estimated: {fraction:.3f} ({mlp_params:,}/{total_params:,} params)")
        return fraction

    def _estimate_attention_mac_fraction(self):
        """Estimate what fraction of total MACs come from attention layers"""
        attention_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.weight.numel()
                total_params += param_count
                if 'attn' in name or 'qkv' in name or 'proj' in name:
                    attention_params += param_count
                    print(f"[🔍] Found Attention layer: {name} ({param_count:,} params)")
        
        fraction = attention_params / max(total_params, 1) if total_params > 0 else 0.3
        print(f"[📊] Attention MAC fraction estimated: {fraction:.3f} ({attention_params:,}/{total_params:,} params)")
        return fraction

