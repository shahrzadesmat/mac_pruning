import torch
from torch import nn
from typing import TypedDict, Any, List, Dict
import traceback


class DependencyAnalyzer:
    """Detects layer dependencies to prevent dimension mismatches"""
    
    def __init__(self, model):
        self.model = model
        self.dependency_graph = {}
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """Build dependency graph between layers"""
        layers = list(self.model.named_modules())
        
        for i, (name1, layer1) in enumerate(layers):
            if not isinstance(layer1, (nn.Linear, nn.Conv2d)):
                continue
                
            self.dependency_graph[name1] = {
                'layer': layer1,
                'depends_on': [],
                'dependents': []
            }
            
            # Find layers this one depends on
            for j, (name2, layer2) in enumerate(layers):
                if i != j and self._are_connected(name1, name2, layer1, layer2):
                    self.dependency_graph[name1]['depends_on'].append(name2)
                    if name2 in self.dependency_graph:
                        self.dependency_graph[name2]['dependents'].append(name1)
    
    def _are_connected(self, name1, name2, layer1, layer2):
        """Check if two layers are directly connected"""
        # For ViT: Check if they're in the same block and consecutive
        if self._in_same_transformer_block(name1, name2):
            # fc1 -> fc2 connection
            if 'fc1' in name2 and 'fc2' in name1:
                return True
            # qkv -> proj connection  
            if 'qkv' in name2 and 'proj' in name1:
                return True
        return False
    
    def _in_same_transformer_block(self, name1, name2):
        """Check if layers are in the same transformer block"""
        # Extract block number: "blocks.0.mlp.fc1" -> "0"
        try:
            block1 = name1.split('.')[1] if 'blocks.' in name1 else None
            block2 = name2.split('.')[1] if 'blocks.' in name2 else None
            return block1 == block2 and block1 is not None
        except:
            return False
    
    def get_coupled_layers(self, layer_name):
        """Get all layers that must be pruned together with this layer"""
        if layer_name not in self.dependency_graph:
            return [layer_name]
        
        coupled = [layer_name]
        
        # Add dependent layers
        for dep in self.dependency_graph[layer_name]['dependents']:
            coupled.append(dep)
        
        # Add layers this depends on
        for dep in self.dependency_graph[layer_name]['depends_on']:
            coupled.append(dep)
            
        return list(set(coupled))

