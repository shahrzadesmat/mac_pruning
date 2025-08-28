from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn


@dataclass
class MLPCouple:
    """Represents a coupled MLP fc1+fc2 pair that must be pruned together"""
    fc1: nn.Linear
    fc2: nn.Linear
    fc1_name: str
    fc2_name: str
    
    def get_hidden_dim(self):
        return self.fc1.out_features
    
    def validate_coupling(self):
        return self.fc1.out_features == self.fc2.in_features

@dataclass  
class AttentionCouple:
    """Represents a coupled QKV+Proj pair that must be pruned together"""
    qkv: nn.Linear
    proj: nn.Linear
    qkv_name: str
    proj_name: str
    
    def get_embed_dim(self):
        return self.qkv.out_features // 3  # QKV is 3x embed_dim
    
    def validate_coupling(self):
        return self.qkv.out_features // 3 == self.proj.in_features

class PruningState(TypedDict):
    query: str
    model: Any  # PyTorch model
    
    # MAC-based fields (primary)
    baseline_macs: float
    target_macs: float
    macs_overshoot_tolerance_pct: float
    macs_undershoot_tolerance_pct: float
    attempted_macs_g: List[float]
    
    # Legacy ratio field (for backward compatibility)
    attempted_pruning_ratios: List[float]
    
    profile_results: Dict
    master_results: Dict
    analysis_results: Dict
    pruning_results: Dict
    fine_tuning_results: Dict
    evaluation_results: Dict
    revision_number: int
    max_revisions: int
    history: List[Dict[str, Any]]
    dataset: str  # 'cifar10' or 'imagenet'
    num_classes: int  # 10 for CIFAR-10, 1000 for ImageNet
    input_size: int  # 32 for CIFAR-10, 224 for ImageNet
    data_path: str  # Path to dataset
