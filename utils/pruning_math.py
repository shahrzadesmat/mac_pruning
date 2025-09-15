from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
import re
from typing import TypedDict, List, Dict, Any


def extract_pruning_ratio(query: str) -> float:
    """Extract pruning ratio from the query string."""
    # Look for patterns like "10 percent" or "10%"
    match = re.search(r'(\d+(?:\.\d+)?)(?:\s*%|\s*percent)', query.lower())
    if match:
        ratio = float(match.group(1)) / 100  # Convert percentage to decimal
        return ratio
    return 0.2  # Default 20% if not found

def extract_mac_target(query):
    """Extract MAC target from user query"""
    import re
    
    # Patterns to match MAC targets in various formats
    mac_patterns = [
        r'(\d+(?:\.\d+)?)\s*G\s*MACs?',              # "5G MACs", "3.2G MAC"
        r'(\d+(?:\.\d+)?)\s*billion\s*MACs?',        # "5 billion MACs"
        r'target.*?(\d+(?:\.\d+)?)\s*G',             # "target 5G"
        r'reduce.*?to\s*(\d+(?:\.\d+)?)\s*G',        # "reduce to 5G"
        r'(\d+(?:\.\d+)?)\s*GMAC',                   # "5GMAC"
    ]
    
    for pattern in mac_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            mac_value = float(match.group(1))
            # print(f"[🔍] Found MAC target in query: {mac_value}G")
            return mac_value
    
    return None


# Helper function for MAC allocation extraction
def extract_isomorphic_group_ratios_from_analysis(analysis_results):
    """Extract MAC allocation strategy from analysis results"""
    architecture_type = analysis_results.get('architecture_type', 'vit')
    
    if architecture_type == 'vit':
        isomorphic_group_ratios = analysis_results.get('isomorphic_group_ratios', {})
        return {
            'mlp_mac_percent': isomorphic_group_ratios.get('mlp_mac_percent', 40.0),
            'qkv_mac_percent': isomorphic_group_ratios.get('qkv_mac_percent', 30.0),
            'proj_mac_percent': isomorphic_group_ratios.get('proj_mac_percent', 8.0),
            'head_mac_percent': isomorphic_group_ratios.get('head_mac_percent', 2.0)
        }
    else:
        return {
            'channel_pruning_ratio': analysis_results.get('channel_pruning_ratio', 0.5)
        }

# Enhanced function for MAC target calculation
def calculate_mac_targets(target_macs, isomorphic_group_ratios, architecture_type):
    """Calculate specific MAC targets for each layer type"""
    if architecture_type == 'vit':
        return {
            'mlp_target_macs': (isomorphic_group_ratios['mlp_mac_percent'] * target_macs) / 100,
            'qkv_target_macs': (isomorphic_group_ratios['qkv_mac_percent'] * target_macs) / 100,
            'proj_target_macs': (isomorphic_group_ratios['proj_mac_percent'] * target_macs) / 100,
            'head_target_macs': (isomorphic_group_ratios['head_mac_percent'] * target_macs) / 100
        }
    else:
        return {
            'channel_target_macs': target_macs,
            'channel_pruning_ratio': isomorphic_group_ratios['channel_pruning_ratio']
        }
