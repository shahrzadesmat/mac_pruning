import json, re
from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn


def deep_merge(d1, d2):
    """
    Deep merge two dictionaries with MAC-aware parameter preservation. 
    Values from d2 take precedence over d1, except for certain critical 
    parameters that should be preserved from d1.
    
    Args:
        d1: First dictionary (target - values should be preserved for critical parameters)
        d2: Second dictionary (source - values override d1 for non-critical parameters)
        
    Returns:
        Merged dictionary with combined values, preserving MAC-critical parameters
    """
    # List of critical parameters that should be preserved from d1
    # Updated to include MAC-specific parameters
    critical_params = [
        # Legacy critical parameters
        'accuracy_threshold', 'max_revisions', 'target_pruning_ratio', 
        'dataset', 'num_classes', 'input_size', 'data_path', 'model_name',
        
        # MAC-specific critical parameters
        'baseline_macs', 'target_macs', 'macs_overshoot_tolerance_pct', 'macs_undershoot_tolerance_pct',

        
        # MAC state tracking (should be preserved)
        'attempted_macs_g', 'mac_efficiency_history', 'accuracy_history',
        
        # MAC measurement configuration
        'mac_measurement_config'
    ]
    
    merged = d1.copy()
    
    for key, value in d2.items():
        if key in critical_params and key in merged and merged[key] is not None:
            # Skip updating these critical parameters if they already exist in d1
            continue
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # If both values are dictionaries, recursively merge them
            merged[key] = deep_merge(merged[key], value)
        else:
            # Otherwise, value from d2 takes precedence
            merged[key] = value
            
    return merged



def parse_llm_json_response(response_content: str) -> dict:
    """
    Robust JSON parsing for LLM responses with MAC-aware validation and multiple fallback strategies.
    Handles comments, markdown blocks, trailing commas, and validates MAC-specific fields.
    """
    import json
    import re
    
    # print(f"[DEBUG] Attempting to parse MAC-aware JSON response ({len(response_content)} chars)")
    
    strategies = [
        # Strategy 1: Direct parsing (in case it's already clean)
        lambda content: json.loads(content.strip()),
        
        # Strategy 2: Clean common issues and parse
        lambda content: _clean_and_parse_json(content),
        
        # Strategy 3: Extract JSON block first, then clean
        lambda content: _extract_and_clean_json(content),
        
        # Strategy 4: Regex field extraction as last resort
        lambda content: _extract_json_fields_regex(content),
        
        # Strategy 5: MAC-specific field extraction (new)
        lambda content: _extract_mac_fields_regex(content)
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            result = strategy(response_content)
            if isinstance(result, dict) and result:
                # Validate and normalize MAC-specific fields
                result = _validate_and_normalize_mac_fields(result)
                # print(f"[✅] MAC-aware JSON parsed successfully using strategy {i}")
                return result
        except Exception as e:
            # print(f"[⚠️] Strategy {i} failed: {e}")
            continue
    
    # print(f"[❌] All JSON parsing strategies failed!")
    return {}

def _clean_and_parse_json(content: str) -> dict:
    """Clean JSON response and parse with MAC-aware handling"""
    import json
    import re
    
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', content)
    cleaned = re.sub(r'```.*$', '', cleaned, flags=re.DOTALL)
    
    # Remove comments line by line
    lines = cleaned.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove # comments (but preserve # inside strings)
        if '#' in line and '"' not in line.split('#')[0]:
            line = line[:line.find('#')]
        elif '#' in line:
            # More careful # removal when quotes are present
            in_string = False
            quote_char = None
            comment_pos = -1
            
            for i, char in enumerate(line):
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == '#' and not in_string:
                    comment_pos = i
                    break
            
            if comment_pos >= 0:
                line = line[:comment_pos]
        
        # Remove // comments
        if '//' in line and '"' not in line.split('//')[0]:
            line = line[:line.find('//')]
        
        line = line.rstrip()
        if line:
            cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix single quotes to double quotes (risky but common issue)
    cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)  # Keys
    cleaned = re.sub(r':\s*\'([^\']*)\'', r': "\1"', cleaned)  # String values
    
    # Handle MAC-specific placeholder values that LLMs might output
    cleaned = re.sub(r'"YOUR_CALCULATED_[^"]*"', '0.0', cleaned)
    cleaned = re.sub(r'"YOUR_[^"]*_MAC_[^"]*"', '0.0', cleaned)
    cleaned = re.sub(r'"YOUR_[^"]*_PERCENTAGE[^"]*"', '0.0', cleaned)
    
    return json.loads(cleaned.strip())

def _extract_and_clean_json(content: str) -> dict:
    """Extract JSON block first, then clean with MAC awareness"""
    import re
    
    # Find JSON block
    json_match = re.search(r'({[\s\S]*})', content)
    if json_match:
        json_block = json_match.group(1)
        return _clean_and_parse_json(json_block)
    else:
        raise ValueError("No JSON block found")

def _extract_json_fields_regex(content: str) -> dict:
    """Extract fields using regex when JSON parsing completely fails (MAC-aware)"""
    import re
    
    # print(f"[🔧] Using MAC-aware regex field extraction as last resort")
    
    result = {}
    
    # Define patterns for all possible fields (updated for MAC support)
    patterns = {
        # Core MAC parameters
        'baseline_macs': r'"baseline_macs":\s*([\d.]+)',
        'target_macs': r'"target_macs":\s*([\d.]+)',
        'macs_overshoot_tolerance_pct': r'"macs_overshoot_tolerance_pct":\s*([\d.]+)',
        'macs_undershoot_tolerance_pct': r'"macs_undershoot_tolerance_pct":\s*([\d.]+)',
        'expected_achieved_macs': r'"expected_achieved_macs":\s*([\d.]+)',
        'mac_efficiency_target': r'"mac_efficiency_target":\s*([\d.]+)',
        
        # Common parameters
        'importance_criterion': r'"importance_criterion":\s*"([^"]+)"',
        'round_to': r'"round_to":\s*(\d+|null)',
        'global_pruning': r'"global_pruning":\s*(true|false)',
        'rationale': r'"rationale":\s*"([^"]*(?:\\.[^"]*)*)"',
        'dataset_adapted': r'"dataset_adapted":\s*(true|false)',
        'continue': r'"continue":\s*(true|false)',
        'stop_reason': r'"stop_reason":\s*"([^"]*)"',
        'directives': r'"directives":\s*"([^"]*)"',
        'parameter_priorities': r'"parameter_priorities":\s*\[([^\]]*)\]',
        'architecture_type': r'"architecture_type":\s*"([^"]+)"',
        
        # Legacy parameters (for backward compatibility)
        'pruning_ratio': r'"pruning_ratio":\s*([\d.]+)',
        
        # ViT-specific MAC allocation
        'mlp_mac_percent': r'"mlp_mac_percent":\s*([\d.]+)',
        'qkv_mac_percent': r'"qkv_mac_percent":\s*([\d.]+)',
        'proj_mac_percent': r'"proj_mac_percent":\s*([\d.]+)',
        'head_mac_percent': r'"head_mac_percent":\s*([\d.]+)',
        
        # Legacy ViT multipliers
        'qkv_multiplier': r'"qkv_multiplier":\s*([\d.]+)',
        'mlp_multiplier': r'"mlp_multiplier":\s*([\d.]+)',
        'proj_multiplier': r'"proj_multiplier":\s*([\d.]+)',
        'head_multiplier': r'"head_multiplier":\s*([\d.]+)',
        
        # CNN-specific
        'channel_pruning_ratio': r'"channel_pruning_ratio":\s*([\d.]+)',
        
        # MAC calculation fields
        'expected_mlp_mac_g': r'"expected_mlp_mac_g":\s*([\d.]+)',
        'expected_qkv_mac_g': r'"expected_qkv_mac_g":\s*([\d.]+)',
        'expected_proj_mac_g': r'"expected_proj_mac_g":\s*([\d.]+)',
        'expected_head_mac_g': r'"expected_head_mac_g":\s*([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1)
            
            # Convert to appropriate type
            if value in ['true', 'false']:
                result[key] = value == 'true'
            elif value == 'null':
                result[key] = None
            elif key in ['pruning_ratio', 'baseline_macs', 'target_macs', 'macs_overshoot_tolerance_pct', 'macs_undershoot_tolerance_pct',
                        'expected_achieved_macs', 'mac_efficiency_target',
                        'mlp_mac_percent', 'qkv_mac_percent', 'proj_mac_percent', 'head_mac_percent',
                        'qkv_multiplier', 'mlp_multiplier', 'proj_multiplier', 'head_multiplier',
                        'channel_pruning_ratio', 'expected_mlp_mac_g', 'expected_qkv_mac_g',
                        'expected_proj_mac_g', 'expected_head_mac_g']:
                try:
                    result[key] = float(value)
                except ValueError:
                    # print(f"[⚠️] Could not convert {key}={value} to float")
                    continue
            elif key in ['round_to']:
                result[key] = int(value) if value != 'null' else None
            elif key == 'parameter_priorities':
                # Parse array of strings
                array_content = value.replace('"', '').replace("'", "")
                result[key] = [item.strip() for item in array_content.split(',') if item.strip()]
            else:
                result[key] = value
    
    # Try to extract MAC allocation as a nested object
    isomorphic_group_ratios_match = re.search(r'"isomorphic_group_ratios":\s*\{([^}]*)\}', content, re.IGNORECASE | re.DOTALL)
    if isomorphic_group_ratios_match:
        isomorphic_group_ratios_content = isomorphic_group_ratios_match.group(1)
        isomorphic_group_ratios = {}
        
        mac_alloc_patterns = {
            'mlp_mac_percent': r'"mlp_mac_percent":\s*([\d.]+)',
            'qkv_mac_percent': r'"qkv_mac_percent":\s*([\d.]+)',
            'proj_mac_percent': r'"proj_mac_percent":\s*([\d.]+)',
            'head_mac_percent': r'"head_mac_percent":\s*([\d.]+)'
        }
        
        for field, pattern in mac_alloc_patterns.items():
            match = re.search(pattern, isomorphic_group_ratios_content, re.IGNORECASE)
            if match:
                try:
                    isomorphic_group_ratios[field] = float(match.group(1))
                except ValueError:
                    continue
        
        if isomorphic_group_ratios:
            result['isomorphic_group_ratios'] = isomorphic_group_ratios
    
    # Try to extract isomorphic_group_ratios for legacy compatibility
    isomorphic_match = re.search(r'"isomorphic_group_ratios":\s*\{([^}]*)\}', content, re.IGNORECASE | re.DOTALL)
    if isomorphic_match:
        isomorphic_content = isomorphic_match.group(1)
        isomorphic_group_ratios = {}
        
        isomorphic_patterns = {
            'qkv_multiplier': r'"qkv_multiplier":\s*([\d.]+)',
            'mlp_multiplier': r'"mlp_multiplier":\s*([\d.]+)',
            'proj_multiplier': r'"proj_multiplier":\s*([\d.]+)',
            'head_multiplier': r'"head_multiplier":\s*([\d.]+)'
        }
        
        for field, pattern in isomorphic_patterns.items():
            match = re.search(pattern, isomorphic_content, re.IGNORECASE)
            if match:
                try:
                    isomorphic_group_ratios[field] = float(match.group(1))
                except ValueError:
                    continue
        
        if isomorphic_group_ratios:
            result['isomorphic_group_ratios'] = isomorphic_group_ratios
    
    # print(f"[🔧] Extracted {len(result)} fields using MAC-aware regex")
    
    # Validate MAC allocation if present
    if 'isomorphic_group_ratios' in result:
        mac_alloc = result['isomorphic_group_ratios']
        total_allocation = sum(mac_alloc.values())
        if total_allocation > 100:
            # print(f"[⚠️] MAC allocation totals {total_allocation:.1f}% > 100% in regex extraction")
            pass
    
    return result


def _validate_and_normalize_mac_fields(result: dict) -> dict:
    """Validate and normalize MAC-specific fields in parsed JSON"""
    if 'target_macs' in result and isinstance(result['target_macs'], (int, float)):
        result['target_macs'] = float(result['target_macs'])
    if 'macs_overshoot_tolerance_pct' in result and isinstance(result['macs_overshoot_tolerance_pct'], (int, float)):
        result['macs_overshoot_tolerance_pct'] = float(result['macs_overshoot_tolerance_pct'])
    if 'macs_undershoot_tolerance_pct' in result and isinstance(result['macs_undershoot_tolerance_pct'], (int, float)):
        result['macs_undershoot_tolerance_pct'] = float(result['macs_undershoot_tolerance_pct'])
    if 'continue' in result and isinstance(result['continue'], str):
        result['continue'] = result['continue'].lower() == 'true'
    return result

def _extract_mac_fields_regex(content: str) -> dict:
    """Extract MAC fields using regex as fallback"""
    import re
    result = {}
    
    # Extract common MAC fields with regex
    patterns = {
        'target_macs': r'"target_macs":\s*([0-9.]+)',
        'macs_overshoot_tolerance_pct': r'"macs_overshoot_tolerance_pct":\s*([0-9.]+)',
        'macs_undershoot_tolerance_pct': r'"macs_undershoot_tolerance_pct":\s*([0-9.]+)',
        'continue': r'"continue":\s*(true|false)',
        'mac_budget_focused': r'"mac_budget_focused":\s*(true|false)',
        'dataset_adapted': r'"dataset_adapted":\s*(true|false)',
        'rationale': r'"rationale":\s*"([^"]*)"',
        'isomorphic_group_ratios_tuning_order': r'"isomorphic_group_ratios_tuning_order":\s*\[(.*?)\]'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            if key in ['target_macs', 'macs_overshoot_tolerance_pct', 'macs_undershoot_tolerance_pct']:
                result[key] = float(value)
            elif key in ['continue', 'mac_budget_focused', 'dataset_adapted']:
                result[key] = value.lower() == 'true'
            else:
                result[key] = value
    
    return result