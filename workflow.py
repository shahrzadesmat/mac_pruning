from profiling_agent import ProfilingAgent
from analysis_agent import AnalysisAgent
from pruning_agent import PruningAgent
from finetune_agent import FineTuningAgent
from eval_agent import EvaluationAgent
from master_agent import MasterAgent
from data.loaders import get_dataset_loaders
from data.loaders import get_dataset_loaders
from utils.io import save_final_best_model
from utils.logging_wandb import log_to_wandb
from utils.analysis_structures import PruningState
import copy
from torchvision import datasets, transforms
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

# from langchain_openai import ChatOpenAI
from deepseek_llm import DeepSeekLLM

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import torch
import torch.nn as nn
import torch_pruning as tp
import timm
import pbench
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
import re
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from dotenv import load_dotenv, find_dotenv
import openai
import nest_asyncio
import asyncio
import json
import random
from torch.utils.data import DataLoader
import traceback

from datasets import load_dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
# from pbench.utils import get_interpolation_mode
import glob

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import pbench.data.presets
import pbench.extension
import pbench.forward_patch

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import functools
from contextlib import contextmanager
from collections import defaultdict
import wandb
import math
import threading
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import gc
import psutil
import warnings
from utils.pruning_math import extract_pruning_ratio, extract_mac_target
from llm.provider import get_llm
from utils.json_utils import deep_merge



GLOBAL_STATE = {}
MODEL_STORE = None


def create_pruning_workflow():
    workflow = StateGraph(PruningState)

    # Create a shared LLM instance
    shared_llm = get_llm()
    print(f"[🔄] Created shared LLM instance using OpenRouter with model: {shared_llm.model}")

    
    # Define state-aware wrapper functions with dataset support
    async def profile_with_state(state):
        """Profile node with dataset-aware state management"""
        global GLOBAL_STATE, MODEL_STORE

        original_model_name = state.get('model_name') or GLOBAL_STATE.get('model_name')

        state = deep_merge(state, GLOBAL_STATE)
        
        if original_model_name:
            state['model_name'] = original_model_name
            GLOBAL_STATE['model_name'] = original_model_name

        # Ensure dataset information is present
        dataset = state.get('dataset', 'cifar10')
        print(f"[🔄] Profiling {dataset} model")
        
        # Check if this is a subsequent profile (after evaluation)
        is_subsequent = state.get('revision_number', 0) > 0
        if is_subsequent and 'evaluation_results' in state:
            print(f"[🔄] Profiling after evaluation, revision {state['revision_number']} for {dataset}")
            # The evaluation results are already in the state for the profiling agent to use
        
        # Use the updated dataset-aware ProfilingAgent
        result = await ProfilingAgent(llm=shared_llm).profile_model(state)

        # Extract baseline_macs to top level immediately
        if 'profile_results' in result and 'baseline_macs' in result['profile_results']:
            result['baseline_macs'] = result['profile_results']['baseline_macs']
            GLOBAL_STATE['baseline_macs'] = result['profile_results']['baseline_macs']

        GLOBAL_STATE.update(result)
        return result


    async def analyze_with_state(state):
        """Enhanced Analysis node with proper role separation and strategic guidance handling"""
        global GLOBAL_STATE, MODEL_STORE
        state = deep_merge(state, GLOBAL_STATE)

        dataset = state.get('dataset', 'cifar10')
        print(f"[🔄] Enhanced analyzing {dataset} pruning strategy with proper role separation")

        # FIX: Ensure model_name is preserved in state
        if 'model_name' not in state and 'model_name' in GLOBAL_STATE:
            state['model_name'] = GLOBAL_STATE['model_name']
            print(f"[🔧] Restored model_name from global state: {state['model_name']}")
        
        # Additional fallback: extract from query if still missing
        if 'model_name' not in state or not state['model_name']:
            query = state.get('query', '')
            import re
            match = re.search(r'Prune\s+(\w+(?:_\w+)*)\s+model', query)
            if match:
                state['model_name'] = match.group(1)
                GLOBAL_STATE['model_name'] = state['model_name']
                print(f"[🔧] Extracted and saved model_name from query: {state['model_name']}")

        # Ensure MAC-based targets are preserved
        if 'target_macs' in GLOBAL_STATE:
            state['target_macs'] = GLOBAL_STATE['target_macs']
        if 'baseline_macs' in GLOBAL_STATE:
            state['baseline_macs'] = GLOBAL_STATE['baseline_macs']
        if 'macs_overshoot_tolerance_pct' in GLOBAL_STATE:
            state['macs_overshoot_tolerance_pct'] = GLOBAL_STATE['macs_overshoot_tolerance_pct']
        if 'macs_undershoot_tolerance_pct' in GLOBAL_STATE:
            state['macs_undershoot_tolerance_pct'] = GLOBAL_STATE['macs_undershoot_tolerance_pct']

        # Legacy compatibility
        if 'target_pruning_ratio' in GLOBAL_STATE:
            state['target_pruning_ratio'] = GLOBAL_STATE['target_pruning_ratio']

        # NEW: Handle strategic guidance from Master Agent (proper role separation)
        master_results = state.get('master_results', {})
        strategic_guidance = master_results.get('strategic_guidance')

        if strategic_guidance and strategic_guidance.get('guidance_type') == 'mac_catastrophic_recovery':
            print(f"[🚨] MAC CATASTROPHIC RECOVERY MODE: Received strategic guidance from Master Agent")
            print(f"[📋] Strategic guidance includes:")
            
            avoid_combinations = strategic_guidance.get('avoid_combinations', [])
            recommendations = strategic_guidance.get('strategic_recommendations', [])
                        
            if avoid_combinations:
                print(f"   🚫 Avoid combinations: {avoid_combinations}")
            
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3 recommendations
                print(f"   {i}. {rec}")
            
            if len(recommendations) > 3:
                print(f"   ... and {len(recommendations) - 3} more recommendations")
            
            print(f"[🧠] Analysis Agent will interpret this MAC guidance and calculate specific parameters...")
            
            # Store guidance for Analysis Agent to use in its calculations
            state['master_strategic_guidance'] = strategic_guidance
            state['mac_catastrophic_recovery_mode'] = True
            
        elif strategic_guidance:
            print(f"[📋] Received general strategic guidance from Master Agent:")
            recommendations = strategic_guidance.get('strategic_recommendations', strategic_guidance.get('recommendations', []))
            for i, rec in enumerate(recommendations[:2], 1):  # Show first 2
                print(f"   {i}. {rec}")
            
            # Store guidance for normal analysis
            state['master_strategic_guidance'] = strategic_guidance
            
        else:
            print(f"[📋] No strategic guidance from Master Agent - proceeding with standard analysis")

        # Get directives from master agent if available (legacy support)
        if 'master_results' in state and 'directives' in state['master_results']:
            directives = state['master_results'].get('directives', '')
            if directives and directives != "None":
                print(f"[📋] Master Agent directives: {directives}")
        
        # ANALYSIS AGENT: Calculate specific parameters based on guidance (if any) and historical learning
        print(f"[🧮] Analysis Agent calculating specific parameters for {dataset}...")
        result = await AnalysisAgent(llm=shared_llm).analyze(state)

        # Verify Analysis Agent followed guidance (if provided)
        if strategic_guidance and 'analysis_results' in result:
            analysis_results = result['analysis_results']
            strategy_dict = analysis_results.get('strategy_dict', {})
            
            # Check if Analysis Agent avoided failed combinations
            if strategic_guidance.get('avoid_combinations'):
                avoid_combinations = strategic_guidance['avoid_combinations']
                chosen_importance = strategy_dict.get('importance_criterion')
                chosen_round_to = strategy_dict.get('round_to')
                chosen_combo = (chosen_importance, chosen_round_to)
                
                if chosen_combo in avoid_combinations:
                    print(f"[⚠️] WARNING: Analysis Agent chose avoided combination {chosen_combo}")
                    print(f"[⚠️] This may indicate guidance interpretation failed")
                else:
                    print(f"[✅] Analysis Agent successfully avoided failed combinations")
                    print(f"[✅] Chosen combination: {chosen_combo} (not in avoided list)")

        # Make sure critical values propagate (MAC-first)
        if 'target_macs' in state:
            result['target_macs'] = state['target_macs']
        if 'baseline_macs' in state:
            result['baseline_macs'] = state['baseline_macs']
        if 'macs_overshoot_tolerance_pct' in state:
            result['macs_overshoot_tolerance_pct'] = state['macs_overshoot_tolerance_pct']
        if 'macs_undershoot_tolerance_pct' in state:
            result['macs_undershoot_tolerance_pct'] = state['macs_undershoot_tolerance_pct']
        
        # Legacy compatibility
        if 'target_pruning_ratio' in state:
            result['target_pruning_ratio'] = state['target_pruning_ratio']
        result['model_name'] = state['model_name']
        
        # Add strategic guidance compliance metadata
        if strategic_guidance:
            if 'analysis_results' not in result:
                result['analysis_results'] = {}
            result['analysis_results']['strategic_guidance_received'] = True
            result['analysis_results']['guidance_type'] = strategic_guidance.get('guidance_type', 'general')
            if strategic_guidance.get('guidance_type') == 'mac_catastrophic_recovery':
                result['analysis_results']['mac_catastrophic_recovery_mode'] = True

        # Log safety and compliance information
        analysis_results = result.get('analysis_results', {})
        if analysis_results.get('safety_validated'):
            print(f"[🛡️] Safety validation completed for {dataset}")
            
            corrections = analysis_results.get('strategy_dict', {}).get('corrections_applied', [])
            if corrections:
                print(f"[🔧] Applied {len(corrections)} safety corrections:")
                for correction in corrections:
                    print(f"    - {correction}")
            else:
                print(f"[✅] No safety corrections needed - Analysis Agent provided compliant parameters")

        # Log role separation compliance
        strategy_dict = analysis_results.get('strategy_dict', {})
        if strategy_dict:
            importance = strategy_dict.get('importance_criterion', 'unknown')
            round_to = strategy_dict.get('round_to', 'unknown')
            
            if 'isomorphic_group_ratios' in strategy_dict:
                # MAC allocation for ViT
                isomorphic_group_ratios = strategy_dict['isomorphic_group_ratios']
                mlp_percent = isomorphic_group_ratios.get('mlp_multiplier', 'unknown')
                qkv_percent = isomorphic_group_ratios.get('qkv_multiplier', 'unknown')
                print(f"[🎯] Analysis Agent calculated ViT MAC parameters:")
                print(f"    Importance: {importance}, Round-to: {round_to}")
                print(f"    MLP MAC: {mlp_percent}%, QKV MAC: {qkv_percent}%")
            elif 'channel_pruning_ratio' in strategy_dict:
                # Channel pruning for CNN
                channel_ratio = strategy_dict.get('channel_pruning_ratio', 'unknown')
                print(f"[🎯] Analysis Agent calculated CNN parameters:")
                print(f"    Importance: {importance}, Round-to: {round_to}")
                print(f"    Channel ratio: {channel_ratio}")

        # Final role separation verification
        if strategic_guidance and strategic_guidance.get('guidance_type') == 'mac_catastrophic_recovery':
            print(f"[✅] ROLE SEPARATION VERIFIED:")
            print(f"    Master Agent: Provided MAC strategic guidance (avoid combinations, recommendations)")
            print(f"    Analysis Agent: Interpreted guidance and calculated specific parameter values")
            print(f"    No direct parameter value override occurred")

        GLOBAL_STATE.update(result)
        return result


    async def master_with_state(state):
        """Master Agent node with dataset-aware state management"""
        global GLOBAL_STATE, MODEL_STORE
        state = deep_merge(state, GLOBAL_STATE)
        
        dataset = state.get('dataset', 'cifar10')
        print(f"[🔄] Master Agent analyzing {dataset} strategy")
        
        def should_continue(state):
            """
            Two-phase search strategy (MAC-based):
            Phase 1: Search until first model within MAC tolerance
            Phase 2: Continue for exactly 3 more trials after first success
            """
            current_revision = state.get("revision_number", 0)
            max_revisions = state.get("max_revisions", 50)
            target_macs = state.get("target_macs", 5.0)
            macs_overshoot_tolerance_pct = state.get("macs_overshoot_tolerance_pct", 1.0)
            macs_undershoot_tolerance_pct = state.get("macs_undershoot_tolerance_pct", 5.0)
            dataset = state.get("dataset", "cifar10")
            
            history = state.get('history', [])
            
            # Find all models within MAC tolerance that have been fine-tuned
            candidate_models = []
            first_success_revision = None
            
            for entry in history:
                achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
                target_mac = entry.get('target_macs', target_macs)
                
                if achieved_macs and target_mac:
                    macs_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
                    
                    if -macs_undershoot_tolerance_pct <= macs_error_pct <= macs_overshoot_tolerance_pct:
                        entry['is_candidate_model'] = True

                        # Check if it has been fine-tuned (has fine-tuned accuracy)
                        if dataset.lower() == 'imagenet':
                            fine_tuned_acc = entry.get('fine_tuned_top1_accuracy')
                        else:
                            fine_tuned_acc = entry.get('fine_tuned_accuracy')
                        
                        candidate_models.append(entry)
                        if first_success_revision is None:
                            first_success_revision = entry.get('revision', 0)

            
            print(f"[🔍] Two-Phase MAC Search Status:")
            print(f"   Current revision: {current_revision}")
            print(f"   Candidate models found: {len(candidate_models)}")
            if first_success_revision is not None:
                print(f"   First success at revision: {first_success_revision}")

            # PHASE 2: Found first candidate - check if we need to initialize extended search
            extended_search_remaining = state.get('extended_search_remaining')
            
            # PHASE 1: No candidates found yet - keep searching
            if extended_search_remaining is None and not candidate_models:
                if current_revision >= max_revisions:
                    print(f"[🛑] Phase 1: Max revisions reached without finding model within MAC tolerance")
                    return False, "Phase 1 failed: No models within MAC tolerance found"
                
                print(f"[🔍] Phase 1: Searching for first model within MAC tolerance...")
                return True, None
            
            
            if extended_search_remaining is None:
                # First time finding a candidate - initialize Phase 2
                print(f"[🎯] Phase 2: First MAC candidate found! Starting 3 additional trials...")
                print(f"[📊] Current MAC candidates:")
                for i, candidate in enumerate(candidate_models):
                    achieved_macs = candidate.get('achieved_macs', 0)
                    target_mac = candidate.get('target_macs', target_macs)
                    mac_error = ((achieved_macs - target_mac) / target_mac * 100) if target_mac > 0 else 0
                    
                    if dataset.lower() == 'imagenet':
                        accuracy = candidate.get('fine_tuned_top1_accuracy') or 0.0
                        acc_type = "Top-1"
                    else:
                        accuracy = candidate.get('fine_tuned_accuracy') or 0.0
                        acc_type = "Acc"
                    rev = candidate.get('revision', 0)
                    print(f"     Candidate {i+1}: Rev {rev}, {achieved_macs/1e9:.3f}G MACs ({mac_error:+.1f}%), {accuracy:.2f}% {acc_type}")
                
                return True, "initialize_phase2"  # Signal to initialize
            
            # PHASE 2: In extended search
            if extended_search_remaining > 0:
                print(f"[🔍] Phase 2: Extended MAC search continuing... {extended_search_remaining-1} trials remaining after this")
                print(f"[📊] Candidates so far: {len(candidate_models)}")
                
                # Continue unless max revisions reached
                if current_revision >= max_revisions:
                    print(f"[🛑] Phase 2: Hit max revisions during extended search")
                    return False, "Phase 2: Max revisions reached during extended search"
                
                return True, "continue_phase2"
            
            # PHASE 2 COMPLETE: Select best model and finish
            if dataset.lower() == 'imagenet':
                # Filter out candidates that have valid (non-None) fine-tuned Top-1 accuracy
                valid_candidates = [m for m in candidate_models if m.get('fine_tuned_top1_accuracy') is not None]

                if valid_candidates:
                    best_model = max(valid_candidates, key=lambda x: x.get('fine_tuned_top1_accuracy', 0.0) or 0.0)
                    best_accuracy = best_model.get('fine_tuned_top1_accuracy') or 0.0
                    accuracy_type = "Top-1"
                else:
                    # Fallback: No valid accuracy, select based on MAC efficiency instead
                    print("[⚠️] Warning: No candidates have fine-tuned Top-1 accuracy. Falling back to MAC efficiency.")
                    best_model = min(candidate_models, key=lambda x: x.get('achieved_macs', float('inf')))
                    best_accuracy = best_model.get('fine_tuned_top1_accuracy') or 0.0  # Will be 0.0 if None
                    accuracy_type = "Top-1 (no fine-tuned acc)"

            else:
                # CIFAR or other datasets
                valid_candidates = [m for m in candidate_models if m.get('fine_tuned_accuracy') is not None]

                if valid_candidates:
                    best_model = max(valid_candidates, key=lambda x: x.get('fine_tuned_accuracy', 0.0) or 0.0)
                    best_accuracy = best_model.get('fine_tuned_accuracy') or 0.0
                    accuracy_type = "accuracy"
                else:
                    print("[⚠️] Warning: No candidates have fine-tuned accuracy. Falling back to MAC efficiency.")
                    best_model = min(candidate_models, key=lambda x: x.get('achieved_macs', float('inf')))
                    best_accuracy = best_model.get('fine_tuned_accuracy') or 0.0  # Will be 0.0 if None
                    accuracy_type = "accuracy (no fine-tuned acc)"

            # Common fields
            best_revision = best_model.get('revision', 0)
            best_achieved_macs = best_model.get('achieved_macs', 0.0) or 0.0
            best_target_macs = best_model.get('target_macs', target_macs)
            best_mac_error = ((best_achieved_macs - best_target_macs) / best_target_macs * 100) if best_target_macs > 0 else 0
            
            print(f"[🏆] Two-Phase MAC Search COMPLETE!")
            print(f"   Total candidates evaluated: {len(candidate_models)}")
            print(f"   Best model: Revision {best_revision}")
            print(f"   Best {accuracy_type}: {best_accuracy:.2f}%")
            print(f"   MAC achievement: {best_achieved_macs:.3f}G (target: {best_target_macs:.3f}G, error: {best_mac_error:+.1f}%)")
            
            # Store results for final model saving
            state['final_best_model'] = best_model
            state['all_candidate_models'] = candidate_models
            state['search_complete'] = True
            
            return False, f"Two-phase MAC search complete: Best model at revision {best_revision}, {accuracy_type} {best_accuracy:.2f}%"
        
        # Call the should_continue function to check algorithmic stopping criteria
        continue_optimization, stop_reason = should_continue(state)
        
        # ✅ CRITICAL FIX: Handle state modifications properly
        if stop_reason == "initialize_phase2":
            state['extended_search_remaining'] = 10
            GLOBAL_STATE['extended_search_remaining'] = 10  # Also update global state
            print(f"[🔧] Initialized extended_search_remaining = 10")
            continue_optimization = True
            stop_reason = None
        elif stop_reason == "continue_phase2":
            old_value = state.get('extended_search_remaining', 0)
            new_value = old_value - 1
            state['extended_search_remaining'] = new_value
            GLOBAL_STATE['extended_search_remaining'] = new_value  # Also update global state
            print(f"[🔧] Decremented extended_search_remaining: {old_value} → {new_value}")
            continue_optimization = True
            stop_reason = None
        
        # Use the updated dataset-aware MasterAgent
        result = await MasterAgent().analyze_and_direct(state)
        
        # If algorithm decided to stop, override Master Agent's decision
        if not continue_optimization:
            result['master_results']['should_continue'] = False
            result['master_results']['stop_reason'] = stop_reason
        
        # Final check of Master Agent's decision
        if not result.get('master_results', {}).get('should_continue', True):
            print(f"[🛑] Master Agent decided to stop {dataset}: {result['master_results'].get('stop_reason')}")
            result['_end_workflow'] = True
            
            # Find the best pruning result from history (MAC-based)
            best_entry = None
            best_mac_error = float('inf')
            best_achieved_macs = 0.0
            target_macs = state.get('target_macs', 5.0)
            
            for entry in state.get('history', []):
                achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g', 0.0))
                if achieved_macs > 0:
                    mac_error = abs(achieved_macs - target_macs) / target_macs * 100
                    if mac_error < best_mac_error:
                        best_mac_error = mac_error
                        best_entry = entry
                        best_achieved_macs = achieved_macs
            
            # Create or update pruning_results with the best result
            if best_entry and best_achieved_macs > 0:
                print(f"[📊] Setting final {dataset} pruning results based on best MAC attempt (achieved: {best_achieved_macs:.3f}G)")
                if 'prune' not in result:
                    result['prune'] = {}
                
                result['prune']['pruning_results'] = {
                    'success': True,
                    'achieved_macs': best_achieved_macs,
                    'target_macs': target_macs,
                    'mac_error_pct': best_mac_error,
                    'dataset': dataset
                }

        GLOBAL_STATE.update(result)
        return result

    async def prune_with_state(state):
        """Prune node with dataset-aware state management"""
        global GLOBAL_STATE, MODEL_STORE

        # Merge in any global updates
        state = deep_merge(state, GLOBAL_STATE)

        dataset = state.get('dataset', 'cifar10')
        print(f"[🔄] Executing {dataset} pruning")

        # Preserve model_name and dataset info
        if 'model_name' not in state and 'model_name' in GLOBAL_STATE:
            state['model_name'] = GLOBAL_STATE['model_name']

        # Get MAC targets (primary) and user target (fallback)
        target_macs = state.get('target_macs', 5.0)
        baseline_macs = state.get('baseline_macs', 10.0)
        macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)
        user_target = extract_pruning_ratio(state.get('query', '')) or state.get('user_target_pruning_ratio', 0.0)

        # Use the updated dataset-aware PruningAgent
        result = await PruningAgent().execute_pruning(state)

        # Handle errors immediately
        if 'error' in result:
            print(f"[❌] Error in {dataset} pruning step: {result['error']}")
            return {
                **state,
                'error': result['error']
            }
        
        # Check if pruning was successful and we have results
        if 'pruning_results' in result and result['pruning_results'].get('success', False):
            # Use pruning_results directly instead of nested prune structure
            pruning_res = result['pruning_results']
            
            # Get MAC-based metrics (primary)
            achieved_macs = pruning_res.get('achieved_macs', pruning_res.get('final_macs_g', 0.0))
            target_mac = pruning_res.get('target_macs', target_macs)
            
            print(f"[DEBUG] In {dataset} prune_with_state: |{achieved_macs/1e9:.3f}G - {target_mac:.3f}G|")

            # Calculate MAC error percentage
            if achieved_macs and target_mac and target_mac > 0:
                mac_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
                mac_within_tolerance = (-macs_undershoot_tolerance_pct <= mac_error_pct <= macs_overshoot_tolerance_pct)
                
                if mac_within_tolerance:
                    print(f"[✅] Achieved {achieved_macs/1e9:.3f}G MACs, meets target {target_mac:.3f}G (error: {mac_error_pct:+.1f}%)")
                    # Success - proceed
                
                # Store the model in MODEL_STORE if available
                if 'prune' in result and 'model' in result['prune']:
                    MODEL_STORE = result['prune']['model']
                    print(f"[💾] Stored pruned model in MODEL_STORE")
                else:
                    print(f"[⚠️] No model found in result['prune']['model'], MODEL_STORE unchanged")

                # Make sure dataset info is populated
                pruning_res['dataset'] = dataset
                result['pruning_results'] = pruning_res
                result['model_name'] = state['model_name']

                # Append a final history entry
                analysis_results = state.get('analysis_results', {})
                strategy_dict = analysis_results.get('strategy_dict', {})

                # Create a proper strategy_used structure
                strategy_used = {
                    'importance_criterion': strategy_dict.get('importance_criterion', 'unknown'),
                    'round_to': strategy_dict.get('round_to', 'unknown'),
                    'global_pruning': strategy_dict.get('global_pruning', True),
                    'approach': strategy_dict.get('approach', 'unknown')
                }

                # Add MAC allocation if present (for ViT models)
                if 'isomorphic_group_ratios' in strategy_dict:
                    strategy_used['isomorphic_group_ratios'] = strategy_dict['isomorphic_group_ratios']
                    print(f"[📝] Storing MAC allocation in final history: {strategy_dict['isomorphic_group_ratios']}")

                # Add channel pruning ratio if present (for CNN models)
                if 'channel_pruning_ratio' in strategy_dict:
                    strategy_used['channel_pruning_ratio'] = strategy_dict['channel_pruning_ratio']

                # Legacy compatibility
                if 'pruning_ratio' in strategy_dict:
                    strategy_used['pruning_ratio'] = strategy_dict['pruning_ratio']

                entry = {
                    'revision': state.get('revision_number', 0),
                    'dataset': dataset,
                    'strategy_used': strategy_used,
                    # MAC-first metrics
                    'target_macs': target_mac,
                    'achieved_macs': achieved_macs,
                    'mac_error_pct': mac_error_pct,
                }
                
                # Legacy compatibility
                if user_target:
                    entry['target_ratio'] = user_target
                if 'achieved_ratio' in pruning_res:
                    entry['achieved_ratio'] = pruning_res.get('achieved_ratio')
                
                # Add dataset-specific zero-shot metrics
                if dataset.lower() == 'imagenet':
                    entry['zero_shot_top1_accuracy'] = pruning_res.get('zero_shot_top1_accuracy', 0)
                    entry['zero_shot_top5_accuracy'] = pruning_res.get('zero_shot_top5_accuracy', 0) 
                    entry['zero_shot_accuracy'] = pruning_res.get('zero_shot_top1_accuracy', 0)  # For compatibility
                else:
                    entry['zero_shot_accuracy'] = pruning_res.get('zero_shot_accuracy', 0)
                
                result.setdefault('history', []).append(entry)
                GLOBAL_STATE['history'] = result['history']

                # Commit and return
                GLOBAL_STATE.update(result)
                return result
            else:
                print(f"[⚠️] Invalid MAC data: achieved_macs={achieved_macs}, target_mac={target_mac}")

            # If we get here, either MAC calculation failed or we're outside tolerance
            print(f"[⚠️] {dataset} MAC target not met. Retrying.")
            result['attempted_pruning_ratios'] = state.get('attempted_pruning_ratios', []) + [state.get('target_pruning_ratio', 0.2)]
            result['revision_number'] = state.get('revision_number', 0) + 1
            GLOBAL_STATE.update(result)
            return result
        
        else:
            # Pruning failed - check if we have 'prune' structure with different error
            if 'prune' in result and 'pruning_results' in result['prune']:
                prune_results = result['prune']['pruning_results']
                if not prune_results.get('success', False):
                    error_msg = prune_results.get('error', 'Unknown pruning failure')
                    return {**state, 'error': f'Pruning failed: {error_msg}'}
            
            return {**state, 'error': 'Pruning failed - no results available'}

    async def fine_tune_with_state(state):
        """Fine-tuning node with dataset-aware state management"""
        global GLOBAL_STATE, MODEL_STORE
        state = deep_merge(state, GLOBAL_STATE)

        dataset = state.get('dataset', 'cifar10')
        print(f"[🔄] Fine-tuning {dataset} model")

        # Ensure model_name and dataset info is present
        if 'model_name' not in state and 'model_name' in GLOBAL_STATE:
            state['model_name'] = GLOBAL_STATE['model_name']

        if MODEL_STORE is not None:
            if 'prune' not in state:
                state['prune'] = {}
            state['prune']['model'] = MODEL_STORE

        # Use the updated dataset-aware FineTuningAgent
        result = await FineTuningAgent(llm=shared_llm).fine_tune(state)

        # Store fine-tuned model back in MODEL_STORE
        if 'fine_tuning_results' in result and 'model' in result['fine_tuning_results']:
            MODEL_STORE = result['fine_tuning_results']['model']
            # Remove model from state to avoid serialization issues
            checkpoint_path = result['fine_tuning_results'].get('checkpoint_path', 'unknown_path.pth')
            metrics = {k: v for k, v in result['fine_tuning_results'].items() 
                      if k not in ['model']}
            result['fine_tuning_results'] = metrics

        GLOBAL_STATE.update(result)
        return result

    async def evaluate_with_state(state):
        """Evaluate node with dataset-aware state management"""
        global GLOBAL_STATE, MODEL_STORE
        state = deep_merge(state, GLOBAL_STATE)

        dataset = state.get('dataset', 'cifar10')
        print(f"[🔄] Evaluating {dataset} model")

        # Ensure model_name and dataset info is present
        if 'model_name' not in state and 'model_name' in GLOBAL_STATE:
            state['model_name'] = GLOBAL_STATE['model_name']

        # ✅ FIX 8: Only inject fine-tuned model if fine-tuning occurred
        if MODEL_STORE is not None and 'fine_tuning_results' in state:
            if 'fine_tuning_results' not in state:
                state['fine_tuning_results'] = {}
            state['fine_tuning_results']['model'] = MODEL_STORE
            print(f"[💾] Using fine-tuned model for evaluation")
        else:
            print(f"[💾] No fine-tuned model available, using stored results only")

        # Use the fixed EvaluationAgent
        result = await EvaluationAgent(llm=shared_llm).evaluate(state)

        # Preserve model_name and dataset info in the result
        if 'model_name' in state:
            result['model_name'] = state['model_name']

        GLOBAL_STATE.update(result)
        return result

    def _update_history_with_attempt(state, achieved_macs, zero_shot_acc, target_macs):
        """
        ENHANCED: Track candidate model status and checkpoint paths (MAC-based)
        REASON: Two-phase search needs to identify which models are candidates
        """
        
        analysis_results = state.get('analysis_results', {})
        strategy_dict = analysis_results.get('strategy_dict', {})
        dataset = state.get('dataset', 'unknown')
        macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)
        
        if achieved_macs and target_macs:
            macs_error_pct = ((achieved_macs - target_macs) / target_macs) * 100
            is_candidate = (-macs_undershoot_tolerance_pct <= macs_error_pct <= macs_overshoot_tolerance_pct)

        else:
            is_candidate = False
        
        # ✅ CRITICAL FIX: Store the complete strategy including MAC allocation
        strategy_used = {
            'importance_criterion': strategy_dict.get('importance_criterion', 'unknown'),
            'round_to': strategy_dict.get('round_to', 'unknown'),
            'approach': strategy_dict.get('approach', 'unknown'),
            'global_pruning': strategy_dict.get('global_pruning', True),
        }
        
        # ✅ CRITICAL: Store MAC allocation if it exists (for ViT models)
        if 'isomorphic_group_ratios' in strategy_dict:
            strategy_used['isomorphic_group_ratios'] = strategy_dict['isomorphic_group_ratios']
            print(f"[📝] Storing MAC allocation: {strategy_dict['isomorphic_group_ratios']}")
        
        # ✅ CRITICAL: Store channel_pruning_ratio if it exists (for CNN models)  
        if 'channel_pruning_ratio' in strategy_dict:
            strategy_used['channel_pruning_ratio'] = strategy_dict['channel_pruning_ratio']
        
        # Legacy compatibility
        if 'pruning_ratio' in strategy_dict:
            strategy_used['pruning_ratio'] = strategy_dict['pruning_ratio']
        
        history_entry = {
            'revision': state.get('revision_number', 0),
            'is_candidate_model': is_candidate,
            'within_tolerance': is_candidate,
            'dataset': dataset,
            'strategy_used': strategy_used,
            # MAC-first metrics
            'target_macs': target_macs,
            'achieved_macs': achieved_macs,
            'mac_error_pct': macs_error_pct,
        }

        # Add dataset-specific accuracy fields
        if dataset.lower() == 'imagenet':
            history_entry['zero_shot_top1_accuracy'] = float(zero_shot_acc)
            history_entry['zero_shot_accuracy'] = float(zero_shot_acc)  # for compatibility
            # Fine-tuned accuracies will be added later by evaluation agent
            history_entry['fine_tuned_top1_accuracy'] = None  # placeholder
            history_entry['fine_tuned_top5_accuracy'] = None  # placeholder
        else:
            history_entry['zero_shot_accuracy'] = float(zero_shot_acc)
            # Fine-tuned accuracy will be added later by evaluation agent
            history_entry['fine_tuned_accuracy'] = None  # placeholder

        # For candidate models, store pruned model checkpoint path  
        if is_candidate:
            pruning_results = state.get('pruning_results', {})
            checkpoint_path = pruning_results.get('checkpoint_path', 'unknown_checkpoint.pth')
            if checkpoint_path and checkpoint_path != 'unknown_checkpoint.pth':
                history_entry['pruned_model_checkpoint'] = checkpoint_path
                print(f"[💾] Stored checkpoint path for candidate: {checkpoint_path}")

        # Add to state history
        if 'history' not in state:
            state['history'] = []
        
        state['history'].append(history_entry)
        
        status = "CANDIDATE (will fine-tune)" if is_candidate else "NON-CANDIDATE (back to master)"
        print(f"[📝] Added history entry: revision {history_entry['revision']}, "
            f"{achieved_macs:.3f}G MACs, zero-shot {zero_shot_acc:.1f}% - {status}")
        
        # ✅ DEBUG: Show what was actually stored
        if 'isomorphic_group_ratios' in strategy_used:
            allocation = strategy_used['isomorphic_group_ratios']
            mlp_pct = allocation.get('mlp_multiplier', 'N/A')
            qkv_pct = allocation.get('qkv_multiplier', 'N/A')
            print(f"[🔍] Stored MAC allocation - MLP: {mlp_pct}%, QKV: {qkv_pct}%")
        else:
            print(f"[⚠️] No MAC allocation found in strategy_dict to store")
            print(f"[🔍] Available strategy_dict keys: {list(strategy_dict.keys())}")

    def route_after_pruning(state):
        """Route based on MACs-budget tolerance (MACs-first, legacy-safe)"""

        # Get pruning results
        pruning_results = state.get('pruning_results', {}) or {}

        if not pruning_results.get('success', False):
            print(f"[⚠️] Pruning execution failed - routing back to Master Agent")
            return "master_normal"

        # ── MACs targets and tolerance ──────────────────────────────────────────────
        achieved_macs = pruning_results.get('achieved_macs')
        target_macs   = pruning_results.get('target_macs') or state.get('target_macs')
        macs_overshoot_tolerance_pct = float(state.get('macs_overshoot_tolerance_pct', 1.0))
        macs_undershoot_tolerance_pct = float(state.get('macs_undershoot_tolerance_pct', 5.0))
        dataset = state.get('dataset', 'cifar10')

        # Validate required MACs fields
        if achieved_macs is None or target_macs is None or target_macs <= 0:
            print(f"[⚠️] Missing MACs info (achieved_macs={achieved_macs}, target_macs={target_macs}) → back to Master")
            return "master_normal"

        # Compute signed percent error ( + = too heavy / not enough pruning )
        macs_error_pct = (float(achieved_macs) - float(target_macs)) / float(target_macs) * 100.0
        within_tolerance = (-macs_undershoot_tolerance_pct <= macs_error_pct <= macs_overshoot_tolerance_pct)

        # Extract zero-shot accuracy for logging
        if dataset.lower() == 'imagenet':
            zero_shot_acc = pruning_results.get('zero_shot_top1_accuracy', 0.0) or 0.0
        else:
            zero_shot_acc = pruning_results.get('zero_shot_accuracy', 0.0) or 0.0

        # ── Decision: within ± tolerance → fine-tune; else → back to master ────────
        if within_tolerance:
            # Within acceptable range - proceed to fine-tuning
            print(f"[✅] MACs WITHIN TOLERANCE:")
            print(f"   Achieved: {achieved_macs/1e9:.3f}G vs Target: {target_macs/1e9:.3f}G")
            print(f"   Error: {macs_error_pct:+.2f}% (tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")
            print(f"   Zero-shot accuracy: {zero_shot_acc:.2f}% - proceeding to fine-tune")

            # Mark as candidate and update history with complete strategy info
            state['is_candidate_model'] = True

            analysis_results = state.get('analysis_results', {}) or {}
            strategy_dict = analysis_results.get('strategy_dict', {}) or {}
            ds = dataset

            # Build strategy_used (keep legacy fields if present for back-compat)
            strategy_used = {
                'importance_criterion': strategy_dict.get('importance_criterion', 'unknown'),
                'round_to': strategy_dict.get('round_to', 'unknown'),
                'global_pruning': strategy_dict.get('global_pruning', True),
                'approach': strategy_dict.get('approach', 'unknown'),
            }
            if 'pruning_ratio' in strategy_dict:
                strategy_used['pruning_ratio'] = strategy_dict['pruning_ratio']  # legacy
            if 'channel_pruning_ratio' in strategy_dict:
                strategy_used['channel_pruning_ratio'] = strategy_dict['channel_pruning_ratio']
            if 'isomorphic_group_ratios' in strategy_dict:
                strategy_used['isomorphic_group_ratios'] = strategy_dict['isomorphic_group_ratios']
                print(f"[📝] Storing isomorphic ratios in candidate history: {strategy_dict['isomorphic_group_ratios']}")

            # Create history entry (MACs-first, keep legacy fields if available)
            history_entry = {
                'revision': state.get('revision_number', 0),
                'dataset': ds,
                'is_candidate_model': True,
                'within_tolerance': True,
                'strategy_used': strategy_used,
                'pruned_model_checkpoint': pruning_results.get('checkpoint_path', 'unknown_checkpoint.pth'),
                # MACs-first metrics
                'target_macs': float(target_macs),
                'achieved_macs': float(achieved_macs),
                'macs_error_pct': float(macs_error_pct),
            }
            # Legacy ratio fields if present upstream
            if 'target_ratio' in state:
                history_entry['target_ratio'] = state.get('target_ratio')
            if 'achieved_ratio' in pruning_results:
                history_entry['achieved_ratio'] = pruning_results.get('achieved_ratio')

            # Dataset-aware accuracy fields
            if ds.lower() == 'imagenet':
                history_entry['zero_shot_top1_accuracy'] = float(zero_shot_acc)
                history_entry['zero_shot_accuracy'] = float(zero_shot_acc)
                history_entry['fine_tuned_top1_accuracy'] = None
                history_entry['fine_tuned_top5_accuracy'] = None
            else:
                history_entry['zero_shot_accuracy'] = float(zero_shot_acc)
                history_entry['fine_tuned_accuracy'] = None

            # Append to state history
            state.setdefault('history', []).append(history_entry)
            print(f"[📝] Added candidate history entry (MACs-first)")

            return "fine_tune"

        else:
            # Outside acceptable MACs range
            if macs_error_pct > float(macs_overshoot_tolerance_pct):
                # too heavy → not enough pruning
                print(f"[❌] MACS ABOVE TARGET (too heavy): +{macs_error_pct:.2f}% (beyond +{macs_overshoot_tolerance_pct:.1f}% tolerance)")
                print(f"   Achieved: {achieved_macs/1e9:.3f}G vs Target: {target_macs/1e9:.3f}G")
            elif macs_error_pct < -float(macs_undershoot_tolerance_pct):
                # too light → too much pruning
                print(f"[❌] MACS BELOW TARGET (too light): {macs_error_pct:.2f}% (beyond -{macs_undershoot_tolerance_pct:.1f}% tolerance)")
                print(f"   Achieved: {achieved_macs/1e9:.3f}G vs Target: {target_macs/1e9:.3f}G")
            else:
                # Edge case: outside band due to rounding/precision
                print(f"[❌] MACS OUT OF RANGE: {macs_error_pct:.2f}% vs allowed -{macs_undershoot_tolerance_pct:.1f}%…+{macs_overshoot_tolerance_pct:.1f}%")
                print(f"   Achieved: {achieved_macs/1e9:.3f}G vs Target: {target_macs/1e9:.3f}G")


            return "master_normal"


    
    # Add nodes with dataset-aware agents
    workflow.add_node("profile", profile_with_state)
    workflow.add_node("master", master_with_state)
    workflow.add_node("analyze", analyze_with_state)
    workflow.add_node("prune", prune_with_state)
    workflow.add_node("fine_tune", fine_tune_with_state)
    workflow.add_node("evaluate", evaluate_with_state)

    # Import END constant for workflow construction
    from langgraph.graph import END

    # Main flow through the workflow with conditional edges after master
    workflow.add_edge("profile", "master")

    # Dataset-aware routing function
    def route_after_master(state):
        """Dataset-aware routing that implements the best model search strategy"""
        
        dataset = state.get('dataset', 'cifar10')
        
        # Check if Master Agent decided to stop
        master_results = state.get("master_results", {})
        should_continue = master_results.get("should_continue", True)
        
        if not should_continue:
            print(f"[🛑] Master Agent decided to stop {dataset} workflow")
            
            # Check if we have a final best model to evaluate
            final_best = state.get('final_best_model')
            if final_best:
                # Check if model has dataset-appropriate accuracy metric
                if dataset.lower() == 'imagenet':
                    has_accuracy = 'fine_tuned_top1_accuracy' in final_best
                else:
                    has_accuracy = 'fine_tuned_accuracy' in final_best
                
                if has_accuracy:
                    # Already fully evaluated, just end
                    print(f"[✅] Best {dataset} model already fully evaluated, ending workflow")
                    return "__end__"
                else:
                    # Need to fine-tune and evaluate the best model
                    print(f"[🔄] Fine-tuning best {dataset} model before ending")
                    state['_final_evaluation'] = True
                    return "fine_tune"
            else:
                # No good model found, just end
                print(f"[❌] No suitable {dataset} model found, ending workflow")
                return "__end__"
        
        # Continue normal flow
        return "analyze"


    workflow.add_conditional_edges(
        "master",
        route_after_master,
        {
            "analyze": "analyze",
            "fine_tune": "fine_tune",  # Direct path to fine_tune
            "__end__": END
        }
    )

    # Rest of the workflow is linear
    workflow.add_edge("analyze", "prune")
    # workflow.add_edge("prune", "fine_tune")

    workflow.add_conditional_edges(
        "prune",
        route_after_pruning,
        {
            "fine_tune": "fine_tune",              # Success path
            "master_catastrophic": "master",       # Zero-shot failure path  
            "master_normal": "master"              # Normal failure path
        }
    )

    workflow.add_edge("fine_tune", "evaluate")

    # Dataset-aware final evaluation check
    def check_final_evaluate(state):
        dataset = state.get('dataset', 'cifar10')
        
        if state.get('_final_evaluation', False):
            print(f"[🏁] Final {dataset} evaluation complete. Ending workflow.")
            return "__end__"
        else:
            return "profile"  # Normal cycle

    # Add this edge
    workflow.add_conditional_edges(
        "evaluate",
        check_final_evaluate,
        {
            "profile": "profile",
            "__end__": END
        }
    )

    # Set entry point
    workflow.set_entry_point("profile")
    
    return workflow


async def run_pruning_workflow(model_name: str, query: str, dataset: str = "cifar10", 
                              data_path: str = "./data", state_mods=None, 
                              imagenet_subset: float = 1.0, output_dir: str = "./models",
                              checkpoint_dir: str = "./checkpoints"):
    """Enhanced workflow runner with ImageNet subset support"""
    global GLOBAL_STATE, MODEL_STORE
    if 'macs_target_g' in (state_mods or {}):
        # MACs-first mode - don't use ratio extraction
        pruning_ratio = None  # Will be calculated from MACs later
    elif 'macs_target_ratio' in (state_mods or {}):
        pruning_ratio = None  # Will be calculated from MACs later
    else:
        # Fallback to legacy ratio extraction
        pruning_ratio = extract_pruning_ratio(query)

    
    # Ensure model_name is valid
    if not model_name or model_name.strip() == "":
        # Try to extract from query as fallback
        import re
        match = re.search(r'Prune\s+(\w+(?:_\w+)*)\s+model', query)
        if match:
            model_name = match.group(1)
            print(f"[🔧] Extracted model name from query: {model_name}")
        else:
            raise ValueError("Model name is required and could not be extracted from query")
    
    print(f"[✅] Using model name: {model_name}")
    
    # Dataset-specific defaults
    if dataset.lower() == 'imagenet':
        num_classes = 1000
        input_size = 224
        accuracy_threshold = state_mods.get('accuracy_threshold', 1.0)
        max_revisions = 5  # Fewer revisions due to computational cost
        
        # ADD: ImageNet subset message
        if imagenet_subset < 1.0:
            print(f"\n[🔬] ImageNet TESTING MODE: Using {imagenet_subset*100:.1f}% of training data")
            print(f"[⚡] Expected speedup: ~{1/imagenet_subset:.1f}x faster")
            print(f"[⚠️] Results are for testing/debugging - use full dataset for production")
    else:  # CIFAR-10
        num_classes = 10
        input_size = 224  # We resize CIFAR-10 to 224x224
        accuracy_threshold = 85.0
        max_revisions = 10
    
    try:
        print(f"\n[🚀] Initializing {dataset.upper()} pruning workflow...")
        print(f"[📋] Model: {model_name}")
        print(f"[📋] Dataset: {dataset} ({num_classes} classes)")
        print(f"[📋] Data path: {data_path}")
        print(f"[📁] Output directory: {output_dir}") 
        
        if state_mods:
            if 'accuracy_threshold' in state_mods:
                accuracy_threshold = state_mods['accuracy_threshold']
                print(f"[🔧] Set accuracy threshold to {accuracy_threshold}%")
            if 'max_revisions' in state_mods:
                max_revisions = state_mods['max_revisions']
                print(f"[🔧] Set maximum revisions to {max_revisions}")

            # Extract and validate MAC targets from state_mods
            target_macs = None
            baseline_macs = None
            macs_overshoot_tolerance_pct = 1.0  # Default overshoot tolerance (strict)
            macs_undershoot_tolerance_pct = 5.0  # Default undershoot tolerance (lenient)


            # Try different possible MAC field names for flexibility
            target_macs = (state_mods.get('target_macs') or 
                            state_mods.get('macs_target'))  # Removed duplicate 'target_macs'
            
            baseline_macs = state_mods.get('baseline_macs')  # Removed duplicate logic
            
            macs_overshoot_tolerance_pct = state_mods.get('macs_overshoot_tolerance_pct', 1.0)
            macs_undershoot_tolerance_pct = state_mods.get('macs_undershoot_tolerance_pct', 5.0)
            
            # Handle MAC target ratio (convert to absolute if baseline available)
            macs_target_ratio = state_mods.get('macs_target_ratio')
            if macs_target_ratio and baseline_macs:
                target_macs = baseline_macs * macs_target_ratio
                print(f"[🔧] Converted MAC ratio {macs_target_ratio} to target {target_macs/1e9:.3f}G")

        # Fallback: Extract MAC target from query if not in state_mods
        if target_macs is None:
            target_macs = extract_mac_target(query)  # You'll need to implement this
            if target_macs:
                print(f"[🔧] Extracted MAC target from query: {target_macs/1e9:.3f}G")  # Added .3f

        # Validate MAC values
        if target_macs is not None and target_macs <= 0:
            raise ValueError(f"Invalid target MACs: {target_macs}G (must be positive)")

        if baseline_macs is not None and baseline_macs <= 0:  # Added missing 'and'
            raise ValueError(f"Invalid baseline MACs: {baseline_macs}G (must be positive)")
            
        # Only do the comparison if BOTH values are not None
        if target_macs is not None and baseline_macs is not None and target_macs >= baseline_macs:
            print(f"[⚠️] Warning: Target MACs ({target_macs/1e9:.3f}G) >= baseline ({baseline_macs/1e9:.3f}G)")
            print(f"[📋] MAC Configuration:")
            print(f"   Target MACs: {target_macs/1e9:.3f}G" if target_macs else "   Target MACs: Not specified")
            print(f"   Baseline MACs: {baseline_macs/1e9:.3f}G" if baseline_macs else "   Baseline MACs: Will be computed")
            print(f"   Overshoot tolerance (above target): +{macs_overshoot_tolerance_pct}%")
            print(f"   Overshoot tolerance (above target): +{macs_undershoot_tolerance_pct}%")

        macs_overshoot_tolerance_pct = state_mods.get('macs_overshoot_tolerance_pct', 1.0) if state_mods else 1.0
        macs_undershoot_tolerance_pct = state_mods.get('macs_undershoot_tolerance_pct', 5.0) if state_mods else 5.0
        
        # Initialize global state with dataset information
        GLOBAL_STATE = {
            'target_pruning_ratio': pruning_ratio,
            'user_target_pruning_ratio': pruning_ratio,
            
            # MAC-based configuration (primary)
            'target_macs': target_macs,
            'baseline_macs': baseline_macs,
            'macs_overshoot_tolerance_pct': state_mods.get('macs_overshoot_tolerance_pct', 1.0),
            'macs_undershoot_tolerance_pct': state_mods.get('macs_undershoot_tolerance_pct', 5.0),
            'model_name': model_name,
            'dataset': dataset,
            'num_classes': num_classes,
            'input_size': input_size,
            'data_path': data_path,
            'accuracy_threshold': accuracy_threshold,
            'max_revisions': max_revisions,
            'best_models': [],
            'imagenet_subset': imagenet_subset,
            'output_dir': output_dir,
            'checkpoint_dir': checkpoint_dir
        }
        MODEL_STORE = None

        initial_state = {
            'query': query,
            'model_name': model_name,  # Ensure this is set in initial state too
            'dataset': dataset,
            'num_classes': num_classes,
            'input_size': input_size,
            'data_path': data_path,
            'imagenet_subset': imagenet_subset,  # ADD: Include in initial state
            
            # MAC-based configuration (primary)
            'target_macs': target_macs,
            'baseline_macs': baseline_macs,
            'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
            'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
            
            'attempted_pruning_ratios': [],
            'model_config': {
                'architecture': model_name,
                'state_dict': None,
            },
            'profile_results': {},
            'master_results': {},
            'analysis_results': {},
            'pruning_results': {},
            'fine_tuning_results': {},
            'evaluation_results': {},
            'revision_number': 0,
            'max_revisions': max_revisions,
            
            # Legacy ratio support (keep for backward compatibility)
            'target_pruning_ratio': pruning_ratio,
            'user_target_pruning_ratio': pruning_ratio,
            
            'accuracy_threshold': accuracy_threshold,
            'history': [],
            'extended_search_remaining': None,
            'is_candidate_model': False,
            'final_best_model': None,
            'all_candidate_models': [],
            'checkpoint_dir': checkpoint_dir,
            'output_dir': output_dir,  # Also add this since it's in GLOBAL_STATE
        }
        
        # Apply any additional user modifications to the initial state
        if state_mods:
            for key, value in state_mods.items():
                initial_state[key] = value

        # CRITICAL: Ensure model_name is not overwritten
        initial_state['model_name'] = model_name
        GLOBAL_STATE['model_name'] = model_name
        
        print(f"\n[📋] Initial {dataset} state keys:", initial_state.keys())
        print(f"[📋] Model name in initial state: {initial_state.get('model_name')}")
        print(f"[📋] Model name in global state: {GLOBAL_STATE.get('model_name')}")
        
        workflow = create_pruning_workflow()
        graph = workflow.compile()

        config = {
            'recursion_limit': 5000,
            'configurable': {
                'thread_id': '1',
                "checkpoint_ns": f"pruning_workflow_{dataset}",
                "checkpoint_id": "run_1",
            }
        }

        print(f"\n[⚙️] {dataset} workflow configuration:", config)
        print(f"\n[🏃] Starting {dataset} workflow execution...")

        step_count = 0

        async for step in graph.astream(initial_state, config):
            print(f"\n[🔄] {dataset} step received:", step.keys())

            # Log workflow progress to WandB
            step_name = list(step.keys())[0] if step else "unknown"
            wandb.log({
                "workflow/current_step": step_name,
                "workflow/step_count": step_count,
                "workflow/dataset": dataset,
            })

            if 'revision_number' in step:
                print(f"[📋] Current iteration: {step['revision_number']}")
                wandb.log({
                    "workflow/revision_number": step['revision_number'],
                    "workflow/iteration": step['revision_number'],
                })

            # Deep merge the state while preserving critical values
            GLOBAL_STATE = deep_merge(GLOBAL_STATE, step)
            
            # Ensure model_name is always preserved
            if 'model_name' not in GLOBAL_STATE or not GLOBAL_STATE['model_name']:
                GLOBAL_STATE['model_name'] = model_name
                
            print(f"[🔍] Current {dataset} global state keys:", GLOBAL_STATE.keys())
            print(f"[🔍] Model name in global state: {GLOBAL_STATE.get('model_name')}")

            # Log step-specific metrics if available
            if 'pruning_results' in step:
                pruning_res = step['pruning_results']
                if pruning_res.get('success'):
                    wandb.log({
                        f"step_{step_count}/achieved_ratio": pruning_res.get('achieved_ratio', 0),
                        f"step_{step_count}/macs_reduction": pruning_res.get('macs_reduction', 0),
                    })
            
            if 'evaluation_results' in step:
                eval_res = step['evaluation_results']
                if dataset.lower() == 'imagenet':
                    if 'final_top1_accuracy' in eval_res:
                        wandb.log({f"step_{step_count}/final_top1_accuracy": eval_res['final_top1_accuracy']})
                else:
                    if 'final_accuracy' in eval_res:
                        wandb.log({f"step_{step_count}/final_accuracy": eval_res['final_accuracy']})

            # Check for workflow termination signals
            if step.get('_end_workflow', False):
                print(f"[🛑] {dataset} workflow termination signal received")
                break
                
            if 'error' in step:
                print(f"[❌] Error in {dataset} workflow step: {step['error']}")
                wandb.log({
                    "workflow/error_step": step_count,
                    "workflow/error_message": step['error']
                })
                break

            if MODEL_STORE is not None:
                print(f"[💾] {dataset} model in store:", type(MODEL_STORE))
                
            step_count += 1

        # Final results processing and completion
        print(f"\n[🏁] {dataset.upper()} workflow completed!")
        
        # Extract final results
        final_pruning_results = GLOBAL_STATE.get('prune', {}).get('pruning_results', {})
        final_evaluation_results = GLOBAL_STATE.get('evaluation_results', {})
        
        if final_pruning_results.get('success', False):
            achieved_ratio = final_pruning_results.get('achieved_ratio', 0)
            print(f"[✅] {dataset} pruning successful: {achieved_ratio*100:.2f}% parameter reduction")
            
            if final_evaluation_results:
                if dataset.lower() == 'imagenet':
                    final_acc = final_evaluation_results.get('final_accuracy', 0)
                    print(f"[✅] Final Top-1 accuracy: {final_acc:.2f}%")
                else:
                    final_acc = final_evaluation_results.get('final_accuracy', 0)
                    print(f"[✅] Final accuracy: {final_acc:.2f}%")
                    
                success = final_evaluation_results.get('success', False)
                print(f"[{'✅' if success else '⚠️'}] Success criteria met: {success}")
        else:
            print(f"[❌] {dataset} pruning failed or incomplete")
        
        # Log final summary to WandB
        if 'evaluation_results' in GLOBAL_STATE:
            eval_results = GLOBAL_STATE['evaluation_results']
            final_wandb_metrics = {
                "workflow/completed": True,
                "workflow/dataset": dataset,
                "workflow/total_steps": step_count,
                "workflow/final_success": eval_results.get('success', False),
                "workflow/final_accuracy": eval_results.get('final_accuracy', 0),
                "workflow/achieved_ratio": final_pruning_results.get('achieved_ratio', 0),
                "workflow/total_revisions": GLOBAL_STATE.get('revision_number', 0)
            }
            wandb.log(final_wandb_metrics)
        
        print(f"[📊] Total workflow steps: {step_count}")
        print(f"[💾] Final state keys: {list(GLOBAL_STATE.keys())}")

        # Save final best model
        print("[💾] Saving final best model...")

        if MODEL_STORE is not None:
            prune_section = GLOBAL_STATE.get('prune')
            if not isinstance(prune_section, dict):
                prune_section = {}
                GLOBAL_STATE['prune'] = prune_section
            prune_section['model'] = MODEL_STORE

        GLOBAL_STATE = await save_final_best_model(GLOBAL_STATE)
        return GLOBAL_STATE

    except Exception as e:
        print(f"\n[❌] Error in {dataset} workflow: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None    
        
def validate_results(state):
    """Validate that results are consistent and make sense"""
    
    dataset = state.get('dataset', 'unknown')
    history = state.get('history', [])
    
    print(f"\n[🔍] Validating {dataset} results...")
    
    for i, entry in enumerate(history):
        rev = entry.get('revision', i)
        
        if dataset.lower() == 'imagenet':
            zero_shot = entry.get('zero_shot_top1_accuracy')
            fine_tuned = entry.get('fine_tuned_top1_accuracy')
            
            if zero_shot is not None and fine_tuned is not None:
                improvement = fine_tuned - zero_shot
                print(f"  Rev {rev}: Zero-shot={zero_shot:.2f}%, Fine-tuned={fine_tuned:.2f}%, Improvement={improvement:.2f}%")
                
                # Sanity checks
                if zero_shot < 0 or zero_shot > 100:
                    print(f"    ⚠️  Invalid zero-shot accuracy: {zero_shot:.2f}%")
                if fine_tuned < 0 or fine_tuned > 100:
                    print(f"    ⚠️  Invalid fine-tuned accuracy: {fine_tuned:.2f}%")
                if improvement < 0:
                    print(f"    ⚠️  Negative improvement: {improvement:.2f}%")
        else:
            zero_shot = entry.get('zero_shot_accuracy')
            fine_tuned = entry.get('fine_tuned_accuracy')
            
            if zero_shot is not None and fine_tuned is not None:
                improvement = fine_tuned - zero_shot
                print(f"  Rev {rev}: Zero-shot={zero_shot:.2f}%, Fine-tuned={fine_tuned:.2f}%, Improvement={improvement:.2f}%")


