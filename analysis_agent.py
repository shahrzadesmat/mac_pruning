from __future__ import annotations
from typing import Any, Dict
import traceback
import random

from langchain_core.messages import SystemMessage, HumanMessage

from llm.provider import get_llm
from llm.enhancer import DatasetAwarePromptEnhancer
from llm.prompts import (
    get_cnn_analysis_content,
    get_vit_analysis_content,
    format_vit_history_analysis,
    format_analysis_prompt,
    _format_analysis_for_llm,
)
import copy
from utils.timing import time_it, time_it_async
from utils.analysis_dependency import DependencyAnalyzer
from utils.analysis_isomorphism import ViTIsomorphicAnalyzer
from utils.analysis_vit import ViTLearningAnalyzer
from utils.analysis_cnn import CNNLearningAnalyzer
from utils.analysis_structures import PruningState
from utils.pruning_safety import PruningSafetyValidator
from utils.json_utils import parse_llm_json_response
from utils.analysis_vit import ViTLearningAnalyzer
from utils.pruning_safety import PruningSafetyValidator



class AnalysisAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.safety_validator = PruningSafetyValidator()
        self.prompt_enhancer = DatasetAwarePromptEnhancer()

    def _get_high_accuracy_threshold(self, dataset: str) -> float:
        """Return accuracy threshold for catastrophic failure detection"""
        if dataset.lower() == "imagenet":
            return 1.0
        elif dataset.lower() in {"cifar10", "cifar-10"}:
            return 85.0
        return 90.0

    def _validate_against_complete_signatures(self, strategy_dict, avoid_complete_signatures, tolerance=0.05):
        """Validate that proposed strategy doesn't repeat complete failed signatures"""
        proposed_importance = strategy_dict.get('importance_criterion')
        proposed_round_to = strategy_dict.get('round_to')
        proposed_ratios = strategy_dict.get('isomorphic_group_ratios', {})
        proposed_mlp = proposed_ratios.get('mlp_multiplier')
        proposed_qkv = proposed_ratios.get('qkv_multiplier')
        
        if not (proposed_mlp and proposed_qkv):
            return True, "No multipliers to validate"
        
        for signature in avoid_complete_signatures:
            if (signature['importance_criterion'] == proposed_importance and
                signature['round_to'] == proposed_round_to and
                signature['mlp_multiplier'] and signature['qkv_multiplier']):
                
                mlp_diff = abs(signature['mlp_multiplier'] - proposed_mlp)
                qkv_diff = abs(signature['qkv_multiplier'] - proposed_qkv)
                
                if mlp_diff < tolerance and qkv_diff < tolerance:
                    mac_error = signature.get('mac_error_pct', 0)
                    error_type = "overshoot" if mac_error > 0 else "undershoot"
                    return False, f"Too similar to failed config: mlp={signature['mlp_multiplier']:.3f}, qkv={signature['qkv_multiplier']:.3f} (caused {mac_error:+.1f}% MAC {error_type})"
        
        return True, "Configuration appears safe"

    def _force_safe_complete_strategy(self, strategy_dict, avoid_complete_signatures, target_ratio, dataset):
        """Generate a strategy that avoids ALL failed complete signatures"""
        
        if dataset.lower() == 'imagenet':
            safe_combinations = [
                ('taylor', 1, 0.25, 0.15), ('taylor', 2, 0.3, 0.2), ('taylor', 4, 0.25, 0.15), ('taylor', 8, 0.2, 0.1), ('taylor', 16, 0.15, 0.1),
                ('l1norm', 1, 0.3, 0.2), ('l1norm', 2, 0.35, 0.25), ('l1norm', 4, 0.3, 0.2), ('l1norm', 8, 0.25, 0.15), ('l1norm', 16, 0.2, 0.1),
                ('l2norm', 1, 0.25, 0.2), ('l2norm', 2, 0.3, 0.2), ('l2norm', 4, 0.3, 0.2), ('l2norm', 8, 0.25, 0.15), ('l2norm', 16, 0.2, 0.1)
            ]
        else:
            safe_combinations = [
                ('taylor', 1, 0.4, 0.2), ('taylor', 2, 0.45, 0.25), ('taylor', 4, 0.4, 0.25), ('taylor', 8, 0.35, 0.2), ('taylor', 16, 0.3, 0.15),
                ('l1norm', 1, 0.5, 0.3), ('l1norm', 2, 0.5, 0.3), ('l1norm', 4, 0.45, 0.25), ('l1norm', 8, 0.4, 0.2), ('l1norm', 16, 0.35, 0.2),
                ('l2norm', 1, 0.45, 0.25), ('l2norm', 2, 0.45, 0.3), ('l2norm', 4, 0.4, 0.25), ('l2norm', 8, 0.35, 0.2), ('l2norm', 16, 0.3, 0.15)
            ]
        
        for importance, round_to, mlp_mult, qkv_mult in safe_combinations:
            # Check if this combination is safe
            is_safe = True
            for signature in avoid_complete_signatures:
                if (signature['importance_criterion'] == importance and
                    signature['round_to'] == round_to and
                    signature['mlp_multiplier'] and signature['qkv_multiplier']):
                    
                    mlp_diff = abs(signature['mlp_multiplier'] - mlp_mult)
                    qkv_diff = abs(signature['qkv_multiplier'] - qkv_mult)
                    
                    if mlp_diff < 0.1 and qkv_diff < 0.1:  # Too close to failed config
                        is_safe = False
                        break
            
            if is_safe:
                print(f"[✅] Found safe complete strategy: {importance}, {round_to}, mlp={mlp_mult}, qkv={qkv_mult}")
                strategy_dict['importance_criterion'] = importance
                strategy_dict['round_to'] = round_to
                strategy_dict['isomorphic_group_ratios']['mlp_multiplier'] = mlp_mult
                strategy_dict['isomorphic_group_ratios']['qkv_multiplier'] = qkv_mult
                strategy_dict['safety_override'] = 'complete_signature_avoidance'
                return strategy_dict
        
        # If no safe combination found, use very conservative defaults
        print(f"[🚨] No safe complete strategy found, using ultra-conservative defaults")
        strategy_dict['importance_criterion'] = 'taylor'
        strategy_dict['round_to'] = 2
        strategy_dict['isomorphic_group_ratios']['mlp_multiplier'] = 0.2
        strategy_dict['isomorphic_group_ratios']['qkv_multiplier'] = 0.1
        strategy_dict['emergency_fallback'] = True
        return strategy_dict

    def safe_format_float(self, value, precision=3, fallback="None"):
        """Safely format a float value that might be None"""
        if value is None:
            return fallback
        try:
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            return fallback

    def safe_format_any(self, value, fallback="None"):
        """Safely format any value that might be None"""
        return str(value) if value is not None else fallback
    
    @time_it_async("3. Enhanced Analysis Agent")
    async def analyze(self, state: PruningState) -> Dict:
        """CLEANED: Enhanced analysis with LLM-based approach only (MACs-aware)"""
        
        dataset = state.get('dataset', 'cifar10')
        model_name = state.get('model_name', 'unknown')

        sg = state.get('master_results', {}).get('strategic_guidance', {}) or {}
        baseline_macs = state.get('baseline_macs', sg.get('baseline_macs'))
        target_macs = state.get('target_macs', sg.get('target_macs'))

        target_ratio = state.get('target_pruning_ratio')
        if target_ratio is None and baseline_macs and target_macs:
            target_ratio = (baseline_macs - target_macs) / baseline_macs
            print(f"[🔧] Calculated target ratio from MACs: {target_ratio:.3f} ({target_ratio*100:.1f}%)")
        elif target_ratio is None:
            raise ValueError("No target_pruning_ratio provided and no MAC targets available to calculate ratio")

        macs_overshoot_tolerance_pct = float(state.get('macs_overshoot_tolerance_pct', sg.get('macs_overshoot_tolerance_pct', 1.0)))
        macs_undershoot_tolerance_pct = float(state.get('macs_undershoot_tolerance_pct', sg.get('macs_undershoot_tolerance_pct', 5.0)))

        # Detect architecture
        cnn_types = ['resnet', 'efficientnet', 'mobilenet', 'densenet', 'convnext', 'resnext']  # include resnext for consistency
        is_cnn = any(cnn_type in model_name.lower() for cnn_type in cnn_types)
        
        # print(f"[🧠] Enhanced Analysis for {dataset} {'CNN' if is_cnn else 'ViT'}")
        # if target_macs is not None:
        #     print(f"[⚙️] MACs-first context: target={target_macs/1e9:.3f}G, "
        #         f"baseline={'?.???' if baseline_macs is None else f'{baseline_macs/1e9:.3f}G'}, "
        #         f"tol=+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%")

        
        if is_cnn:
            # print(f"[🔄] Using CNN LLM-based historical learning")
            llm_response = await self._execute_cnn_analysis_with_history(state, model_name)
        else:
            # print(f"[🔄] Using ViT LLM-based analysis")
            # print(f"[🔄] Using ViT LLM-based historical learning")  # CHANGED!
            llm_response = await self._execute_vit_analysis_with_history(state, model_name)  # NEW!
        
        if is_cnn and 'channel_pruning_ratio' in llm_response:
            channel_ratio = llm_response['channel_pruning_ratio']
            print(f"[🤖] LLM selected channel ratio: {channel_ratio:.4f}")
            if (baseline_macs is not None) and (target_macs is not None):
                derived = max(0.0, min(1.0, 1 - (target_macs / baseline_macs)))
                print(f"[🎯] Derived reduction (display): {derived:.3f} ({derived*100:.1f}%)")
        
        # Use LLM strategy directly
        validated_strategy = llm_response.copy()
        validated_strategy['safety_validated'] = True
        validated_strategy['corrections_applied'] = []
        print(f"[✅] Using LLM strategy directly - no safety overrides")

        # MACs-first: ignore ratio as a control parameter; rely on MAC budget downstream
        # if (target_macs is not None) and (baseline_macs is not None):
            # validated_strategy.pop('channel_pruning_ratio', None)
            # validated_strategy.pop('pruning_ratio', None)


        # Propagate MACs context forward if LLM or state provided it
        # Prefer values from the LLM response, fallback to ones we read from state/SG
        target_macs_out = validated_strategy.get('target_macs', target_macs)
        baseline_macs_out = validated_strategy.get('baseline_macs', baseline_macs)
        macs_overshoot_tol_out = validated_strategy.get('macs_overshoot_tolerance_pct', macs_overshoot_tolerance_pct)
        macs_undershoot_tol_out = validated_strategy.get('macs_undershoot_tolerance_pct', macs_undershoot_tolerance_pct)

        extra_state = {}
        if target_macs_out is not None:
            # also provide raw ops for modules that expect absolute MACs
            extra_state['target_macs'] = float(target_macs_out)
            # extra_state['target_macs'] = int(target_macs_out)
            extra_state['macs_overshoot_tolerance_pct'] = float(macs_overshoot_tol_out)
            extra_state['macs_undershoot_tolerance_pct'] = float(macs_undershoot_tol_out)
        if baseline_macs_out is not None:
            extra_state['baseline_macs'] = float(baseline_macs_out)

        return {
            **extra_state,
            'analysis_results': {
                'strategy_dict': validated_strategy,
                'importance_criterion': validated_strategy.get('importance_criterion', 'taylor'),
                'channel_pruning_ratio': None if (target_macs is not None and baseline_macs is not None)
                                        else validated_strategy.get('channel_pruning_ratio'),
                'pruning_ratio': None if (target_macs is not None and baseline_macs is not None)
                                else validated_strategy.get('pruning_ratio', target_ratio),
                'round_to_value': validated_strategy.get('round_to'),
                'isomorphic_group_ratios': validated_strategy.get('isomorphic_group_ratios', {}),
                'safety_validated': True,
                'dataset': dataset,
                'architecture_type': 'cnn' if is_cnn else 'vit',
                'llm_driven': True,
                'baseline_macs': baseline_macs_out,
                'target_macs': target_macs_out,
                'macs_overshoot_tolerance_pct': macs_overshoot_tol_out,
                'macs_undershoot_tolerance_pct': macs_undershoot_tol_out,
            }
        }


    def _generate_bidirectional_strategic_guidance(self, history, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """Generate direction-aware strategic guidance based on MAC failure patterns"""
        
        if not history:
            return ""
        
        # print(f"[🔍] Analyzing {len(history)} history entries for bidirectional guidance")

        # Analyze MAC failure patterns
        mac_overshoot_count = 0
        mac_undershoot_count = 0
        dangerous_overshoot_zones = []
        dangerous_undershoot_zones = []
        
        for entry in history:
            achieved_macs = entry.get('achieved_macs', 0)
            target_mac_entry = entry.get('target_macs', target_macs)
            strategy = entry.get('strategy_used', {})
            ratios = strategy.get('isomorphic_group_ratios', {})
            
            if achieved_macs and target_mac_entry:
                mac_error_pct = (achieved_macs - target_mac_entry) / target_mac_entry * 100
                mlp_mult = ratios.get('mlp_multiplier', 0)
                qkv_mult = ratios.get('qkv_multiplier', 0)
                
                if mac_error_pct > macs_overshoot_tolerance_pct:  # MAC overshoot (insufficient pruning)
                    mac_overshoot_count += 1
                    if mlp_mult and qkv_mult:  # Only add if we have multiplier data
                        dangerous_overshoot_zones.append({
                            'mlp_range': (max(0, mlp_mult - 0.1), mlp_mult + 0.1),
                            'qkv_range': (max(0, qkv_mult - 0.1), qkv_mult + 0.1),
                            'mac_error_pct': mac_error_pct,
                            'reason': f'Caused +{mac_error_pct:.1f}% MAC overshoot (exceeded +{macs_overshoot_tolerance_pct:.1f}% limit)'
                        })
                elif mac_error_pct < -macs_undershoot_tolerance_pct:  # MAC undershoot (excessive pruning)
                    mac_undershoot_count += 1
                    if mlp_mult and qkv_mult:  # Only add if we have multiplier data
                        dangerous_undershoot_zones.append({
                            'mlp_range': (mlp_mult - 0.1, mlp_mult + 0.1),
                            'qkv_range': (qkv_mult - 0.1, qkv_mult + 0.1),
                            'mac_error_pct': mac_error_pct,
                            'reason': f'Caused {mac_error_pct:.1f}% MAC undershoot (exceeded -{macs_undershoot_tolerance_pct:.1f}% limit)'
                        })
        
        # Generate bidirectional guidance
        if mac_overshoot_count > mac_undershoot_count:
            guidance = f"""
        🚨 CRITICAL DIRECTION ANALYSIS - MAC OVERSHOOT DOMINANT:
        ...
        🚫 DANGEROUS UNDERPRUNING ZONES (AVOID THESE RANGES):
        """
            for i, zone in enumerate(dangerous_overshoot_zones[:5]):  # Show top 5
                mlp_min, mlp_max = zone['mlp_range']
                qkv_min, qkv_max = zone['qkv_range']
                guidance += f"""
        Zone {i+1}: mlp_multiplier {mlp_min:.3f}-{mlp_max:.3f}, qkv_multiplier {qkv_min:.3f}-{qkv_max:.3f}
        Reason: {zone['reason']}"""

        elif mac_undershoot_count > mac_overshoot_count:
            guidance = f"""
        🚨 CRITICAL DIRECTION ANALYSIS - MAC UNDERSHOOT DOMINANT:
        ...
        🚫 DANGEROUS OVERPRUNING ZONES (AVOID THESE RANGES):
        """
            for i, zone in enumerate(dangerous_undershoot_zones[:5]):  # Show top 5
                mlp_min, mlp_max = zone['mlp_range']
                qkv_min, qkv_max = zone['qkv_range']
                guidance += f"""
        Zone {i+1}: mlp_multiplier {mlp_min:.3f}-{mlp_max:.3f}, qkv_multiplier {qkv_min:.3f}-{qkv_max:.3f}
        Reason: {zone['reason']}"""

            
            else:
                guidance = f"""
        🚨 CRITICAL DIRECTION ANALYSIS - MIXED PATTERN:
        ==============================================
        - Mixed failure pattern: {mac_overshoot_count} overshoot, {mac_undershoot_count} undershoot
        - Previous multipliers show inconsistent results
        - SOLUTION: Use systematic approach with moderate multipliers
        - FOCUS: Find optimal balance between over/under-shooting attempts
        """

        return guidance

    async def _execute_cnn_analysis_with_history(self, state, model_name):
        """ENHANCED: Full utilization of CNNLearningAnalyzer capabilities"""
        
        # Extract context
        dataset = state.get('dataset', 'cifar10')
        target_ratio = state.get('target_pruning_ratio')
        history = state.get('history', [])
        current_revision = state.get('revision_number', 0)

        # Helper to normalize to GMacs
        def _to_g(val):
            if val is None:
                return None
            try:
                v = float(val)
            except Exception:
                return None
            return v / 1e9 if v > 1e6 else v

        # MAC context extraction
        baseline_macs_raw = (
            state.get('baseline_macs')
            or state.get('master_results', {}).get('strategic_guidance', {}).get('baseline_macs')
        )
        target_macs_raw = (
            state.get('target_macs')
            or state.get('master_results', {}).get('strategic_guidance', {}).get('target_macs')
        )
        macs_overshoot_tolerance_pct = float(
            state.get('macs_overshoot_tolerance_pct',
                    state.get('master_results', {}).get('strategic_guidance', {}).get('macs_overshoot_tolerance_pct', 1.0))
        )
        macs_undershoot_tolerance_pct = float(
            state.get('macs_undershoot_tolerance_pct',
                    state.get('master_results', {}).get('strategic_guidance', {}).get('macs_undershoot_tolerance_pct', 5.0))
        )

        baseline_macs = _to_g(baseline_macs_raw) if baseline_macs_raw else None
        target_macs = _to_g(target_macs_raw) if target_macs_raw else None

        # Calculate target_ratio from MACs if needed
        if target_ratio is None and baseline_macs and target_macs:
            target_ratio = (baseline_macs - target_macs) / baseline_macs
            print(f"[🔧] Calculated target ratio from MACs: {target_ratio:.3f} ({target_ratio*100:.1f}%)")
        elif target_ratio is None:
            raise ValueError("No target_pruning_ratio provided and no MAC targets available to calculate ratio")

        # REMOVE THESE DUPLICATE LINES:
        # macs_overshoot_tolerance_pct = float(state.get('macs_overshoot_tolerance_pct', 1.0))
        # macs_undershoot_tolerance_pct = float(state.get('macs_undershoot_tolerance_pct', 5.0))

        try:
            # 1. CREATE CNN LEARNING ANALYZER
            cnn_learning_analyzer = CNNLearningAnalyzer()

            # 2. USE COMPREHENSIVE ANALYSIS FIRST
            # print(f"[📊] Running comprehensive CNN channel pattern analysis...")
            comprehensive_analysis = cnn_learning_analyzer.analyze_cnn_channel_patterns(
                history, target_ratio, dataset,
                baseline_macs=baseline_macs,
                target_macs=target_macs,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
            )

            # 3. CHECK FOR LOCAL OPTIMUM
            local_optimum_trap = cnn_learning_analyzer._detect_cnn_local_optimum_trap(
                history, target_macs, dataset
            )
            
            if local_optimum_trap:
                print(f"[🕳️] CNN Local optimum detected - using escape strategy")
                escape_strategy = cnn_learning_analyzer._generate_cnn_escape_strategy(local_optimum_trap, target_macs)
                return await self._llm_calculate_cnn_strategy_with_escape_guidance(
                    target_ratio, model_name, history, dataset, escape_strategy
                )

            # 4. HANDLE STRATEGIC GUIDANCE
            strategic_guidance = state.get('master_results', {}).get('strategic_guidance')
            
            if strategic_guidance:
                # Inject MAC context and comprehensive analysis into strategic guidance
                enhanced_guidance = dict(strategic_guidance)
                enhanced_guidance.update({
                    'baseline_macs': baseline_macs,
                    'target_macs': target_macs,
                    'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                    'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
                    'comprehensive_analysis': comprehensive_analysis
                })
                
                guidance_type = strategic_guidance.get('guidance_type')
                
                if guidance_type == 'high_accuracy_preservation':
                    print(f"[🎯] HIGH-ACCURACY PRESERVATION: Using enhanced learning with guidance")
                    strategy = await self._llm_calculate_cnn_strategy_with_enhanced_learning(
                        target_ratio, model_name, history, dataset, enhanced_guidance
                    )
                    approach = "cnn_high_accuracy_with_enhanced_learning"
                    
                elif guidance_type == 'catastrophic_recovery':
                    print(f"[🚨] CATASTROPHIC RECOVERY: Using enhanced learning with guidance")
                    strategy = await self._llm_calculate_cnn_strategy_with_enhanced_learning(
                        target_ratio, model_name, history, dataset, enhanced_guidance
                    )
                    approach = "cnn_catastrophic_recovery_with_enhanced_learning"
                    
                else:
                    print(f"[📋] GENERAL GUIDANCE: Using enhanced learning with strategic guidance")
                    strategy = await self._llm_calculate_cnn_strategy_with_enhanced_learning(
                        target_ratio, model_name, history, dataset, enhanced_guidance
                    )
                    approach = "cnn_enhanced_learning_with_guidance"

            elif history:
                print(f"[📚] Using enhanced historical learning with comprehensive analysis")
                enhanced_guidance = {
                    'baseline_macs': baseline_macs,
                    'target_macs': target_macs,
                    'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                    'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
                    'comprehensive_analysis': comprehensive_analysis
                }
                strategy = await self._llm_calculate_cnn_strategy_with_enhanced_learning(
                    target_ratio, model_name, history, dataset, enhanced_guidance
                )
                approach = "cnn_enhanced_historical_learning_with_analysis"

            else:
                print(f"[🎯] First attempt - using enhanced baseline strategy")
                strategy = await self._llm_calculate_cnn_strategy_baseline(
                    target_ratio, model_name, dataset
                )
                approach = "cnn_enhanced_baseline"

            # 5. APPLY SAFETY VALIDATION WITH ERROR HANDLING
            try:
                safety_validator = PruningSafetyValidator()
                
                print(f"[🛡️] Applying safety validation to CNN strategy...")
                validated_strategy = safety_validator.validate_and_correct(
                    strategy, target_ratio, dataset, history,
                    macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
                )
            except Exception as safety_error:
                print(f"[⚠️] Safety validation failed: {safety_error}, using original strategy")
                validated_strategy = strategy

            # 6. CHECK FOR REPETITION
            already_tried, previous_entry = self._is_cnn_strategy_already_tried(validated_strategy, history, tolerance=0.02)
            if already_tried:
                print(f"[🔄] CNN strategy already tried, requesting LLM variation")
                validated_strategy = await self._llm_create_cnn_strategy_variation(
                    validated_strategy, previous_entry, target_ratio, model_name, dataset
                )
                approach += "_with_variation"

            # 7. ADD COMPREHENSIVE METADATA
            validated_strategy.update({
                "approach": approach,
                "llm_calculated": True,
                "fully_llm_driven": True,
                "comprehensive_analysis_used": True,
                "safety_validated": True,
                "enhanced_learning_applied": True,
                "revision": current_revision,
                "dataset": dataset,
                "model_name": model_name,
                "history_count": len(history),
                "baseline_macs": baseline_macs,
                "target_macs": target_macs,
                "macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
                "macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
            })

            print(f"[✅] Enhanced CNN strategy complete: {approach}")
            return validated_strategy

        except Exception as e:
            print(f"[❌] Enhanced CNN strategy generation failed: {e}")
            return self._create_emergency_cnn_fallback_strategy(target_ratio, model_name, dataset, str(e), state)


    async def _llm_calculate_cnn_strategy_with_enhanced_learning(
        self,
        target_ratio,
        model_name,
        history,
        dataset,
        strategic_guidance=None,
        ):
        """
        Enhanced CNN learning method using CNNLearningAnalyzer
        MACs-first: if baseline/target MACs are provided in strategic_guidance, steer the LLM to hit them within tolerance.
        """

        # Helper to normalize to GMacs (accept raw ops or G)
        def _to_g(val):
            if val is None:
                return None
            try:
                v = float(val)
            except Exception:
                return None
            return v / 1e9 if v > 1e6 else v

        round_to_instruction = """Use round_to=2 as the standard choice because:
        1. PRECISION: Finer granularity allows more precise MAC targeting
        2. ACCURACY PRESERVATION: Smaller pruning increments reduce risk of removing critical channel combinations
        3. FINE-GRAINED CONTROL: Enables hitting exact MAC budgets more accurately
        4. ITERATIVE REFINEMENT: Easier to fine-tune results with smaller steps"""


        # Pull optional MACs-first context
        baseline_macs = None
        target_macs = None
        macs_overshoot_tolerance_pct = 1.0
        macs_undershoot_tolerance_pct = 5.0
        if strategic_guidance:
            baseline_macs = _to_g(
                strategic_guidance.get("baseline_macs", strategic_guidance.get("baseline_macs"))
            )
            target_macs = _to_g(
                strategic_guidance.get("target_macs", strategic_guidance.get("target_macs"))
            )
            macs_overshoot_tolerance_pct = float(strategic_guidance.get("macs_overshoot_tolerance_pct", 1.0))
            macs_undershoot_tolerance_pct = float(strategic_guidance.get("macs_undershoot_tolerance_pct", 5.0))

        # Create the enhanced learning analyzer
        learning_analyzer = CNNLearningAnalyzer()

        print(f"[🧠] Enhanced CNN Analysis with Comprehensive Learning:")
        print(f"   Target (ratio fallback): {target_ratio:.3f}")
        print(f"   Model: {model_name}")
        print(f"   History entries: {len(history)}")
        print(f"   Dataset: {dataset}")

        baseline_str = f"{baseline_macs:.3f}G" if baseline_macs is not None else "N/A"
        target_str   = f"{target_macs:.3f}G"   if target_macs   is not None else "N/A"
        print(f"   MACs-first: baseline={baseline_str}, target={target_str}, tol=+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%")

        print(" macs_overshoot_tolerance_pct value in analysis_agent.py is:",  macs_overshoot_tolerance_pct)
        print(" macs_undershoot_tolerance_pct value in analysis_agent.py is:",  macs_undershoot_tolerance_pct)

        # Get comprehensive analysis and base prompt
        base_prompt = learning_analyzer.create_enhanced_cnn_learning_prompt(
            target_ratio, model_name, history, dataset, strategic_guidance, baseline_macs, 
            target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
        )

        # Prepend MACs-first header if available
        if target_macs is not None:
            macs_header = f"""MACs-FIRST OBJECTIVE
            =====================
            - Baseline MACs: {baseline_str}
            - Target  MACs:  {target_macs:.3f}G
            CRITICAL ROUND_TO INSTRUCTION: {round_to_instruction}
            - Overshoot tolerance (strict): +{macs_overshoot_tolerance_pct:.1f}%
            - Undershoot tolerance (lenient): -{macs_undershoot_tolerance_pct:.1f}%
            - Acceptable range: {target_macs * (1 - macs_undershoot_tolerance_pct / 100):.3f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.3f}G

            Your proposed channel_pruning_ratio MUST yield achieved MACs within this band.
            Heuristic: higher channel pruning → fewer MACs; lower channel pruning → more MACs.
            If you cannot compute MACs exactly, reason from history to select a ratio that will land within tolerance.
            """
            enhanced_prompt = macs_header + "\n" + base_prompt
        else:
            enhanced_prompt = base_prompt

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a CNN pruning expert with advanced learning capabilities who learns from historical patterns (MACs-first when provided)."),
                HumanMessage(content=enhanced_prompt),
            ])

            # print(f"[DEBUG] Enhanced CNN learning LLM response length: {len(response.content)}")

            strategy_dict = parse_llm_json_response(response.content)

            # Check if LLM provided channel_pruning_ratio
            if strategy_dict and "channel_pruning_ratio" in strategy_dict:
                channel_ratio = strategy_dict.get("channel_pruning_ratio", 0.0)
                confidence = strategy_dict.get("confidence_level", "unknown")

                print(f"[🧠] Enhanced CNN LLM learning results:")
                print(f"   Channel ratio: {channel_ratio:.3f}")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")
                print(f"   Confidence: {confidence}")
                print(f"   Learning applied: {strategy_dict.get('learning_applied', False)}")

                insights = strategy_dict.get("learning_insights", [])
                if insights:
                    print("   Key insights:")
                    for insight in insights:
                        print(f"     - {insight}")
                
                # Attach metadata for success path
                strategy_dict["enhanced_learning_used"] = True
                strategy_dict["historical_analysis_applied"] = True
                if target_macs is not None:
                    strategy_dict["baseline_macs"] = baseline_macs
                    strategy_dict["target_macs"] = target_macs
                    strategy_dict["macs_overshoot_tolerance_pct"] = macs_overshoot_tolerance_pct
                    strategy_dict["macs_undershoot_tolerance_pct"] = macs_undershoot_tolerance_pct
                    strategy_dict.setdefault("calculation_details", {})["mode"] = "macs_first"
                
                return strategy_dict
                
            elif strategy_dict:
                # LLM didn't provide channel_pruning_ratio, but we have other data
                print(f"[⚠️] LLM response missing channel_pruning_ratio, calculating from MAC constraints")
                
                # Calculate channel ratio from MAC-derived target ratio
                if target_macs is not None and baseline_macs is not None:
                    mac_derived_ratio = (baseline_macs - target_macs) / baseline_macs
                    channel_ratio = mac_derived_ratio  # Use the full MAC-derived ratio
                    strategy_dict["channel_pruning_ratio"] = channel_ratio
                    print(f"[🔧] Calculated channel ratio from MACs: {channel_ratio:.3f}")
                else:
                    # Final fallback
                    channel_ratio = 0.3
                    strategy_dict["channel_pruning_ratio"] = channel_ratio
                    print(f"[⚠️] Using conservative fallback channel ratio: {channel_ratio:.3f}")
                
                print(f"[🧠] Enhanced CNN LLM learning results:")
                print(f"   Channel ratio: {channel_ratio:.3f} (calculated)")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")

                # Attach metadata for fallback path
                strategy_dict["enhanced_learning_used"] = True
                strategy_dict["historical_analysis_applied"] = True
                if target_macs is not None:
                    strategy_dict["baseline_macs"] = baseline_macs
                    strategy_dict["target_macs"] = target_macs
                    strategy_dict["macs_overshoot_tolerance_pct"] = macs_overshoot_tolerance_pct
                    strategy_dict["macs_undershoot_tolerance_pct"] = macs_undershoot_tolerance_pct
                    strategy_dict.setdefault("calculation_details", {})["mode"] = "macs_first"
                
                return strategy_dict

            else:
                print(f"[❌] LLM response parsing failed completely")
                raise ValueError("Enhanced CNN learning strategy parsing failed")

        except Exception as e:
            print(f"[❌] Enhanced CNN learning LLM failed: {e}")
            print("[🔄] Falling back to standard CNN learning method")
            # Fallback to existing method (ratio-only)
            return await self._llm_calculate_full_strategy_with_history(
                target_ratio, model_name, history, dataset
            )

    # ADD: CNN escape guidance handler
    async def _llm_calculate_cnn_strategy_with_escape_guidance(self, target_ratio, model_name, history, dataset, escape_guidance):
        """
        Handle CNN escape from local optimum
        """
        excellent_baseline = escape_guidance['strategic_guidance']['excellent_baseline']
        exploration_directions = escape_guidance['strategic_guidance']['exploration_directions']
        forbidden_actions = escape_guidance['strategic_guidance']['forbidden_actions']
        
        # Extract MAC tolerance context if present
        macs_overshoot_tolerance_pct = escape_guidance['strategic_guidance'].get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = escape_guidance['strategic_guidance'].get('macs_undershoot_tolerance_pct', 5.0)

        prompt = f"""You are a CNN pruning expert helping escape a local optimum trap.

        SITUATION: Found excellent CNN result but small adjustments cause catastrophic failure.

        EXCELLENT BASELINE:
        - Accuracy: {excellent_baseline['accuracy']:.2f}%
        - Channel ratio: {excellent_baseline['working_channel_ratio']:.3f}
        - Achieved: {excellent_baseline['achieved_ratio']*100:.2f}% reduction

        ESCAPE STRATEGY: Try meaningfully different approaches, not micro-adjustments

        EXPLORATION DIRECTIONS:
        {chr(10).join(f"- {direction['direction']}: {direction['rationale']}" for direction in exploration_directions)}

        FORBIDDEN ACTIONS:
        {chr(10).join(f"- {action}" for action in forbidden_actions)}

        Calculate NEW channel ratio significantly different from {excellent_baseline['working_channel_ratio']:.3f}

        OUTPUT FORMAT (JSON only):
        {{
            "channel_pruning_ratio": YOUR_DIVERSIFIED_VALUE,
            "importance_criterion": "taylor|l1norm|l2norm", 
            "round_to": X,
            "global_pruning": true,
            "rationale": "ESCAPE STRATEGY: Avoiding local optimum by trying [direction] with channel_ratio=[value] because [reasoning]. This is meaningfully different from excellent baseline {excellent_baseline['working_channel_ratio']:.3f}.",
            "architecture_type": "cnn",
            "escape_strategy_applied": true
        }}"""

        # Execute LLM call for escape strategy
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a CNN expert escaping local optimum traps through diversified exploration."),
            HumanMessage(content=prompt)
        ])
        
        return parse_llm_json_response(response.content)

    # ADD: CNN strategy repetition checker
    def _is_cnn_strategy_already_tried(self, proposed_strategy, history, tolerance=0.02):
        """
        Check if proposed CNN strategy was already attempted
        """
        proposed_channel_ratio = proposed_strategy.get('channel_pruning_ratio')
        proposed_round_to = proposed_strategy.get('round_to')
        proposed_importance = proposed_strategy.get('importance_criterion')
        
        if not proposed_channel_ratio:
            return False, None
        
        for entry in history:
            strategy_used = entry.get('strategy_used', {})
            used_channel_ratio = (strategy_used.get('channel_ratio_used') or 
                                 strategy_used.get('channel_pruning_ratio') or  
                                 strategy_used.get('channel_ratio'))
            used_round_to = strategy_used.get('round_to')
            used_importance = strategy_used.get('importance_criterion')
            
            # Check if parameters are very similar
            if (used_channel_ratio and 
                abs(used_channel_ratio - proposed_channel_ratio) < tolerance and
                used_round_to == proposed_round_to and
                used_importance == proposed_importance):
                
                achieved = entry.get('achieved_ratio', 0)
                target = entry.get('target_ratio', 0)
                
                print(f"[🔄] CNN STRATEGY REPETITION DETECTED:")
                print(f"   Previous: channel={used_channel_ratio:.4f}, round_to={used_round_to}, importance={used_importance}")
                print(f"   Proposed: channel={proposed_channel_ratio:.4f}, round_to={proposed_round_to}, importance={proposed_importance}")
                if achieved is not None and target is not None:
                    print(f"   Previous result: {achieved*100:.1f}% (target: {target*100:.1f}%)")
                else:
                    print(f"   Previous result: achieved={achieved}, target={target}")
                return True, entry
        
        return False, None

    # ADD: CNN strategy variation creator
    async def _llm_create_cnn_strategy_variation(self, original_strategy, previous_entry, target_ratio, model_name, dataset):
        """
        Create intelligent CNN variation when strategy was already tried
        """
        previous_achieved = previous_entry.get('achieved_ratio', 0)
        original_channel = original_strategy.get('channel_pruning_ratio', 0.3)
        
        prompt = f"""You are a CNN pruning expert creating a variation of a strategy that was already tried.

        SITUATION: The original CNN strategy was already attempted and needs adjustment.

        ORIGINAL STRATEGY:
        - Channel ratio: {original_channel:.4f}
        - Importance: {original_strategy.get('importance_criterion', 'unknown')}
        - Round-to: {original_strategy.get('round_to', 'unknown')}

        PREVIOUS RESULT:
        - Achieved: {previous_achieved*100:.1f}% parameter reduction
        - Target: {target_ratio*100:.1f}% parameter reduction

        TASK: Create a modified CNN strategy that will get closer to the {target_ratio*100:.1f}% target.

        ANALYSIS:
        - If previous overshoot: Reduce channel ratio
        - If previous undershoot: Increase channel ratio  
        - Consider changing importance criterion or round_to if needed
        - Make smart adjustments based on {model_name} characteristics

        OUTPUT FORMAT (JSON only):
        {{
            "channel_pruning_ratio": YOUR_ADJUSTED_VALUE,
            "importance_criterion": "taylor|l1norm|l2norm",
            "round_to": X,
            "global_pruning": true,
            "rationale": "CNN variation strategy: explain the adjustments made and why",
            "architecture_type": "cnn"
        }}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a CNN pruning expert creating intelligent strategy variations."),
                HumanMessage(content=prompt)
            ])
            
            strategy_dict = parse_llm_json_response(response.content)
            
            if strategy_dict and 'channel_pruning_ratio' in strategy_dict:
                channel_ratio = float(strategy_dict['channel_pruning_ratio'])
                
                print(f"[🤖] LLM CNN variation strategy:")
                print(f"   Original: {original_channel:.4f} → Variation: {channel_ratio:.4f}")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")
                
                return strategy_dict
            else:
                print(f"[⚠️] LLM CNN variation failed, using simple adjustment")
                return self._simple_cnn_strategy_variation(original_strategy, previous_achieved, target_ratio)
                
        except Exception as e:
            print(f"[⚠️] LLM CNN variation failed: {e}")
            return self._simple_cnn_strategy_variation(original_strategy, previous_achieved, target_ratio)

    # ADD: Simple CNN variation fallback
    def _simple_cnn_strategy_variation(self, original_strategy, previous_achieved, target_ratio):
        """Simple mathematical variation when LLM fails for CNN"""
        
        original_channel = original_strategy.get('channel_pruning_ratio', 0.3)
        
        if previous_achieved > target_ratio:
            # overshoot, reduce by 10%
            new_channel = original_channel * 0.9
            direction = "reduced (overshoot)"
        else:
            # undershoot, increase by 10%
            new_channel = original_channel * 1.1
            direction = "increased (undershoot)"
        
        new_channel = max(0.1, min(0.7, new_channel))
        
        varied_strategy = original_strategy.copy()
        varied_strategy['channel_pruning_ratio'] = new_channel
        varied_strategy['rationale'] = f"Simple CNN variation: {original_channel:.4f} → {new_channel:.4f} ({direction})"
        
        print(f"[🔧] Simple CNN variation: {original_channel:.4f} → {new_channel:.4f} ({direction})")
        
        return varied_strategy

    # ADD: CNN baseline strategy creator
    async def _llm_calculate_cnn_strategy_baseline(self, target_ratio, model_name, dataset):
        """
        Use LLM to determine baseline CNN strategy for first attempt
        """

        round_to_instruction = """Use round_to=2 as the standard choice because:
        1. PRECISION: Finer granularity allows more precise MAC targeting
        2. ACCURACY PRESERVATION: Smaller pruning increments reduce risk of removing critical channel combinations  
        3. FINE-GRAINED CONTROL: Enables hitting exact MAC budgets more accurately
        4. ITERATIVE REFINEMENT: Easier to fine-tune results with smaller steps"""

        
        prompt = f"""You are an expert in CNN pruning for {model_name} on {dataset}.

        RESEARCH-BASED IMPORTANCE CRITERION PRIORITY:
        Your system uses isomorphic pruning (global_pruning=true) which groups isomorphic structures.
        Combined with Taylor criterion, this achieves SOTA performance:
        - For CNNs: Isomorphic + Taylor shows >93% correlation with oracle ranking
        - For ViTs: Isomorphic + Taylor dramatically outperforms magnitude-based methods
        - Taylor uses gradient information vs. L1/L2 which rely on potentially biased weight distributions

        CRITICAL IMPORTANCE SELECTION:
        Use "taylor" as your first choice unless you have specific evidence from history that Taylor consistently failed.
        Taylor importance uses gradient information and generally provides superior accuracy preservation.
        Only deviate from Taylor if you can justify why based on the specific model and dataset combination.

        RECOMMENDED: Start with "taylor" as importance_criterion based on research evidence. Consider l1norm or l2norm only if historical analysis shows consistent Taylor underperformance for this specific model/dataset combination.


        CRITICAL ROUND_TO INSTRUCTION: {round_to_instruction}

        TASK: Determine the baseline CNN channel pruning strategy to achieve {target_ratio*100:.0f}% parameter reduction.

        This is the FIRST ATTEMPT - you have no historical data to learn from.
        Make your best educated guess based on CNN architecture knowledge.

        CNN ARCHITECTURE ANALYSIS FOR {model_name.upper()}:
        - ResNets: Have skip connections and bottleneck blocks that amplify pruning effects
        - ConvNets: Use standard convolutions that scale more linearly with channel pruning
        - Different architectures respond differently to channel pruning ratios

        DATASET REQUIREMENTS ({dataset.upper()}):
        - ImageNet: 1000 classes, complex features, need accuracy preservation
        - CIFAR-10: 10 classes, simpler features, can be more aggressive

        BASELINE STRATEGY GUIDELINES:
        - Start conservatively to avoid catastrophic accuracy loss
        - Channel ratio controls aggressiveness of pruning
        - Consider architecture-specific sensitivities

        OUTPUT FORMAT (JSON only):
        {{
            "importance_criterion": "taylor|l1norm|l2norm",
            "channel_pruning_ratio": YOUR_CONSERVATIVE_ESTIMATE,
            "round_to": 2,  // Use 2 unless historical analysis shows repeated failures
            "global_pruning": true,
            "rationale": "Baseline CNN strategy for {model_name} on {dataset}: Using conservative channel ratio to target {target_ratio*100:.0f}% reduction while preserving accuracy. Channel ratio chosen because [reasoning], importance chosen because [reasoning].",
            "architecture_type": "cnn"
        }}

        CRITICAL: This is a baseline estimate. Choose conservative channel ratio that is likely to achieve the target without catastrophic failure."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a CNN pruning expert making your first baseline strategy attempt."),
                HumanMessage(content=prompt)
            ])
            
            strategy_dict = parse_llm_json_response(response.content)
            
            if strategy_dict and 'channel_pruning_ratio' in strategy_dict:
                channel_ratio = strategy_dict.get('channel_pruning_ratio', 0)
                
                print(f"[🤖] LLM baseline CNN strategy for {model_name}:")
                print(f"   Channel ratio: {channel_ratio:.3f}")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")
                
                return strategy_dict
            else:
                print(f"[⚠️] LLM failed to provide valid baseline CNN strategy")
                raise ValueError("LLM baseline CNN strategy parsing failed")
                    
        except Exception as e:
            print(f"[❌] LLM baseline CNN strategy calculation failed: {e}")
            print(f"[🔄] Using hardcoded conservative fallback")
            
            # Conservative fallback when LLM completely fails
            if 'resnet' in model_name.lower():
                fallback_channel = target_ratio * 0.6
                fallback_importance = "taylor"
                fallback_round_to = 2 if dataset.lower() == 'imagenet' else 4
            elif 'convnext' in model_name.lower():
                fallback_channel = target_ratio * 0.5  # More conservative for ConvNext
                fallback_importance = "taylor"
                fallback_round_to = 2
            else:
                fallback_channel = target_ratio * 0.7
                fallback_importance = "taylor"
                fallback_round_to = 2
            
            return {
                "importance_criterion": fallback_importance,
                "channel_pruning_ratio": fallback_channel,
                "round_to": fallback_round_to,
                "global_pruning": True,
                "rationale": f"Conservative fallback for {model_name} on {dataset} due to LLM failure: {str(e)}",
                "architecture_type": "cnn",
                "fallback_used": True
            }

    # ADD: Emergency CNN fallback
    def _create_emergency_cnn_fallback_strategy(self, target_ratio, model_name, dataset, error_msg, state):
        """
        Emergency fallback when both CNN LLM methods fail
        """
        
        print(f"[🚨] Emergency CNN fallback for {model_name} on {dataset}")
        
        # Ultra-conservative approach based on model type
        if 'resnet' in model_name.lower():
            emergency_channel = target_ratio * 0.4  # Very conservative
            emergency_importance = "taylor"
            emergency_round_to = 2
        elif 'convnext' in model_name.lower():
            emergency_channel = target_ratio * 0.3  # Ultra conservative for ConvNext
            emergency_importance = "taylor"
            emergency_round_to = 2
        else:
            emergency_channel = target_ratio * 0.5
            emergency_importance = "taylor"
            emergency_round_to = 2
        
        print(f"[🔧] Emergency CNN fallback: channel={emergency_channel:.4f}, {emergency_importance}, round_to={emergency_round_to}")

        extended_search_remaining = state.get('extended_search_remaining')
        
        return {
            "importance_criterion": emergency_importance,
            "channel_pruning_ratio": emergency_channel,
            "round_to": emergency_round_to,
            "global_pruning": True,
            "rationale": f"Emergency fallback for {model_name} on {dataset}. All LLM methods failed: {error_msg}",
            "architecture_type": "cnn",
            "emergency_fallback": True,
            "llm_failed": True,
            "extended_search_remaining": extended_search_remaining
        }

    async def _llm_calculate_cnn_strategy_with_guidance(
        self,
        target_ratio,
        model_name,
        history,
        dataset,
        strategic_guidance,
        ):
        """
        Use LLM to calculate CNN strategy based on Master Agent's strategic guidance.
        MACs-first: if target MACs are provided in strategic_guidance, aim to hit them within tolerance.
        """

        round_to_instruction = """Use round_to=2 as the standard choice because:
    1. PRECISION: Finer granularity allows more precise MAC targeting
    2. ACCURACY PRESERVATION: Smaller pruning increments reduce risk of removing critical channel combinations
    3. FINE-GRAINED CONTROL: Enables hitting exact MAC budgets more accurately
    4. ITERATIVE REFINEMENT: Easier to fine-tune results with smaller steps"""


        # ---------- Pull MACs-first context (if provided) ----------
        def _to_g(val):
            if val is None:
                return None
            try:
                v = float(val)
            except Exception:
                return None
            # assume values > 1e6 are raw ops; convert to G
            return v / 1e9 if v > 1e6 else v

        baseline_macs = _to_g(
            strategic_guidance.get("baseline_macs", strategic_guidance.get("baseline_macs"))
        )
        target_macs = _to_g(
            strategic_guidance.get("target_macs", strategic_guidance.get("target_macs"))
        )
        macs_overshoot_tolerance_pct = float(strategic_guidance.get("macs_overshoot_tolerance_pct", 1.0))
        macs_undershoot_tolerance_pct = float(strategic_guidance.get("macs_undershoot_tolerance_pct", 5.0))

        # ---------- Existing context ----------
        history_analysis = self._analyze_channel_ratio_history(
            history, target_ratio, dataset, strategic_guidance.get('accuracy_threshold', 75.0)
        )
        avoid_combinations = strategic_guidance.get("avoid_combinations", [])
        recommendations = strategic_guidance.get("strategic_recommendations", [])
        failure_analysis = strategic_guidance.get("failure_analysis", {})
        acceptable_range = strategic_guidance.get("acceptable_range", {})

        avoided_combinations_text = ""
        if avoid_combinations:
            avoided_combinations_text = f"""
        🚨 ABSOLUTELY FORBIDDEN COMBINATIONS - DO NOT USE THESE:
        {chr(10).join(f"❌ importance_criterion='{combo[0]}', round_to={combo[1]} - CAUSED CATASTROPHIC FAILURE" for combo in avoid_combinations)}

        IF YOU CHOOSE ANY OF THESE COMBINATIONS, THE SYSTEM WILL FAIL.
        YOU MUST CHOOSE DIFFERENT VALUES FOR BOTH importance_criterion AND round_to.
        """

        macs_line = ""
        if target_macs is not None:
            macs_line = (
                f"\nMACs-FIRST OBJECTIVE:\n"
                f"- Baseline MACs: {(f'{baseline_macs:.3f}G' if baseline_macs is not None else 'N/A')}\n"
                f"- Target  MACs:  {target_macs:.3f}G\n"
                f"- Overshoot tolerance (strict): +{macs_overshoot_tolerance_pct:.1f}%\n"
                f"- Undershoot tolerance (lenient): -{macs_undershoot_tolerance_pct:.1f}%\n"
                f"- Acceptable range: {target_macs * (1 - macs_undershoot_tolerance_pct / 100):.3f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.3f}G\n"
                f"- Your strategy MUST land achieved MACs within this band."
            )

        guidance_text = f"""
        🚨 CATASTROPHIC RECOVERY - STRATEGIC GUIDANCE FROM MASTER AGENT 🚨
        ================================================================
        CRITICAL SITUATION: Previous parameter combinations have repeatedly failed to meet target.{macs_line}

        {avoided_combinations_text}

        STRATEGIC RECOMMENDATIONS FROM MASTER AGENT:
        {chr(10).join(f"- {rec}" for rec in recommendations)}

        FAILURE ANALYSIS:
        {chr(10).join(f"- {key}: {value}" for key, value in failure_analysis.items()) if failure_analysis else "No specific failure patterns identified"}

        ACCEPTABLE PARAMETER-REDUCTION RANGE (fallback if MACs target not provided):
        - Minimum acceptable: {acceptable_range.get('min_acceptable', target_ratio)*100:.1f}%
        - Maximum acceptable: {acceptable_range.get('max_acceptable', target_ratio + 0.02)*100:.1f}%
        - ANY undershoot is catastrophic in ratio-mode; prefer slight overshoot only when MACs target is absent.
        ================================================================
        """

        # ---------- Prompt (MACs-first primary, ratio as fallback) ----------
        prompt = f"""You are an expert Analysis Agent in CNN pruning for {model_name} on {dataset}.

        CRITICAL ROUND_TO INSTRUCTION: {round_to_instruction}

        {guidance_text}

        {history_analysis}

        🧠 CHANNEL RATIO → OUTCOME UNDERSTANDING:
        For CNN models, channel_pruning_ratio controls aggressiveness:
        - HIGHER ratios = MORE aggressive pruning = HIGHER parameter reduction & LOWER MACs (compute) but ACCURACY risk
        - LOWER ratios = LESS aggressive pruning = LOWER parameter reduction & HIGHER MACs (compute) but safer

        🎯 YOUR ENHANCED TASK (OPTIMIZE 3 PARAMETERS):
        1) CHANNEL_PRUNING_RATIO  — primary control to reach the compute budget (MACs-first). If no MACs target is given, match parameter reduction target.
        2) IMPORTANCE_CRITERION   — secondary control for accuracy preservation
        3) ROUND_TO               — fine-tuning for accuracy/hardware alignment

        PARAMETER LEARNING STRATEGY:
        - If the same channel_ratio produced a consistent MACs or reduction, compute a NEW ratio mathematically.
        - MACs-first when a target is provided: adjust ratio such that achieved MACs fall within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% of the target.
        - Fallback (no MACs target): new_ratio = old_ratio × (target_reduction / achieved_reduction)

        🚨 CRITICAL ENFORCEMENT RULES:
        1) DO NOT use any forbidden (importance_criterion, round_to) combinations above.
        2) You MAY vary channel_pruning_ratio — do not reuse the same failing value.
        3) If MACs target is present, you MUST propose a ratio that hits the MACs tolerance band.

        PARAMETER OPTIONS (examples, not limits):
        - channel_pruning_ratio: 0.10, 0.12, 0.15, 0.17, 0.19, 0.20, 0.22, 0.25
        - importance_criterion: "taylor", "l1norm", "l2norm"
        - round_to: 1, 2, 4, 8, 16

        OUTPUT FORMAT (JSON only):
        {{
        "channel_pruning_ratio": YOUR_CALCULATED_VALUE,
        "importance_criterion": "taylor|l1norm|l2norm",
        "round_to": X,
        "global_pruning": true,
        "parameter_tuning_order": ["channel_pruning_ratio", "importance_criterion", "round_to"],
        "rationale": "MACs-first if available: baseline={baseline_macs if baseline_macs is not None else 'N/A'}G, target={target_macs if target_macs is not None else 'N/A'}G, tol=+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%. If MACs not provided, use ratio target {target_ratio*100:.1f}%. Explain how history informed the new ratio.",    "architecture_type": "cnn",
        "calculation_details": {{
            "mode": "{'macs_first' if target_macs is not None else 'ratio_fallback'}",
            "historical_pattern": "channel_ratio=X → achieved MACs/ratio = Y",
            "macs_target_g": {"null" if target_macs is None else f"{target_macs:.3f}"},
            "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct:.2f},
            "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct:.2f},
            "math": "If MACs target present: choose ratio to land within tolerance; else new_ratio = old_ratio × (target/achieved)",
            "forbidden_check_passed": true
        }}
        }}

        FINAL VERIFICATION:
        1) If MACs target present: Will the proposed ratio land achieved MACs within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% of target?
        2) If MACs not present: Will the proposed ratio achieve {target_ratio*100:.1f}% ± 2% parameter reduction?
        3) Did you avoid forbidden combinations?"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an Analysis Agent who must learn from history and strictly avoid forbidden parameter combinations (MACs-first when available)."),
                HumanMessage(content=prompt),
            ])

            strategy_dict = parse_llm_json_response(response.content)

            if strategy_dict and "importance_criterion" in strategy_dict and "round_to" in strategy_dict:
                chosen_importance = strategy_dict["importance_criterion"]
                chosen_round_to = strategy_dict["round_to"]
                chosen_channel_ratio = strategy_dict.get("channel_pruning_ratio", 0.2)
                chosen_combination = (chosen_importance, chosen_round_to)

                if chosen_combination in avoid_combinations:
                    print(f"[🚨] CRITICAL: LLM chose forbidden combination {chosen_combination}. Forcing safe alternative...")
                    strategy_dict = self._force_different_combination_cnn(
                        strategy_dict, avoid_combinations, target_ratio, dataset
                    )
                    print(
                        f"[🔧] Forced correction → ({strategy_dict['importance_criterion']}, {strategy_dict['round_to']}) "
                        f"with channel_ratio={strategy_dict.get('channel_pruning_ratio', 0.2):.3f}"
                    )
                else:
                    print(f"[✅] LLM avoided forbidden combos → {chosen_combination} | channel_ratio={chosen_channel_ratio:.3f}")

                # Attach MACs context so downstream nodes can log/validate MACs-first selection
                if target_macs is not None:
                    strategy_dict["baseline_macs"] = baseline_macs
                    strategy_dict["target_macs"] = target_macs
                    strategy_dict["macs_overshoot_tolerance_pct"] = macs_overshoot_tolerance_pct
                    strategy_dict["macs_undershoot_tolerance_pct"] = macs_undershoot_tolerance_pct

                return strategy_dict

            # If JSON invalid or missing required keys
            print("[⚠️] LLM CNN strategic guidance output invalid; falling back to baseline")
            return await self._llm_calculate_full_strategy_baseline(target_ratio, model_name, dataset)

        except Exception as e:
            print(f"[❌] CNN strategic guidance interpretation failed: {e}")
            return await self._llm_calculate_full_strategy_baseline(target_ratio, model_name, dataset)



    def _analyze_channel_ratio_history(self, history, target_ratio, dataset, accuracy_threshold=75.0):
        """Analyze historical channel ratio patterns for LLM learning"""
        
        if not history:
            return "No historical data available for channel ratio analysis."
        
        analysis = ["📊 HISTORICAL CHANNEL RATIO ANALYSIS:"]
        analysis.append("=" * 50)
        
        # Collect channel ratio data
        ratio_results = []
        for i, entry in enumerate(history, 1):
            strategy = entry.get('strategy_used', {})
            channel_ratio = (strategy.get('channel_ratio_used') or 
                            strategy.get('channel_pruning_ratio') or 
                            0.2)  # fallback
            achieved = entry.get('achieved_ratio', 0)
            importance = strategy.get('importance_criterion', 'unknown')
            round_to = strategy.get('round_to', 'unknown')
            
            ratio_results.append((channel_ratio, achieved, importance, round_to))
            analysis.append(f"Attempt {i}: channel_ratio={channel_ratio:.3f} → {achieved*100:.1f}% reduction (using {importance}, round_to={round_to})")
        
        # Calculate patterns
        if ratio_results:
            # Most common channel ratio
            most_used_ratio = max(set(r[0] for r in ratio_results), key=lambda x: [r[0] for r in ratio_results].count(x))  
            matching_results = [r[1] for r in ratio_results if r[0] == most_used_ratio]
            avg_achieved_with_most_used = sum(matching_results) / len(matching_results)
            # IDENTIFY HIGH-ACCURACY RESULTS THAT JUST NEED ADJUSTMENT
            high_accuracy_undershoots = []
            for ratio, achieved, importance, round_to in ratio_results:
                deviation = achieved - target_ratio
                if deviation < 0 and deviation > -0.01:  # Small undershoot (< 1%)
                    # Get accuracy for this attempt
                    for entry in history:
                        if (abs(entry.get('achieved_ratio', 0) - achieved) < 0.001 and
                            entry.get('strategy_used', {}).get('importance_criterion') == importance):
                            if dataset.lower() == 'imagenet':
                                accuracy = entry.get('zero_shot_top1_accuracy', 0)
                                if accuracy > accuracy_threshold:  # High accuracy threshold for ImageNet
                                    high_accuracy_undershoots.append({
                                        'ratio': ratio, 'achieved': achieved, 'accuracy': accuracy,
                                        'importance': importance, 'round_to': round_to, 'deviation': deviation
                                    })
                            else:
                                accuracy = entry.get('zero_shot_accuracy', 0)
                                if accuracy > accuracy_threshold:  # High accuracy threshold for CIFAR-10
                                    high_accuracy_undershoots.append({
                                        'ratio': ratio, 'achieved': achieved, 'accuracy': accuracy,
                                        'importance': importance, 'round_to': round_to, 'deviation': deviation
                                    })
                            break

            
            # Calculate what ratio should work
            suggested_ratio = most_used_ratio * (target_ratio / avg_achieved_with_most_used)
            
            analysis.append("")
            analysis.append("🔍 KEY PATTERN IDENTIFIED:")
            analysis.append(f"  Most used channel_ratio: {most_used_ratio:.3f}")
            analysis.append(f"  Average achieved with this ratio: {avg_achieved_with_most_used*100:.1f}%")
            analysis.append(f"  Target needed: {target_ratio*100:.1f}%")
            analysis.append("")
            analysis.append("🧮 MATHEMATICAL CALCULATION:")
            analysis.append(f"  new_ratio = {most_used_ratio:.3f} × ({target_ratio:.3f} / {avg_achieved_with_most_used:.3f})")
            analysis.append(f"  new_ratio = {suggested_ratio:.3f}")
            analysis.append("")
            analysis.append(f"💡 RECOMMENDATION: Try channel_ratio ≈ {suggested_ratio:.3f} to achieve {target_ratio*100:.1f}% reduction")
            analysis.append("Then optimize importance_criterion and round_to for accuracy.")
        
        return "\n".join(analysis)
        
    def _force_different_combination_cnn(self, strategy_dict, avoid_combinations, target_ratio, dataset):
        """Force a different parameter combination for CNN when LLM chooses forbidden ones"""
        
        # All possible combinations for CNN - now including channel ratio options
        importance_options = ['taylor', 'l1norm', 'l2norm']
        round_to_options = [1, 2, 4, 8, 16, None]
        channel_ratio_options = [0.10, 0.12, 0.15, 0.17, 0.19, 0.20, 0.22, 0.25]
        
        # Get the original channel ratio if LLM provided one
        original_channel_ratio = strategy_dict.get('channel_pruning_ratio', 0.2)
        
        # Find a combination not in avoid_combinations
        for importance in importance_options:
            for round_to in round_to_options:
                combination = (importance, round_to)
                if combination not in avoid_combinations:
                    print(f"[🔧] Found safe CNN combination: {combination}")
                    
                    # Keep the LLM's channel ratio if it calculated a new one
                    if original_channel_ratio != 0.2:
                        print(f"[🔧] Keeping LLM's calculated channel ratio: {original_channel_ratio:.3f}")
                        strategy_dict['channel_pruning_ratio'] = original_channel_ratio
                    else:
                        # If LLM didn't calculate a new ratio, suggest one based on target
                        suggested_ratio = target_ratio * 0.65  # Conservative estimate
                        strategy_dict['channel_pruning_ratio'] = suggested_ratio
                        print(f"[🔧] Calculated fallback channel ratio: {suggested_ratio:.3f}")
                    
                    strategy_dict['importance_criterion'] = importance
                    strategy_dict['round_to'] = round_to
                    strategy_dict['rationale'] = f"FORCED CORRECTION: Original choice was forbidden. Using safe combination {combination} with channel_ratio={strategy_dict['channel_pruning_ratio']:.3f} to avoid catastrophic failure. " + strategy_dict.get('rationale', '')
                    
                    # Add forced correction metadata
                    strategy_dict['forced_correction_applied'] = True
                    strategy_dict['original_choice_was_forbidden'] = True
                    strategy_dict['safe_combination_used'] = combination
                    strategy_dict['channel_ratio_preserved'] = (original_channel_ratio != 0.2)
                    
                    return strategy_dict
        
        # Fallback: Try different channel ratios with safe importance/round_to
        print(f"[⚠️] All importance/round_to combinations forbidden, trying different channel ratios")
        
        # Use conservative importance/round_to and vary channel ratio
        safe_importance = 'taylor'  # Most conservative
        safe_round_to = 8          # Hardware efficient
        safe_combination = (safe_importance, safe_round_to)
        
        # If even the safe combination is forbidden, use different round_to
        if safe_combination in avoid_combinations:
            for alt_round_to in [4, 2, 16, 1]:
                alt_combination = (safe_importance, alt_round_to)
                if alt_combination not in avoid_combinations:
                    safe_round_to = alt_round_to
                    safe_combination = alt_combination
                    break
        
        # Calculate a conservative channel ratio based on target
        fallback_channel_ratio = target_ratio * 0.6  # Even more conservative
        
        print(f"[🔧] Using ultimate fallback: {safe_combination} with channel_ratio={fallback_channel_ratio:.3f}")
        
        strategy_dict['importance_criterion'] = safe_importance
        strategy_dict['round_to'] = safe_round_to
        strategy_dict['channel_pruning_ratio'] = fallback_channel_ratio
        strategy_dict['rationale'] = f"EMERGENCY FALLBACK: All combinations exhausted. Using conservative {safe_combination} with calculated channel_ratio={fallback_channel_ratio:.3f} based on target {target_ratio:.3f}."
        
        # Add fallback metadata
        strategy_dict['emergency_fallback_applied'] = True
        strategy_dict['all_combinations_forbidden'] = True
        strategy_dict['calculated_channel_ratio'] = fallback_channel_ratio
        
        return strategy_dict
        
    async def _llm_apply_tiny_adjustment(self, target_ratio, model_name, strategic_guidance, dataset):
        """
        DEPRECATED: This method should no longer be used
        Replaced with natural learning approach
        """
        print(f"[⚠️] _llm_apply_tiny_adjustment called but deprecated")
        print(f"[🔄] Redirecting to natural learning approach")
        
        # Redirect to the natural learning approach
        return await self._llm_calculate_vit_strategy_with_guidance(
            target_ratio, model_name, [], dataset, strategic_guidance
        )

    async def _execute_vit_analysis_with_history(self, state, model_name):
        """
        NEW METHOD: ViT analysis with historical learning (MACs-first)
        REASON: Learn from previous attempts and aim for a specific MACs budget
        """
        dataset = state.get('dataset', 'cifar10')
        target_ratio = state.get('target_pruning_ratio')

        # Get MACs values from multiple possible sources with proper fallbacks
        baseline_macs_val = (
            state.get('baseline_macs') or 
            state.get('master_results', {}).get('strategic_guidance', {}).get('baseline_macs')
        )
        target_macs_val = (
            state.get('target_macs') or 
            state.get('master_results', {}).get('strategic_guidance', {}).get('target_macs')
        )

        # Normalize to units of G (if absolute ops provided)
        def _to_g(val):
            if val is None:
                return None
            try:
                v = float(val)
            except Exception:
                return None
            return v / 1e9 if v > 1e6 else v

        baseline_macs = _to_g(baseline_macs_val)
        target_macs = _to_g(target_macs_val)

        if target_ratio is None and baseline_macs and target_macs:
            target_ratio = (baseline_macs - target_macs) / baseline_macs
            print(f"[🔧] Calculated target ratio from MACs: {target_ratio:.3f} ({target_ratio*100:.1f}%)")
        elif target_ratio is None:
            raise ValueError("No target_pruning_ratio provided and no MAC targets available to calculate ratio")


        history = state.get('history', [])
        current_revision = state.get('revision_number', 0)

        # --- MACs-first context pulled from state (if available) ---
        macs_overshoot_tolerance_pct = float(state.get('macs_overshoot_tolerance_pct', 1.0))
        macs_undershoot_tolerance_pct = float(state.get('macs_undershoot_tolerance_pct', 5.0))
        

        print(f"[🧠] ViT Analysis with Historical Learning (MACs-first):")
        print(f"   Target ratio: {target_ratio:.3f} (user-specified)")
        print(f"   Target MACs:  {f'{target_macs:.3f}G' if target_macs is not None else 'N/A'} "
            f"(baseline {f'{baseline_macs:.3f}G' if baseline_macs is not None else 'N/A'}, "
            f"tolerance +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")
        print(f"   Model: {model_name}")
        print(f"   Revision: {current_revision}")
        print(f"   History entries: {len(history)}")

        try:
            strategic_guidance = state.get('master_results', {}).get('strategic_guidance')

            if strategic_guidance:
                guidance_type = strategic_guidance.get('guidance_type')

                if guidance_type == 'high_accuracy_preservation':
                    print(f"[🎯] HIGH-ACCURACY PRESERVATION MODE: Using tiny adjustment")
                    strategy = await self._llm_apply_tiny_adjustment(
                        target_ratio, model_name, strategic_guidance, dataset
                    )
                    approach = "high_accuracy_preservation"

                elif guidance_type == 'catastrophic_recovery':
                    print(f"[🚨] CATASTROPHIC RECOVERY MODE: Using strategic guidance")
                    accuracy_threshold = float(state.get('accuracy_threshold', 1.0))
                    sg = dict(strategic_guidance)
                    sg.update({
                        'baseline_macs': baseline_macs,
                        'target_macs': target_macs,
                        'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                        'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
                        'accuracy_threshold': accuracy_threshold,
                    })
                    strategy = await self._llm_calculate_vit_strategy_with_guidance(
                        target_ratio, model_name, history, dataset, sg
                    )
                    approach = "catastrophic_recovery_with_guidance"

                else:
                    print(f"[📋] GENERAL STRATEGIC GUIDANCE: Using guidance")
                    accuracy_threshold = float(state.get('accuracy_threshold', 1.0))
                    sg = dict(strategic_guidance)
                    sg.update({
                        'baseline_macs': baseline_macs,
                        'target_macs': target_macs,
                        'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                        'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
                        'accuracy_threshold': accuracy_threshold,
                    })
                    strategy = await self._llm_calculate_vit_strategy_with_history(
                        target_ratio,
                        model_name,
                        history,
                        dataset,
                        baseline_macs=baseline_macs,
                        target_macs=target_macs,
                        macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                        macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                    )
                    approach = "general_strategic_guidance"

            elif history:
                # MACs-first, history-driven
                print(f"[📚] Using historical learning from {len(history)} previous attempts (MACs-first)")
                strategy = await self._llm_calculate_vit_strategy_baseline(
                    target_ratio,
                    model_name,
                    dataset,
                    baseline_macs=baseline_macs,
                    target_macs=target_macs,
                    macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                    macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                )
                approach = "vit_macs_first_historical_learning"

            else:
                # First attempt – MACs-first baseline
                print(f"[🎯] First attempt - using MACs-first baseline strategy")
                strategy = await self._llm_calculate_vit_strategy_baseline(
                    target_ratio,
                    model_name,
                    dataset,
                    baseline_macs=baseline_macs,
                    target_macs=target_macs,
                    macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                    macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
                )
                approach = "vit_macs_first_baseline_estimate"

            # Check if this exact strategy was tried before (only if not high-accuracy preservation)
            if approach != "high_accuracy_preservation":
                already_tried, previous_entry = self._is_vit_strategy_already_tried_enhanced(
                    strategy, history, tolerance=0.02
                )
                if already_tried:
                    print(f"[🔄] ViT strategy already tried, requesting LLM variation (MACs-first)")
                    strategy = await self._llm_create_vit_strategy_variation(
                        strategy, previous_entry, target_ratio, model_name, dataset,
                        macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
                    )
                    approach += "_with_variation"

            # Add metadata
            strategy.update({
                "approach": approach,
                "llm_calculated": True,
                "fully_llm_driven": True,
                "revision": current_revision,
                "dataset": dataset,
                "model_name": model_name,
                "history_count": len(history),
                "enhanced_learning_applied": True,
                # propagate MACs context for downstream nodes
                "baseline_macs": baseline_macs,
                "target_macs": target_macs,
                "macs_overshoot_tolerance_pct": macs_overshoot_tolerance_pct,
                "macs_undershoot_tolerance_pct": macs_undershoot_tolerance_pct,
            })

            print(f"[✅] ViT strategy complete: {approach}")
            return strategy

        except Exception as e:
            print(f"[❌] ViT strategy generation failed: {e}")
            print(f"[🔄] Using emergency fallback")
            # Emergency fallback (keep signature)
            return self._create_emergency_vit_fallback_strategy(target_ratio, model_name, dataset, str(e), state)

        
    async def _llm_calculate_vit_strategy_with_history(
        self,
        target_ratio,
        model_name,
        history,
        dataset,
        baseline_macs=None,
        target_macs=None,
        macs_overshoot_tolerance_pct=1.0,
        macs_undershoot_tolerance_pct=5.0,
        ):
        """
        Use LLM to determine ViT isomorphic ratios based on historical results.
        """

        # Format history (existing helper; assumed to include MACs after your previous changes)
        history_text = self._format_vit_history_for_llm_learning(
            history, 
            target_ratio, 
            macs_overshoot_tolerance_pct, 
            macs_undershoot_tolerance_pct
        )

        # Optional debug of learning progress (unchanged)
        self._analyze_learning_progress(history, target_ratio)

        # Safe strings for prompt
        baseline_str = f"{baseline_macs:.3f}G" if baseline_macs is not None else "N/A"
        target_str   = f"{target_macs:.3f}G"   if target_macs   is not None else "N/A"

        # Dataset-aware accuracy threshold for narrative only
        try:
            accuracy_threshold = self._get_high_accuracy_threshold(dataset)
        except Exception:
            accuracy_threshold = 75.0


        # Calculate direction based on actual history
        direction_analysis = ""
        if history:
            last_entry = history[-1]
            last_achieved_macs = last_entry.get('achieved_macs', 0)
            last_target_macs = last_entry.get('target_macs', target_macs)
            
            if last_achieved_macs and last_target_macs:
                mac_error_pct = (last_achieved_macs - last_target_macs) / last_target_macs * 100
                
                if mac_error_pct > macs_overshoot_tolerance_pct:
                    direction_analysis = f"""
                    SITUATION ANALYSIS: Achieved MACs ({last_achieved_macs/1e9:.3f}G) > Target MACs ({last_target_macs/1e9:.3f}G)
                    MAC Error: +{mac_error_pct:.1f}% (exceeds +{macs_overshoot_tolerance_pct:.1f}% limit - insufficient pruning)
                    DIRECTION NEEDED: HIGHER multipliers for more aggressive pruning to reduce MACs
                    """
                elif mac_error_pct < -macs_undershoot_tolerance_pct:
                    direction_analysis = f"""
                    SITUATION ANALYSIS: Achieved MACs ({last_achieved_macs/1e9:.3f}G) < Target MACs ({last_target_macs/1e9:.3f}G)
                    MAC Error: {mac_error_pct:.1f}% (exceeds -{macs_undershoot_tolerance_pct:.1f}% limit - excessive pruning)
                    DIRECTION NEEDED: LOWER multipliers for less aggressive pruning to increase MACs
                    """
                else:
                    direction_analysis = f"""
                    SITUATION ANALYSIS: Achieved MACs ({last_achieved_macs/1e9:.3f}G) ≈ Target MACs ({last_target_macs/1e9:.3f}G)
                    MAC Error: {mac_error_pct:.1f}% (within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
                    DIRECTION NEEDED: Minor adjustments to fine-tune to exact target
                    """

        prompt = f"""You are an expert in ViT pruning for {model_name} on {dataset}.

        CRITICAL MACs-FIRST TASK: Analyze historical ViT pruning attempts and determine optimal isomorphic ratios
        to achieve a MACs budget of {target_str} (baseline {baseline_str}) within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance.

        CRITICAL MULTIPLIER-MAC RELATIONSHIP (MEMORIZE THIS):
        =====================================================
        - HIGHER multipliers = MORE aggressive pruning = FEWER MACs (closer to target if overweight)
        - LOWER multipliers = LESS aggressive pruning = MORE MACs (closer to target if underweight)
        - If achieved MACs > target MACs (too heavy): INCREASE multipliers to prune more
        - If achieved MACs < target MACs (too light): DECREASE multipliers to prune less

        {direction_analysis}

        HISTORICAL DATA (most relevant snippets first):
        {history_text}

        EXPLICIT REASONING CHAIN (complete these based on the relationship above):
        Step 1: "Across {len(history)} attempts, the highest accuracy observed was ___%."
        Step 2: "That attempt used mlp=___, qkv=___ and achieved ___G MACs with error ___% vs target."
        Step 3: "PATTERN ANALYSIS: The dominant failure pattern is [MAC overshoot/MAC undershoot/mixed results]."
        Step 4: "DIRECTION NEEDED: Based on MAC errors, I need [HIGHER/LOWER/FINE-TUNED] multipliers because [achieved MACs are consistently above/below/near target]."
        Step 5: "LEARNED RELATIONSHIP: In this system, increasing multipliers → fewer MACs (more pruning); decreasing multipliers → more MACs (less pruning)."
        Step 6: "TARGET ACHIEVEMENT: To reach {target_str}, I need to [increase/decrease/fine-tune] multipliers by [small/moderate/significant] amounts."
        Step 7: "SAFETY: I will adjust MLP more than QKV to preserve attention accuracy, using [conservative/moderate/aggressive] changes."
        Step 8: "CALCULATION: My new multipliers will be: mlp=___ (was ___), qkv=___ (was ___) with justification."

        LEARNING OBJECTIVES:
        1) Identify which (mlp,qkv) settings produced MACs closest to the target (within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%).
        2) Prefer strategies that kept accuracy ≥ {accuracy_threshold:.1f}% when possible.
        3) Treat attention (QKV) conservatively; move most change into MLP when adjusting.
        4) Make minimal changes if a high-accuracy attempt is near the target MACs band.
        5) Predict specific multipliers that will place achieved MACs inside the +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance band.

        SUCCESS BAND (MACs-first):
        - Allowed achieved MACs error: -{macs_undershoot_tolerance_pct:.1f}% … +{macs_overshoot_tolerance_pct:.1f}% relative to the target MACs.

        VIT ISOMORPHIC HINTS:
        - MLP multipliers govern feed-forward pruning (often ~60–70% of params/compute).
        - QKV multipliers govern attention pruning (often ~30–40%); prune conservatively.
        - Keep proj_multiplier=0.0 and head_multiplier=0.0 for stability.

        OUTPUT FORMAT (JSON only, no markdown, no extra text):
        {{
        "reasoning_chain_completed": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ... Step 6: ... Step 7: ... Step 8: ...",
        "learned_relationship": "Increasing multipliers → [lower/higher] MACs; decreasing multipliers → [higher/lower] MACs",
        "direction_decision": "I will [increase/decrease] mlp by __ and [increase/decrease] qkv by __ to meet +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% band",
        "best_previous_attempt": "MACs ___G (error ___%), mlp=___, qkv=___, accuracy ___%",
        "improvement_strategy": "Minimal/Moderate changes focused on MLP/QKV because ...",
        "importance_criterion": "taylor|l1norm|l2norm",
        "pruning_ratio": {target_ratio:.3f},
        "round_to": 2,def get_dataset_specific_content(
        "global_pruning": true,
        "parameter_tuning_order": ["importance_criterion", "pruning_ratio", "round_to"],
        "rationale": "MACs-first: target {target_str} within +{macs_overshoot_tolerance_pct:.1f}% / -{macs_undershoot_tolerance_pct:.1f}%. Based on history, these multipliers should land inside the band while preserving accuracy.",
            "architecture_type": "vit",
            "isomorphic_group_ratios": {{
                "qkv_multiplier": YOUR_HISTORY_DRIVEN_VALUE,
                "mlp_multiplier": YOUR_HISTORY_DRIVEN_VALUE,
                "proj_multiplier": 0.0,
                "head_multiplier": 0.0
            }},
        "confidence_check": "These multipliers will bring MACs into the +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% band because ..."
        }}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a ViT pruning expert who must reason through historical data step-by-step to improve multiplier predictions (MACs-first)."),
                HumanMessage(content=prompt)
            ])

            # print(f"[DEBUG] Raw LLM response in history learning (MACs-first): len={len(response.content)}")
            # print(f"[DEBUG] First 500 chars: {response.content[:500]}")
            # print(f"[DEBUG] History entries provided: {len(history)}")

            strategy_dict = parse_llm_json_response(response.content)

            # print(f"[DEBUG] Parsed strategy_dict: {strategy_dict}")
            # if isinstance(strategy_dict, dict):
            #     print(f"[DEBUG] Suggested importance: {strategy_dict.get('importance_criterion')}")
            #     print(f"[DEBUG] Suggested round_to: {strategy_dict.get('round_to')}")
            #     print(f"[DEBUG] Suggested isomorphic ratios: {strategy_dict.get('isomorphic_group_ratios', {})}")

            if strategy_dict and 'isomorphic_group_ratios' in strategy_dict:
                # Optional validation hook (kept as-is)
                strategy_dict = self._validate_learning_reasoning(strategy_dict, history, target_ratio)

                ratios = strategy_dict['isomorphic_group_ratios']
                mlp_mult = ratios.get('mlp_multiplier', 0)
                qkv_mult = ratios.get('qkv_multiplier', 0)

                print(f"[🧠] LLM learned from {len(history)} ViT attempts (MACs-first):")
                print(f"   Predicted MLP multiplier: {mlp_mult:.3f}")
                print(f"   Predicted QKV multiplier: {qkv_mult:.3f}")
                print(f"   Learned importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Learned round-to: {strategy_dict.get('round_to', 'unknown')}")

                learning_summary = (strategy_dict.get('rationale') or '')[:200]
                if learning_summary:
                    print(f"   Learning insight: {learning_summary}...")

                reasoning = strategy_dict.get('reasoning_chain_completed', '')
                if reasoning:
                    print(f"   Reasoning chain: {reasoning[:150]}...")

                direction = strategy_dict.get('direction_decision', '')
                if direction:
                    print(f"   Direction decision: {direction[:120]}...")

                if strategy_dict.get('learning_correction_applied'):
                    print(f"   ⚠️ Learning correction was applied")
                else:
                    print(f"   ✅ Learning direction validated")

                return strategy_dict
            else:
                print(f"[⚠️] LLM failed to learn valid multipliers from history")
                raise ValueError("LLM ViT historical learning parsing failed")

        except Exception as e:
            print(f"[❌] LLM ViT historical learning failed: {e}")
            print(f"[🔄] Falling back to baseline strategy (MACs-first)")
            # Pass through MACs context for consistency with baseline
            return await self._llm_calculate_vit_strategy_baseline(
                target_ratio, model_name, dataset,
                baseline_macs=baseline_macs,
                target_macs=target_macs,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
            )
        


    def _analyze_learning_progress(self, history, target_ratio):
        """Analyze if learning is progressing correctly"""
        
        print(f"[📊] LEARNING PROGRESS ANALYSIS:")
        print(f"   Target: {target_ratio*100:.1f}% parameter reduction")
        print(f"   Acceptable: {target_ratio*100:.1f}% to {(target_ratio + 0.02)*100:.1f}% (0% to +2% overshoot only)")
        
        if len(history) < 2:
            print(f"   Insufficient history for progress analysis")
            return
        
        recent_attempts = history[-5:] if len(history) >= 5 else history
        
        print(f"   Last {len(recent_attempts)} attempts:")
        for i, entry in enumerate(recent_attempts):
            achieved = entry.get('achieved_ratio', 0)
            deviation = achieved - target_ratio  # Don't use abs() - we care about direction
            
            # Status based on acceptable range: target to target+2%
            if deviation >= 0 and deviation <= 0.02:
                status = "✅ ACCEPTABLE"
            elif deviation < 0:
                status = f"❌ undershoot"
            else:  # deviation > 0.02
                status = f"❌ overshoot"
                
            print(f"     {status} Attempt {len(history)-len(recent_attempts)+i+1}: {achieved*100:.1f}% (deviation: {deviation*100:+.1f}%)")
        
        # Check if learning is moving in right direction (toward acceptable range)
        if len(recent_attempts) >= 3:
            # Calculate distance from acceptable range (not just target)
            def distance_from_acceptable(achieved):
                if achieved >= target_ratio and achieved <= target_ratio + 0.02:
                    return 0  # Within acceptable range
                elif achieved < target_ratio:
                    return target_ratio - achieved  # Undershooting
                else:  # achieved > target_ratio + 0.02
                    return achieved - (target_ratio + 0.02)  # Overshooting beyond tolerance
            
            distances = [distance_from_acceptable(e.get('achieved_ratio', 0)) for e in recent_attempts]
            recent_trend = distances[-1] - distances[-3]
            
            if recent_trend < -0.01:  # Moving closer to acceptable range
                print(f"   📈 Learning trend: IMPROVING (distance to acceptable range decreased by {abs(recent_trend)*100:.1f}%)")
            elif recent_trend > 0.01:  # Moving away from acceptable range
                print(f"   📉 Learning trend: WORSENING (distance to acceptable range increased by {recent_trend*100:.1f}%)")
            else:
                print(f"   ➡️ Learning trend: STABLE (minimal change)")
        
        # Check for repeated failures
        undershoot_count = sum(1 for e in recent_attempts if e.get('achieved_ratio', 0) < target_ratio)
        overshoot_count = sum(1 for e in recent_attempts if e.get('achieved_ratio', 0) > target_ratio + 0.02)
        acceptable_count = len(recent_attempts) - undershoot_count - overshoot_count
        
        print(f"   📊 Recent pattern: {undershoot_count} undershoot, {acceptable_count} acceptable, {overshoot_count} overshoot")
        
        if undershoot_count == len(recent_attempts):
            print(f"   ⚠️ PATTERN: All {undershoot_count} recent attempts undershoot target")
            print(f"   💡 RECOMMENDATION: Need HIGHER multipliers to reach acceptable range")
        elif overshoot_count == len(recent_attempts):
            print(f"   ⚠️ PATTERN: All {overshoot_count} recent attempts overshoot tolerance")
            print(f"   💡 RECOMMENDATION: Need LOWER multipliers to stay within +2% tolerance")

    def _validate_learning_reasoning(self, strategy_dict, history, target_ratio):
        """Simple validation - basic safety net for obvious directional errors"""
        
        if not history or len(history) < 2:
            return strategy_dict
        
        print(f"[🔍] VALIDATING LEARNING DIRECTION:")
        
        # Get current and previous multipliers
        current_ratios = strategy_dict.get('isomorphic_group_ratios', {})
        current_mlp = current_ratios.get('mlp_multiplier', 0)
        current_qkv = current_ratios.get('qkv_multiplier', 0)
        
        # Get last attempt results
        last_entry = history[-1]
        last_achieved = last_entry.get('achieved_ratio', 0)
        last_strategy = last_entry.get('strategy_used', {})

        # Get last attempt MAC results  
        achieved_macs = last_entry.get('achieved_macs')
        target_macs = last_entry.get('target_macs')

        last_ratios = last_strategy.get('isomorphic_group_ratios', last_strategy)
        last_mlp = last_ratios.get('mlp_multiplier', 0)
        last_qkv = last_ratios.get('qkv_multiplier', 0)
        
        avg_multiplier_increased = ((current_mlp + current_qkv) / 2) > ((last_mlp + last_qkv) / 2)

        if achieved_macs and target_macs:
            mac_error_pct = (achieved_macs - target_macs) / target_macs * 100
            # Get tolerance values from the strategy or use defaults
            macs_overshoot_tolerance_pct = strategy_dict.get('macs_overshoot_tolerance_pct', 10.0)
            macs_undershoot_tolerance_pct = strategy_dict.get('macs_undershoot_tolerance_pct', 5.0)
            
            is_mac_overshooting = mac_error_pct > macs_overshoot_tolerance_pct
            is_mac_undershooting = mac_error_pct < -macs_undershoot_tolerance_pct
            
            print(f"   MAC error: {mac_error_pct:+.1f}% (achieved: {achieved_macs/1e9:.3f}G vs target: {target_macs/1e9:.3f}G)")
            print(f"   MAC overshooting: {is_mac_overshooting}")
            print(f"   MAC undershooting: {is_mac_undershooting}")
        else:
            is_mac_overshooting = False
            is_mac_undershooting = False
            print(f"   No MAC data available for validation")

        print(f"   Multipliers increased: {avg_multiplier_increased}")
        
        # MAC-based corrections
        if is_mac_overshooting and not avg_multiplier_increased:
            print(f"   ❌ MAC ERROR: MACs too high but didn't increase multipliers")
            correction_factor = 1.4
            strategy_dict['isomorphic_group_ratios']['mlp_multiplier'] = min(2.0, current_mlp * correction_factor)
            strategy_dict['isomorphic_group_ratios']['qkv_multiplier'] = min(1.8, current_qkv * correction_factor)
            strategy_dict['learning_correction_applied'] = True
            print(f"   🔧 Applied 40% increase to reduce MACs")
            
        elif is_mac_undershooting and avg_multiplier_increased:
            print(f"   ❌ MAC ERROR: MACs too low but increased multipliers")
            reduction_factor = 0.7
            strategy_dict['isomorphic_group_ratios']['mlp_multiplier'] = max(0.1, current_mlp * reduction_factor)
            strategy_dict['isomorphic_group_ratios']['qkv_multiplier'] = max(0.1, current_qkv * reduction_factor)
            strategy_dict['learning_correction_applied'] = True
            print(f"   🔧 Applied 30% reduction to increase MACs")
        else:
            print(f"   ✅ Learning direction appears correct")
            strategy_dict['learning_correction_applied'] = False
        
        return strategy_dict
 
    async def _llm_calculate_vit_strategy_with_guidance(self, target_ratio, model_name, history, dataset, strategic_guidance):
        """
        ENHANCED: Natural learning prompt without magic numbers
        Integrates the advanced learning system from paste.txt
        """
        
        guidance_type = strategic_guidance.get('guidance_type')

        sg = strategic_guidance or {}
        target_macs = sg.get('target_macs')
        baseline_macs = sg.get('baseline_macs')
        macs_overshoot_tolerance_pct = sg.get('macs_overshoot_tolerance_pct', 1.0)
        macs_undershoot_tolerance_pct = sg.get('macs_undershoot_tolerance_pct', 5.0)
        accuracy_threshold = sg.get('accuracy_threshold')

        if guidance_type == 'high_accuracy_preservation':
            # ✅ NEW: Specialized prompt for learning from successful strategies
            learning_data = strategic_guidance.get('learning_data', {})
            
            successful_accuracy = learning_data.get('successful_accuracy', 0)
            successful_achieved = learning_data.get('successful_achieved_ratio', 0)
            successful_multipliers = learning_data.get('successful_multipliers', {})
            deviation = learning_data.get('deviation_from_target', 0)
            successful_strategy = learning_data.get('successful_strategy', {})
            
            prompt = f"""You are a ViT pruning expert analyzing a SUCCESSFUL high-accuracy strategy.

            SUCCESSFUL STRATEGY ANALYSIS:
            =============================
            Model: {model_name} on {dataset}
            Target: {target_ratio*100:.2f}% parameter reduction
            Successful result: {successful_achieved*100:.2f}% parameter reduction  
            Accuracy achieved: {successful_accuracy:.2f}% (EXCELLENT!)
            Deviation: {deviation*100:.2f}% ({'undershoot' if deviation < 0 else 'overshoot'})

            SUCCESSFUL MULTIPLIERS:
            - MLP multiplier: {successful_multipliers.get('mlp_multiplier', 'unknown')}
            - QKV multiplier: {successful_multipliers.get('qkv_multiplier', 'unknown')}
            - Importance: {successful_strategy.get('importance_criterion', 'unknown')}
            - Round-to: {successful_strategy.get('round_to', 'unknown')}

            YOUR LEARNING TASK:
            ==================
            1. ANALYZE the mathematical relationship:
            - These multipliers achieved {successful_achieved*100:.2f}% reduction
            - Target is {target_ratio*100:.2f}% reduction  
            - Calculate what multipliers would achieve the target

            2. MATHEMATICAL REASONING:
            - If multiplier X gives Y% reduction, what multiplier gives Z% reduction?
            - Use proportional scaling: new_multiplier = old_multiplier × (target_reduction / achieved_reduction)
            - BUT also consider that the relationship may not be perfectly linear

            3. PRESERVE SUCCESS:
            - Keep the same importance_criterion ({successful_strategy.get('importance_criterion')}) that worked
            - Keep the same round_to ({successful_strategy.get('round_to')}) that worked
            - Only adjust the multipliers to hit the target

            4. AVOID MAGIC NUMBERS:
            - Don't use arbitrary factors like 1.05x or 1.1x
            - Calculate based on the observed data and mathematical relationship
            - Show your mathematical reasoning

            CALCULATION EXAMPLE:
            If old_mlp_multiplier × target_ratio = {successful_multipliers.get('mlp_multiplier', 0.6) * target_ratio:.3f} actual MLP pruning
            And this gave {successful_achieved*100:.2f}% total reduction
            Then to get {target_ratio*100:.2f}% total reduction, I need...
            [Show your calculation]

            ❗IMPORTANT: All multiplier values must be valid floats. Do not return "null", "none", or placeholder strings.

            OUTPUT FORMAT (JSON only):
            {{
                "importance_criterion": "{successful_strategy.get('importance_criterion', 'taylor')}",
                "pruning_ratio": {target_ratio:.3f},
                "round_to": {successful_strategy.get('round_to', 2)},
                "global_pruning": true,
                "rationale": "MATHEMATICAL LEARNING: Successful strategy achieved {successful_achieved*100:.2f}% with multipliers [{successful_multipliers.get('mlp_multiplier')}, {successful_multipliers.get('qkv_multiplier')}]. To achieve {target_ratio*100:.2f}%, I calculated [show math]: new_mlp = X, new_qkv = Y. Preserved successful importance_criterion and round_to.",
                "architecture_type": "vit",
                "learning_applied": true,
                "mathematical_calculation": {{
                    "observed_mlp": {successful_multipliers.get('mlp_multiplier', 0.6)},
                    "observed_qkv": {successful_multipliers.get('qkv_multiplier', 0.3)},
                    "observed_reduction": {successful_achieved:.4f},
                    "target_reduction": {target_ratio:.4f},
                    "scaling_factor": "YOUR_CALCULATED_FACTOR",
                    "calculated_mlp": "YOUR_CALCULATED_MLP",
                    "calculated_qkv": "YOUR_CALCULATED_QKV",
                    "reasoning": "Explain your mathematical approach"
                }},
                "isomorphic_group_ratios": {{
                    "qkv_multiplier": <float>,       // e.g. 0.012
                    "mlp_multiplier": <float>,       // e.g. 0.045
                    "proj_multiplier": 0.0,
                    "head_multiplier": 0.0
                }}
            }}"""
            
            try:
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are a ViT pruning expert who learns from successful patterns and applies mathematical reasoning."),
                    HumanMessage(content=prompt)
                ])
                
                # print(f"[DEBUG] Enhanced learning LLM response length: {len(response.content)}")
                
                strategy_dict = parse_llm_json_response(response.content)

                # print("[🧪] strategy_dict keys:", strategy_dict.keys())
                # print("[🧪] strategy_dict['mathematical_calculation']:", strategy_dict.get("mathematical_calculation"))

                
                if strategy_dict and 'isomorphic_group_ratios' in strategy_dict:
                    ratios = strategy_dict['isomorphic_group_ratios']
                    mlp_mult = ratios.get('mlp_multiplier', 0)
                    qkv_mult = ratios.get('qkv_multiplier', 0)
                    confidence = strategy_dict.get('confidence_level', 'unknown')
                    
                    print(f"[🧠] Enhanced LLM learning results:")
                    print(f"   MLP multiplier: {mlp_mult:.3f}")
                    print(f"   QKV multiplier: {qkv_mult:.3f}")
                    print(f"   Confidence: {confidence}")
                    print(f"   Learning applied: {strategy_dict.get('learning_applied', False)}")
                    
                    # Show learning insights if provided
                    math_calc = strategy_dict.get('mathematical_calculation', {})
                    if math_calc:
                        print(f"   Mathematical reasoning:")
                        print(f"     Observed: mlp={math_calc.get('observed_mlp')}, qkv={math_calc.get('observed_qkv')}")
                        print(f"     Scaling factor: {math_calc.get('scaling_factor')}")
                        print(f"     New: mlp={math_calc.get('calculated_mlp')}, qkv={math_calc.get('calculated_qkv')}")
                    
                    # Add enhanced learning metadata
                    strategy_dict['enhanced_learning_used'] = True
                    strategy_dict['historical_analysis_applied'] = True
                    
                    return strategy_dict
                else:
                    print(f"[⚠️] Enhanced learning LLM failed to provide valid strategy")
                    raise ValueError("Enhanced learning strategy parsing failed")
                    
            except Exception as e:
                print(f"[❌] Enhanced learning LLM failed: {e}")
                print(f"[🔄] Falling back to standard learning method")
                

                return await self._llm_calculate_vit_strategy_with_history(
                    target_ratio, model_name, history, dataset,
                    baseline_macs=baseline_macs, target_macs=target_macs, 
                    macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                    macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
                )

        else:   
            # CREATE AND USE THE LEARNING ANALYZER FOR GUIDANCE TOO:
            learning_analyzer = ViTLearningAnalyzer()

            # Get comprehensive analysis even with strategic guidance
            learning_analysis = learning_analyzer.analyze_vit_multiplier_patterns(
                history, dataset,
                target_macs=target_macs,
                accuracy_threshold=accuracy_threshold,
                baseline_macs=baseline_macs,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
            )

            # Format learning analysis for inclusion in guidance
            analysis_text = _format_analysis_for_llm(learning_analysis, target_ratio, dataset)
            
            # Extract COMPLETE constraint information ONLY
            avoid_complete_signatures = strategic_guidance.get('avoid_complete_signatures', [])
            failed_multiplier_patterns = strategic_guidance.get('failed_multiplier_patterns', {})
            recommendations = strategic_guidance.get('strategic_recommendations', [])
            failure_analysis = strategic_guidance.get('failure_analysis', {})
            acceptable_range = strategic_guidance.get('acceptable_range', {})

            bidirectional_guidance = self._generate_bidirectional_strategic_guidance(
                history, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
            )

            print(f"[🔥] Bidirectional guidance generated: {len(bidirectional_guidance)} characters")
            if bidirectional_guidance:
                print(f"[🔥] Bidirectional guidance preview:")
                print(bidirectional_guidance[:300] + "..." if len(bidirectional_guidance) > 300 else bidirectional_guidance)
            else:
                print(f"[⚠️] No bidirectional guidance generated")

            # ✅ NEW: Format complete constraints for LLM
            complete_constraints_text = ""
            if avoid_complete_signatures:
                complete_constraints_text = f"""
            🚨 COMPLETE FAILED CONFIGURATIONS (DO NOT REPEAT ANY OF THESE):
            """
                for i, signature in enumerate(avoid_complete_signatures[:10]):
                    complete_constraints_text += f"""
            ❌ FAILED CONFIG {i+1}:
            - importance_criterion: {signature['importance_criterion']}
            - round_to: {signature['round_to']}
            - mlp_multiplier: {signature['mlp_multiplier']:.3f}
            - qkv_multiplier: {signature['qkv_multiplier']:.3f}
            - Result: {signature['mac_error_pct']:+.1f}% MAC error ({signature['failure_type']})
            """

            # ✅ NEW: Format multiplier pattern constraints
            multiplier_constraints_text = ""
            if failed_multiplier_patterns:
                multiplier_constraints_text = f"""
            🚨 DANGEROUS MULTIPLIER RANGES (AVOID THESE ZONES):
            """
                undershoot_patterns = failed_multiplier_patterns.get('undershoot_patterns', [])
                for pattern in undershoot_patterns[:5]:
                    mlp_min, mlp_max = pattern['mlp_range']
                    qkv_min, qkv_max = pattern['qkv_range']
                    multiplier_constraints_text += f"""
            ⚠️ DANGER ZONE: mlp_multiplier {mlp_min:.3f}-{mlp_max:.3f}, qkv_multiplier {qkv_min:.3f}-{qkv_max:.3f}
            Reason: {pattern['reason']}
            """

            guidance_text = f"""
            🚨 CATASTROPHIC RECOVERY - STRATEGIC GUIDANCE FROM MASTER AGENT 🚨
            ================================================================
            CRITICAL SITUATION: Previous parameter combinations have repeatedly failed to meet target.

            {bidirectional_guidance}

            {analysis_text}

            {complete_constraints_text}

            {multiplier_constraints_text}

            STRATEGIC RECOMMENDATIONS FROM MASTER AGENT:
            {chr(10).join(f"- {rec}" for rec in recommendations)}

            FAILURE ANALYSIS:
            {chr(10).join(f"- {key}: {value}" for key, value in failure_analysis.items()) if failure_analysis else "No specific failure patterns identified"}

            ACCEPTABLE RANGE REQUIREMENTS:
            - Minimum acceptable: {acceptable_range.get('min_acceptable', target_ratio)*100:.1f}%
            - Maximum acceptable: {acceptable_range.get('max_acceptable', target_ratio + 0.02)*100:.1f}%
            - ANY undershoot is catastrophic - must meet or exceed target
            ================================================================
            """

            multiplier_analysis = self._analyze_multiplier_patterns(history, target_ratio, dataset, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
            # existing heuristic direction text
            direction_guidance = self._generate_direction_guidance(history, target_ratio)

            # >>> NEW: compute explicit direction BEFORE building the prompt <<<
            explicit_direction_guidance = ""
            last_entry = history[-1] if history else None
            if last_entry:
                last_achieved_macs = last_entry.get('achieved_macs', 0)
                last_target_macs = last_entry.get('target_macs', target_macs)
                if last_achieved_macs and last_target_macs:
                    mac_error_pct = (last_achieved_macs - last_target_macs) / max(last_target_macs, 1e-9) * 100.0
                    if mac_error_pct > 0:
                        explicit_direction_guidance = f"""
            EXPLICIT DIRECTION REQUIRED:
            ============================
            Last attempt: {last_achieved_macs/1e9:.3f}G MACs vs target {last_target_macs/1e9:.3f}G
            MAC Error: +{mac_error_pct:.1f}% (TOO HEAVY - insufficient pruning)

            SOLUTION: You must use HIGHER multipliers than before to achieve more aggressive pruning.
            Previous multipliers were TOO LOW. Increase them by at least 50-100%.
            """
                    else:
                        explicit_direction_guidance = f"""
            EXPLICIT DIRECTION REQUIRED:
            ============================
            Last attempt: {last_achieved_macs/1e9:.3f}G MACs vs target {last_target_macs/1e9:.3f}G  
            MAC Error: {mac_error_pct:.1f}% (TOO LIGHT - excessive pruning)

            SOLUTION: You must use LOWER multipliers than before to reduce pruning aggressiveness.
            Previous multipliers were TOO HIGH. Decrease them by 10-20%.
            """

            # now build the prompt INCLUDING the explicit direction guidance
            prompt = f"""You are an expert Analysis Agent in ViT pruning for {model_name} on {dataset}.

            {explicit_direction_guidance}

            {guidance_text}

            {multiplier_analysis}

            {direction_guidance}

            🧠 MULTIPLIER → OUTCOME UNDERSTANDING:
            ===========================================
            For ViT models, multipliers control pruning aggressiveness:
            - HIGHER multipliers = MORE aggressive pruning = HIGHER parameter reduction + RISK of accuracy collapse
            - LOWER multipliers = LESS aggressive pruning = LOWER parameter reduction + SAFER accuracy

            Mathematical relationship:
            - Actual MLP pruning % = mlp_multiplier × {target_ratio:.3f}
            - Actual QKV pruning % = qkv_multiplier × {target_ratio:.3f}
            - Total parameter reduction ≈ weighted combination of MLP + QKV pruning

            🎯 YOUR SPECIFIC TASK:
            =====================
            Based on the historical analysis above:
            1. IDENTIFY what went wrong with previous multiplier choices
            2. DETERMINE the direction you need to adjust (higher/lower multipliers)
            3. CALCULATE specific multiplier values that will achieve {target_ratio*100:.1f}% ± 2% parameter reduction
            4. ENSURE your multipliers are SIGNIFICANTLY different from failed attempts (minimum 0.3 difference)

            🚨 CRITICAL ENFORCEMENT RULES:
            =============================
            1. YOU MUST NOT use multiplier values in any of the dangerous zones above
            2. CHECK YOUR COMPLETE CONFIGURATION against ALL constraint lists before finalizing
            3. Your (importance_criterion, round_to, mlp_multiplier, qkv_multiplier) must avoid ALL failed complete signatures
            4. You CAN reuse (importance_criterion, round_to) combinations if you use different multipliers
            5. State your validation: "VERIFIED: My complete configuration avoids all failed patterns"

            CRITICAL VALIDATION REQUIREMENTS:
            1. Check your COMPLETE CONFIGURATION (importance, round_to, mlp_multiplier, qkv_multiplier) against failed complete signatures above
            2. Check your (mlp_multiplier, qkv_multiplier) against the dangerous zones above
            3. You CAN reuse (importance, round_to) combinations if you use different multipliers
            4. State your validation: "VERIFIED: My complete configuration avoids all failed patterns"

            🔧 CRITICAL PARAMETER CONSTRAINTS:
            =================================
            ROUND-TO VALUES: Must be integers only!
            - Valid: 1, 2, 4, 8, 16, or null
            - NEVER use: 0.1, 0.01, 0.5, or any decimals

            IMPORTANCE CRITERION OPTIONS (RESEARCH-BASED PRIORITY ORDER):
            - "taylor": PREFERRED - Uses gradient information, proven superior to magnitude-based methods
            - "l1norm": FALLBACK - Use only if Taylor has failed repeatedly 
            - "l2norm": ALTERNATIVE FALLBACK - Use only if both Taylor and L1 have failed

            CRITICAL: Your system already uses isomorphic pruning by default (global_pruning=true), so "taylor" gives you the optimal Isomorphic + Taylor combination that achieves SOTA results.

            DEFAULT CHOICE: Always start with "taylor" unless historical analysis shows repeated Taylor failures.

            OUTPUT FORMAT (JSON only):
            {{
            "importance_criterion": "taylor|l1norm|l2norm",
            "pruning_ratio": {target_ratio:.3f},
            "round_to": X,
            "global_pruning": true,
            "rationale": "COMPLETE SIGNATURE CHECK: I verified my complete configuration is NOT in the failed list. HISTORICAL ANALYSIS: [analyze what went wrong with previous multipliers]. DIRECTION DECISION: [explain whether you need higher/lower multipliers and why]. CALCULATION: [show mathematical reasoning for your specific multiplier values]. UNIQUENESS: [confirm your multipliers are significantly different from failed attempts].",
            "architecture_type": "vit",
            "isomorphic_group_ratios": {{
                "qkv_multiplier": YOUR_CALCULATED_VALUE,
                "mlp_multiplier": YOUR_CALCULATED_VALUE,
                "proj_multiplier": 0.0,
                "head_multiplier": 0.0
            }},
            "calculation_details": {{
                "complete_signature_check_passed": true,
                "chosen_complete_signature": "(importance_criterion_value, round_to_value, mlp_multiplier_value, qkv_multiplier_value)",
                "previous_failure_analysis": "Explain what specific complete configurations caused what problems",
                "target_achievement_strategy": "Explain how your multipliers will achieve target parameter reduction",
                "uniqueness_verification": "Confirm your complete configuration is different from all previous failed attempts"
            }}
            }}

            FINAL VERIFICATION STEP:
            Before submitting your response, double-check:
            1. Is my complete configuration (importance_criterion, round_to, mlp_multiplier, qkv_multiplier) in the failed list? If YES, choose different values.
            2. Are my multipliers mathematically justified to hit {target_ratio*100:.1f}% target?
            3. Are my multipliers significantly different from previous failed attempts?

            CRITICAL: You MUST avoid the complete failed signatures. Base your choices on OBSERVED PATTERNS from history."""
            
            try:
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are an Analysis Agent who must learn from historical complete signature failures and STRICTLY AVOID complete failed configurations."),
                    HumanMessage(content=prompt)
                ])
                
                # Debug output
                # print(f"[DEBUG] Raw LLM response in strategic guidance:")
                # print(f"[DEBUG] Response length: {len(response.content)}")
                # print(f"[DEBUG] First 500 chars: {response.content[:500]}")
                # print(f"[DEBUG] Complete signatures to avoid: {len(avoid_complete_signatures)}")
                
                strategy_dict = parse_llm_json_response(response.content)
                
                # print(f"[DEBUG] Parsed strategy_dict: {strategy_dict}")
                # print(f"[DEBUG] Suggested importance: {strategy_dict.get('importance_criterion')}")
                # print(f"[DEBUG] Suggested round_to: {strategy_dict.get('round_to')}")
                # print(f"[DEBUG] Suggested isomorphic ratios: {strategy_dict.get('isomorphic_group_ratios', {})}")
                
                # ✅ NEW: Validate against complete signatures
                if strategy_dict and avoid_complete_signatures:
                    is_safe, safety_message = self._validate_against_complete_signatures(
                        strategy_dict, avoid_complete_signatures
                    )
                    
                    if not is_safe:
                        print(f"[🚨] COMPLETE SIGNATURE VIOLATION: {safety_message}")
                        print(f"[🔧] Forcing safe alternative...")
                        strategy_dict = self._force_safe_complete_strategy(
                            strategy_dict, avoid_complete_signatures, target_ratio, dataset
                        )
                    else:
                        print(f"[✅] LLM avoided complete failed signatures")
                
                # Keep the multiplier uniqueness validation
                if strategy_dict and 'isomorphic_group_ratios' in strategy_dict:
                    ratios = strategy_dict['isomorphic_group_ratios']
                    is_unique, msg = self._validate_multiplier_uniqueness(ratios, history)
                    
                    if not is_unique:
                        print(f"[🚨] LLM suggested non-unique multipliers: {msg}")
                        print(f"[🔧] Applying forced variation...")
                        strategy_dict = self._force_unique_multipliers(strategy_dict, history, target_ratio)
                
                return strategy_dict
                
            except Exception as e:
                print(f"[❌] Strategic guidance interpretation failed: {e}")
                return await self._llm_calculate_vit_strategy_baseline(target_ratio, model_name, dataset)


    def _force_different_combination(self, strategy_dict, avoid_combinations, target_ratio, dataset):
        """Force a different parameter combination when LLM chooses forbidden ones"""
        
        # All possible combinations
        importance_options = ['taylor', 'l1norm', 'l2norm']
        round_to_options = [1, 2, 4, 8, 16, None]
        
        # Find a combination not in avoid_combinations
        for importance in importance_options:
            for round_to in round_to_options:
                combination = (importance, round_to)
                if combination not in avoid_combinations:
                    print(f"[🔧] Found safe combination: {combination}")
                    strategy_dict['importance_criterion'] = importance
                    strategy_dict['round_to'] = round_to
                    strategy_dict['rationale'] = f"FORCED CORRECTION: Original choice was forbidden. Using safe combination {combination} to avoid catastrophic failure. " + strategy_dict.get('rationale', '')
                    
                    # Add forced correction metadata
                    if 'calculation_details' not in strategy_dict:
                        strategy_dict['calculation_details'] = {}
                    strategy_dict['calculation_details']['forced_correction_applied'] = True
                    strategy_dict['calculation_details']['original_choice_was_forbidden'] = True
                    strategy_dict['calculation_details']['safe_combination_used'] = combination
                    
                    return strategy_dict
        
        # Fallback (should never happen)
        print(f"[⚠️] Could not find safe combination, using conservative fallback")
        strategy_dict['importance_criterion'] = 'taylor'
        strategy_dict['round_to'] = 1
        return strategy_dict


    def _analyze_multiplier_patterns(self, history, target_ratio, dataset, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """Analyze specific multiplier → outcome patterns"""
        if not history:
            return "No historical multiplier data to analyze."
        
        analysis = ["📊 MULTIPLIER PATTERN ANALYSIS:"]
        analysis.append("=" * 50)
        
        for i, entry in enumerate(history):
            strategy = entry.get('strategy_used', {})
            ratios = strategy.get('isomorphic_group_ratios', {})
            mlp_mult = ratios.get('mlp_multiplier', 'unknown')
            qkv_mult = ratios.get('qkv_multiplier', 'unknown')
            achieved = entry.get('achieved_ratio', 0)
            accuracy = entry.get('zero_shot_top1_accuracy', 0) if dataset.lower() == 'imagenet' else entry.get('zero_shot_accuracy', 0)
            
            # Calculate actual pruning percentages
            if isinstance(mlp_mult, (int, float)) and isinstance(qkv_mult, (int, float)):
                actual_mlp_pruning = mlp_mult * target_ratio
                actual_qkv_pruning = qkv_mult * target_ratio
                
                analysis.append(f"Attempt {i+1}: MULTIPLIERS → OUTCOMES")
                analysis.append(f"  Input: mlp_multiplier={mlp_mult:.1f}, qkv_multiplier={qkv_mult:.1f}")
                analysis.append(f"  Actual pruning: {actual_mlp_pruning*100:.1f}% MLP, {actual_qkv_pruning*100:.1f}% QKV")
                analysis.append(f"  Results: {achieved*100:.1f}% param reduction, {accuracy:.1f}% accuracy")
                
                # Classify result
                if accuracy < 5:
                    analysis.append(f"  Status: 💥 CATASTROPHIC - multipliers were TOO AGGRESSIVE")
                elif achieved < target_ratio * (1 - macs_undershoot_tolerance_pct / 100.0):
                    analysis.append(f"  Status: 📉 UNDERSHOOT - multipliers were too conservative")
                elif achieved > target_ratio * (1 + macs_overshoot_tolerance_pct / 100.0):
                    analysis.append(f"  Status: 📈 OVERSHOOT - multipliers were too aggressive")
                else:
                    analysis.append(f"  Status: ✅ SUCCESS - good multiplier balance")
                analysis.append("")
        
        # Add pattern insights
        analysis.append("🔍 KEY INSIGHTS:")
        catastrophic_mults = []
        for entry in history:
            if (entry.get('zero_shot_top1_accuracy', 0) if dataset.lower() == 'imagenet' 
                else entry.get('zero_shot_accuracy', 0)) < 5:
                ratios = entry.get('strategy_used', {}).get('isomorphic_group_ratios', {})
                mlp = ratios.get('mlp_multiplier')
                qkv = ratios.get('qkv_multiplier')
                if mlp is not None and qkv is not None:
                    catastrophic_mults.append((mlp, qkv))
        
        if catastrophic_mults:
            analysis.append(f"- CATASTROPHIC combinations: {catastrophic_mults}")
            analysis.append(f"- These multipliers caused <5% accuracy - NEVER use similar values")
            avg_cat_mlp = sum(m[0] for m in catastrophic_mults) / len(catastrophic_mults)
            avg_cat_qkv = sum(m[1] for m in catastrophic_mults) / len(catastrophic_mults)
            analysis.append(f"- Average catastrophic: mlp={avg_cat_mlp:.1f}, qkv={avg_cat_qkv:.1f}")
            analysis.append(f"- RECOMMENDATION: Use multipliers significantly lower than {avg_cat_mlp:.1f}/{avg_cat_qkv:.1f}")
        
        return "\n".join(analysis)

    def _generate_direction_guidance(self, history, target_ratio, dataset, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """Generate specific direction guidance based on history"""
        if not history:
            return "No direction guidance available - first attempt."
        
        last_entry = history[-1]
        last_achieved = last_entry.get('achieved_ratio', 0)
        last_accuracy = last_entry.get('zero_shot_top1_accuracy', 0)
        
        guidance = ["🧭 DIRECTION GUIDANCE:"]
        guidance.append("=" * 30)
        
        if last_accuracy < 5:
            guidance.append("❌ LAST ATTEMPT: Catastrophic accuracy collapse")
            guidance.append("🔽 DIRECTION: Use MUCH LOWER multipliers (reduce by 0.4-0.6)")
            guidance.append("💡 REASON: Previous multipliers caused over-pruning")
        elif last_achieved < target_ratio * (1 - macs_undershoot_tolerance_pct / 100.0):
            guidance.append("📉 LAST ATTEMPT: undershoot parameter target")
            guidance.append("🔼 DIRECTION: Use HIGHER multipliers (increase by 0.2-0.3)")
            guidance.append("💡 REASON: Need more aggressive pruning to hit target")
        elif last_achieved > target_ratio * (1 + macs_overshoot_tolerance_pct / 100.0):
            guidance.append("📈 LAST ATTEMPT: overshoot parameter target")
            guidance.append("🔽 DIRECTION: Use LOWER multipliers (decrease by 0.1-0.2)")
            guidance.append("💡 REASON: Pruning was too aggressive")
        else:
            guidance.append("✅ LAST ATTEMPT: Hit parameter target but check accuracy")
            guidance.append("🎯 DIRECTION: Minor adjustments for accuracy optimization")
        
        return "\n".join(guidance)

    def _validate_multiplier_uniqueness(self, proposed_ratios, history, min_diff=0.3):
        """Validate that proposed multipliers are sufficiently different"""
        prop_mlp = proposed_ratios.get('mlp_multiplier')
        prop_qkv = proposed_ratios.get('qkv_multiplier')
        
        if prop_mlp is None or prop_qkv is None:
            return True, "No multipliers to validate"
        
        for i, entry in enumerate(history):
            ratios = entry.get('strategy_used', {}).get('isomorphic_group_ratios', {})
            hist_mlp = ratios.get('mlp_multiplier')
            hist_qkv = ratios.get('qkv_multiplier')
            
            if hist_mlp is not None and hist_qkv is not None:
                mlp_diff = abs(prop_mlp - hist_mlp)
                qkv_diff = abs(prop_qkv - hist_qkv)
                
                if mlp_diff < min_diff and qkv_diff < min_diff:
                    return False, f"Too similar to attempt {i+1}: ({hist_mlp:.1f}, {hist_qkv:.1f})"
        
        return True, "Multipliers are sufficiently unique"

    def _force_unique_multipliers(self, strategy_dict, history, target_ratio):
        """Force unique multipliers when LLM suggests duplicates"""
        
        # Get LLM's intended multipliers
        proposed_ratios = strategy_dict['isomorphic_group_ratios']
        llm_mlp = proposed_ratios.get('mlp_multiplier', 0.5)
        llm_qkv = proposed_ratios.get('qkv_multiplier', 0.3)
        
        # Find all previously used multipliers
        used_mults = []
        for entry in history:
            ratios = entry.get('strategy_used', {}).get('isomorphic_group_ratios', {})
            mlp = ratios.get('mlp_multiplier')
            qkv = ratios.get('qkv_multiplier')
            if mlp is not None and qkv is not None:
                used_mults.append((mlp, qkv))
        
        print(f"[🎯] LLM intended: mlp={llm_mlp}, qkv={llm_qkv}")
        print(f"[📊] Previously used: {used_mults}")
        
        # ✅ NEW: Generate variations AROUND the LLM's intended values
        base_mlp = llm_mlp
        base_qkv = llm_qkv
        
        # Try small variations around LLM's choice
        variations = [
            (base_mlp + 0.1, base_qkv),      # Slight increase in MLP
            (base_mlp - 0.1, base_qkv),      # Slight decrease in MLP  
            (base_mlp, base_qkv + 0.1),      # Slight increase in QKV
            (base_mlp, base_qkv - 0.1),      # Slight decrease in QKV
            (base_mlp + 0.05, base_qkv + 0.05),  # Both slight increase
            (base_mlp - 0.05, base_qkv - 0.05),  # Both slight decrease
        ]
        
        # Find first variation that's unique and reasonable
        for mlp_try, qkv_try in variations:
            # Keep within reasonable bounds
            mlp_try = max(0.1, min(2.0, mlp_try))
            qkv_try = max(0.1, min(1.5, qkv_try))
            
            # Check if unique
            is_unique = True
            for used_mlp, used_qkv in used_mults:
                if abs(mlp_try - used_mlp) < 0.2 and abs(qkv_try - used_qkv) < 0.2:
                    is_unique = False
                    break
            
            if is_unique:
                strategy_dict['isomorphic_group_ratios']['mlp_multiplier'] = mlp_try
                strategy_dict['isomorphic_group_ratios']['qkv_multiplier'] = qkv_try
                print(f"[✅] SMART variation: mlp={mlp_try:.1f}, qkv={qkv_try:.1f} (near LLM intent)")
                return strategy_dict
        
        # If no close variation works, use LLM's original choice anyway
        # print(f"[⚠️] No unique variation found, keeping LLM choice: mlp={llm_mlp}, qkv={llm_qkv}")
        strategy_dict['isomorphic_group_ratios']['mlp_multiplier'] = llm_mlp
        strategy_dict['isomorphic_group_ratios']['qkv_multiplier'] = llm_qkv
        return strategy_dict

        
    def _format_vit_history_for_llm_learning(self, history, target_ratio, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct):
        """
        Format ViT history for LLM learning - focus on multiplier relationships
        REASON: LLM needs structured data to learn patterns from previous attempts
        """
        
        if not history:
            return "No historical ViT data available for learning."
        
        # Asymmetric tolerance (percent MACs error): acceptable range is [-undershoot, +overshoot]
        lower_tol_pct = float(macs_undershoot_tolerance_pct)
        upper_tol_pct = float(macs_overshoot_tolerance_pct)

        formatted = []
        formatted.append(f"LEARNING FROM {len(history)} PREVIOUS VIT ATTEMPTS:")
        formatted.append("=" * 60)
        
        # Track patterns for summary
        overshoots = 0
        undershoots = 0
        catastrophic_failures = 0
        successful_attempts = 0
        within_tolerance_count = 0
        
        for i, entry in enumerate(history, 1):
            strategy = entry.get('strategy_used', {})
            ratios = strategy.get('isomorphic_group_ratios', strategy)  # Fallback if nested
            
            # Extract values safely FIRST
            mlp_mult = ratios.get('mlp_multiplier')
            qkv_mult = ratios.get('qkv_multiplier')
            achieved_ratio = entry.get('achieved_ratio', 0)
            target_was = entry.get('target_ratio', target_ratio)
            
            # THEN convert to safe strings for formatting
            mlp_mult_str = self.safe_format_float(mlp_mult, 3, "unknown")
            qkv_mult_str = self.safe_format_float(qkv_mult, 3, "unknown")
            importance_str = self.safe_format_any(strategy.get('importance_criterion'), "unknown")
            round_to_str = self.safe_format_any(strategy.get('round_to'), "unknown")
            
            # Calculate deviation from MACs target
            achieved_ops = entry.get('achieved_macs') or entry.get('final_macs') or entry.get('macs_after')
            target_ops = entry.get('target_macs')

            if achieved_ops and target_ops:
                # signed percent error in MACs: + = too heavy (not enough pruning), − = too light (too much pruning)
                macs_error_pct = (float(achieved_ops) - float(target_ops)) / max(float(target_ops), 1e-9) * 100.0

                # Policy mapping:
                #   ratio overshoot 0…+2%  ⇢  MACs error −0…−2%  (<= 0 and ≥ −tol)
                #   ratio undershoot       ⇢  MACs error > 0     (too heavy)
                #   ratio overshoot >2%    ⇢  MACs error < −tol  (too light beyond limit)

                if (-lower_tol_pct <= macs_error_pct <= upper_tol_pct):  # within asymmetric tolerance
                    within_tolerance_count += 1
                    successful_attempts += 1
                    status = f"🎯 SUCCESS (MACs error {macs_error_pct:.2f}% within [-{lower_tol_pct:.1f}%, +{upper_tol_pct:.1f}%])"
                elif macs_error_pct > upper_tol_pct:  # insufficient pruning - need HIGHER multipliers
                    undershoots += 1
                    status = f"📉 INSUFFICIENT PRUNING (MACs +{macs_error_pct:.2f}% over budget — beyond +{upper_tol_pct:.1f}% limit; NEED HIGHER MULTIPLIERS)"
                elif macs_error_pct < -lower_tol_pct:  # too much pruning beyond allowed undershoot
                    overshoots += 1
                    status = f"📈 OVERSHOOT (MACs {macs_error_pct:.2f}% under budget – beyond {lower_tol_pct:.1f}% limit)"
                else:
                    catastrophic_failures += 1
                    status = f"💥 CATASTROPHIC (MACs miss {macs_error_pct:.2f}%)"
            else:
                # Fallback if MACs aren’t recorded for this entry
                catastrophic_failures += 1
                status = "💥 CATASTROPHIC (missing MACs fields)"

                
            formatted.append(f"Attempt {i}: {status}")
            formatted.append(f"  Strategy: mlp_mult={mlp_mult_str}, qkv_mult={qkv_mult_str}, importance={importance_str}, round_to={round_to_str}")
            formatted.append(f"  Target: {target_was*100:.1f}% → Achieved: {achieved_ratio*100:.1f}%")
            formatted.append(f"  Deviation: {(achieved_ratio - target_was)*100:.1f}% (tolerance: -{macs_undershoot_tolerance_pct:.1f}% to +{macs_overshoot_tolerance_pct:.1f}%)")


            # print("DEBUG entry keys:", list(entry.keys()))
            # print("DEBUG dataset raw:", entry.get("dataset"), type(entry.get("dataset")))

            # Add accuracy context for learning
            dataset = entry.get('dataset', '').lower()
            if dataset == 'imagenet':
                accuracy = entry.get('zero_shot_top1_accuracy', entry.get('zero_shot_accuracy', 0))
                acc_label = "Top-1"
            else:
                accuracy = entry.get('zero_shot_accuracy', 0)
                acc_label = "Acc"
            
            if accuracy is not None:
                formatted.append(f"  Zero-shot {acc_label}: {accuracy:.2f}%")
            
            formatted.append("")
        
        # Rest of the method remains the same...
        return "\n".join(formatted)
    

    def _is_vit_strategy_already_tried(self, proposed_strategy, history, macs_overshoot_tolerance_pct=1.0, tolerance=0.05, macs_undershoot_tolerance_pct=5.0):
        """
        Check if proposed ViT strategy (multiplier combination) was already attempted.
        Note: macs_overshoot_tolerance_pct and macs_undershoot_tolerance_pct are included for
        signature consistency across callers but are not used in this uniqueness check.
        """

        proposed_ratios = proposed_strategy.get('isomorphic_group_ratios', {})
        proposed_mlp = proposed_ratios.get('mlp_multiplier')
        proposed_qkv = proposed_ratios.get('qkv_multiplier')
        proposed_importance = proposed_strategy.get('importance_criterion')
        proposed_round_to = proposed_strategy.get('round_to')

        # Require numeric multipliers to compare
        if not isinstance(proposed_mlp, (int, float)) or not isinstance(proposed_qkv, (int, float)):
            return False, None  # Can't compare without valid numeric ratios

        for entry in history:
            strategy_used = entry.get('strategy_used', {})
            used_ratios = strategy_used.get('isomorphic_group_ratios', strategy_used)  # Fallback

            used_mlp = used_ratios.get('mlp_multiplier')
            used_qkv = used_ratios.get('qkv_multiplier')
            # Try both locations for these fields
            used_importance = strategy_used.get('importance_criterion') or used_ratios.get('importance_criterion')
            used_round_to = strategy_used.get('round_to') or used_ratios.get('round_to')

            # Compare only if both historical multipliers are numeric
            if isinstance(used_mlp, (int, float)) and isinstance(used_qkv, (int, float)):
                same_mults = (
                    abs(used_mlp - proposed_mlp) < tolerance and
                    abs(used_qkv - proposed_qkv) < tolerance
                )
                same_meta = (used_importance == proposed_importance and used_round_to == proposed_round_to)

                if same_mults and same_meta:
                    achieved = entry.get('achieved_ratio', 0)
                    target = entry.get('target_ratio', 0)

                    # print(f"[🔄] VIT STRATEGY REPETITION DETECTED:")
                    # print(f"   Previous: mlp={used_mlp:.3f}, qkv={used_qkv:.3f}, importance={used_importance}, round_to={used_round_to}")
                    # print(f"   Proposed: mlp={proposed_mlp:.3f}, qkv={proposed_qkv:.3f}, importance={proposed_importance}, round_to={proposed_round_to}")
                    # print(f"   Previous result: {achieved*100:.1f}% (target: {target*100:.1f}%)")
                    return True, entry

        return False, None

    
    def _is_vit_strategy_already_tried_enhanced(self, proposed_strategy, history, tolerance=0.1, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        ENHANCED: Better detection of similar strategies with intelligent tolerance.
        Note: macs_overshoot_tolerance_pct and macs_undershoot_tolerance_pct are included for
        signature consistency across callers but are not used in this uniqueness check.
        """
        
        proposed_ratios = proposed_strategy.get('isomorphic_group_ratios', {})
        proposed_mlp = proposed_ratios.get('mlp_multiplier')
        proposed_qkv = proposed_ratios.get('qkv_multiplier')
        proposed_importance = proposed_strategy.get('importance_criterion')
        proposed_round_to = proposed_strategy.get('round_to')
        
        # Require numeric multipliers to compare
        if not isinstance(proposed_mlp, (int, float)) or not isinstance(proposed_qkv, (int, float)):
            return False, None
        
        for entry in history:
            strategy_used = entry.get('strategy_used', {})
            used_ratios = strategy_used.get('isomorphic_group_ratios', strategy_used)
            
            used_mlp = used_ratios.get('mlp_multiplier')
            used_qkv = used_ratios.get('qkv_multiplier')
            used_importance = strategy_used.get('importance_criterion') or used_ratios.get('importance_criterion')
            used_round_to = strategy_used.get('round_to') or used_ratios.get('round_to')
            
            # Enhanced similarity check with intelligent tolerance
            if (isinstance(used_mlp, (int, float)) and isinstance(used_qkv, (int, float)) and
                abs(used_mlp - proposed_mlp) < tolerance and
                abs(used_qkv - proposed_qkv) < tolerance and
                used_importance == proposed_importance and 
                used_round_to == proposed_round_to):
                
                achieved = entry.get('achieved_ratio', 0)
                target = entry.get('target_ratio', 0)
                
                # print(f"[🔄] ENHANCED SIMILARITY DETECTION:")
                # print(f"   Previous: mlp={used_mlp:.3f}, qkv={used_qkv:.3f}, importance={used_importance}, round_to={used_round_to}")
                # print(f"   Proposed: mlp={proposed_mlp:.3f}, qkv={proposed_qkv:.3f}, importance={proposed_importance}, round_to={proposed_round_to}")
                # print(f"   Previous result: {achieved*100:.1f}% (target: {target*100:.1f}%)")
                # print(f"   Similarity tolerance: {tolerance}")
                return True, entry
        
        return False, None


    async def _llm_calculate_vit_strategy_baseline(self, target_ratio, model_name, dataset, baseline_macs=None, target_macs=None, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        Use LLM to determine baseline ViT strategy for first attempt (NO HISTORY)
        MACs-first: aim to hit the target MACs within +macs_overshoot_tolerance_pct / -macs_undershoot_tolerance_pct
        """
        # Safe strings for prompt (avoid formatting None with :.3f)
        baseline_str = f"{baseline_macs:.3f}G" if baseline_macs is not None else "N/A"
        target_str   = f"{target_macs:.3f}G"   if target_macs   is not None else "N/A"

        prompt = f"""You are an expert in ViT pruning for {model_name} on {dataset}.

        TASK: Propose a baseline ViT isomorphic pruning strategy that hits a MACs budget of {target_str} (baseline {baseline_str}).
        TOLERANCE: Achieved MACs must be within +{macs_overshoot_tolerance_pct:.1f}% / -{macs_undershoot_tolerance_pct:.1f}% of the target MACs (MACs-first objective).

        This is the FIRST ATTEMPT — there is no historical data. Make a conservative, accuracy-preserving choice.

        VIT ARCHITECTURE CONTEXT ({model_name.upper()}):
        - ViTs comprise MLP (feed-forward) blocks and attention blocks (QKV + projection).
        - MLP usually contributes ~60–70% of parameters; attention ~30–40%.
        - In a MACs-first objective, tune multipliers to bring MACs near the target, favoring MLP pruning over attention.

        DATASET HINTS ({dataset.upper()}):
        - ImageNet: strong accuracy preservation; keep attention pruning minimal or zero.
        - CIFAR-10: can be more aggressive; still avoid over-pruning attention early.

        BASELINE GUIDELINES:
        - Start conservative; prefer pruning in MLP over QKV.
        - Keep proj_multiplier=0.0 and head_multiplier=0.0.
        - Choose integer-friendly round_to for hardware alignment.

        OUTPUT FORMAT (JSON only):
        {{
        "importance_criterion": "taylor|l1norm|l2norm",
        "pruning_ratio": {target_ratio:.3f},
        "round_to": X,
        "global_pruning": true,
        "parameter_tuning_order": ["importance_criterion", "pruning_ratio", "round_to"],
        "rationale": "MACs-first baseline for {model_name} on {dataset}: target {target_str} within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%. Prefer MLP pruning, protect attention. Explain why chosen multipliers should land near the MACs target.",
        "architecture_type": "vit",
        "isomorphic_group_ratios": {{
            "qkv_multiplier": YOUR_CONSERVATIVE_ESTIMATE,
            "mlp_multiplier": YOUR_CONSERVATIVE_ESTIMATE,
            "proj_multiplier": 0.0,
            "head_multiplier": 0.0
        }}
        }}
        CRITICAL: Pick conservative multipliers likely to land within the asymmetric MACs tolerance band while preserving accuracy."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a ViT pruning expert making your first baseline strategy attempt (MACs-first)."),
                HumanMessage(content=prompt)
            ])

            strategy_dict = parse_llm_json_response(response.content)

            if strategy_dict and 'isomorphic_group_ratios' in strategy_dict:
                ratios = strategy_dict['isomorphic_group_ratios']
                mlp_mult = ratios.get('mlp_multiplier', 0)
                qkv_mult = ratios.get('qkv_multiplier', 0)

                print(f"[🤖] LLM baseline strategy for {model_name} (MACs-first):")
                print(f"   Target MACs: {target_str} (baseline {baseline_str}), tol +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%")
                print(f"   MLP multiplier: {mlp_mult:.3f}")
                print(f"   QKV multiplier: {qkv_mult:.3f}")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")

                return strategy_dict
            else:
                print(f"[⚠️] LLM failed to provide valid baseline ViT strategy")
                raise ValueError("LLM baseline ViT strategy parsing failed")

        except Exception as e:
            print(f"[❌] LLM baseline ViT strategy calculation failed: {e}")
            print(f"[🔄] Using hardcoded conservative fallback (MACs-first mindset)")

            # Conservative fallback when LLM completely fails
            if dataset.lower() == 'imagenet':
                # Protect attention on ImageNet in baseline fallback
                fallback_mlp = min(1.0, (target_ratio * 0.8) / 0.6)  # conservative tilt toward MLP
                fallback_qkv = 0.0                                   # no attention pruning in baseline fallback
                fallback_importance = "taylor"
                fallback_round_to = 2
            else:
                fallback_mlp = min(1.5, target_ratio / 0.6)  # moderate for CIFAR-10
                fallback_qkv = min(1.0, target_ratio / 0.4)  # still cautious with attention
                fallback_importance = "taylor"
                fallback_round_to = 1

            return {
                "importance_criterion": fallback_importance,
                "pruning_ratio": target_ratio,
                "round_to": fallback_round_to,
                "global_pruning": True,
                "parameter_tuning_order": ["importance_criterion", "pruning_ratio", "round_to"],
                "rationale": (
                    f"Conservative MACs-first fallback for {model_name} on {dataset} due to LLM failure: {str(e)}. "
                    f"Target {target_str} (baseline {baseline_str}), tol +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%. "
                    f"Favor MLP pruning and protect attention to preserve accuracy."
                ),
                "architecture_type": "vit",
                "isomorphic_group_ratios": {
                    "qkv_multiplier": fallback_qkv,
                    "mlp_multiplier": fallback_mlp,
                    "proj_multiplier": 0.0,
                    "head_multiplier": 0.0
                },
                "fallback_used": True
            }      

    def _create_emergency_vit_fallback_strategy(self, target_ratio, model_name, dataset, error_msg, state, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        Emergency fallback when both ViT LLM methods fail
        REASON: Provide ultra-safe fallback to prevent complete system failure (MACs-first mindset)
        """
        print(f"[🚨] Emergency ViT fallback for {model_name} on {dataset} (MACs-first)")

        # Ultra-conservative approach based on dataset (favor accuracy; keep MACs near/under budget)
        if dataset.lower() == 'imagenet':
            # For ImageNet, protect attention completely in emergency; keep MLP modest
            emergency_mlp = min(0.8, target_ratio * 0.5)  # very conservative MLP actual ≈ 0.5×target_ratio
            emergency_qkv = 0.0                           # skip attention pruning in emergency
            emergency_importance = "taylor"
            emergency_round_to = 2
        else:  # CIFAR-10
            # More permissive but still conservative
            emergency_mlp = min(1.2, target_ratio * 0.8)
            emergency_qkv = min(0.6, target_ratio * 0.4)
            emergency_importance = "taylor"
            emergency_round_to = 2

        print(f"[🔧] Emergency ViT fallback: mlp={emergency_mlp:.3f}, qkv={emergency_qkv:.3f}, "
            f"{emergency_importance}, round_to={emergency_round_to} (tol +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")

        extended_search_remaining = state.get('extended_search_remaining')

        return {
            "importance_criterion": emergency_importance,
            "pruning_ratio": target_ratio,
            "round_to": emergency_round_to,
            "global_pruning": True,
            "parameter_tuning_order": ["importance_criterion", "pruning_ratio", "round_to"],
            "rationale": (
                f"Emergency fallback for {model_name} on {dataset} (LLM failed: {error_msg}). "
                f"MACs-first policy: aim to land within -{macs_undershoot_tolerance_pct:.1f}% to +{macs_overshoot_tolerance_pct:.1f}% of MACs target; "
                f"protect attention (qkv={emergency_qkv:.2f}) and use conservative MLP "
                f"(mlp={emergency_mlp:.2f}) to preserve accuracy."
            ),
            "architecture_type": "vit",
            "isomorphic_group_ratios": {
                "qkv_multiplier": emergency_qkv,
                "mlp_multiplier": emergency_mlp,
                "proj_multiplier": 0.0,
                "head_multiplier": 0.0
            },
            "emergency_fallback": True,
            "llm_failed": True,
            "extended_search_remaining": extended_search_remaining
        }

    def _format_history_for_llm(self, history, target_ratio, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        MACs-first history formatting - focus on hitting the MACs budget
        Allowed band: achieved MACs within asymmetric tolerance of target MACs
        """
        if not history:
            return "No previous attempts."

        overshoot_tol_pct = macs_overshoot_tolerance_pct
        undershoot_tol_pct = macs_undershoot_tolerance_pct

        formatted = []
        for i, entry in enumerate(history, 1):
            strategy = entry.get('strategy_used', {})
            channel_ratio = strategy.get('channel_ratio_used', strategy.get('channel_pruning_ratio', 'unknown'))

            # Prefer MACs-first fields if available
            achieved_macs = entry.get('achieved_macs')
            target_macs   = entry.get('target_macs')
            macs_status_line = None

            if achieved_macs is not None and target_macs is not None and target_macs > 0:
                macs_error_pct = (float(achieved_macs) - float(target_macs)) / float(target_macs) * 100.0
                tol_pct = tolerance * 100.0

                if -undershoot_tol_pct <= macs_error_pct <= overshoot_tol_pct:
                    status = f"🎯 SUCCESS (MACs {macs_error_pct:+.2f}% vs target; allowed -{undershoot_tol_pct:.1f}%/+{overshoot_tol_pct:.1f}%)"
                elif macs_error_pct > overshoot_tol_pct:
                    status = f"📉 undershoot (too heavy: +{macs_error_pct:.2f}% over MACs budget beyond +{overshoot_tol_pct:.1f}% limit)"
                elif macs_error_pct < -undershoot_tol_pct:
                    status = f"📈 overshoot (too light: {macs_error_pct:.2f}% under budget beyond -{undershoot_tol_pct:.1f}% limit)"
                else:
                    status = f"💥 CATASTROPHIC (MACs miss {macs_error_pct:+.2f}%)"

                macs_status_line = (
                    f"Attempt {i}: channel_ratio={channel_ratio} → "
                    f"achieved {achieved_macs/1e9:.3f}G MACs, target {target_macs/1e9:.3f}G ({status})"
                )
                formatted.append(macs_status_line)
            else:
                # Fallback to legacy ratio-based summary if MACs not recorded
                achieved_ratio = entry.get('achieved_ratio', 0.0)
                target_was     = entry.get('target_ratio', target_ratio)

                if achieved_ratio > 0:
                    distance_from_target = abs(achieved_ratio - target_was)
                    if distance_from_target > 0.02:
                        status = "💥 CATASTROPHIC (far from target)"
                    elif distance_from_target <= (macs_undershoot_tolerance_pct/100):  # Use undershoot as fallback tolerance
                        status = "🎯 SUCCESS (within tolerance)"
                    else:
                        status = "❌ MISSED"
                else:
                    status = "💥 FAILED"

                formatted.append(
                    f"Attempt {i}: channel_ratio={channel_ratio} → "
                    f"achieved {achieved_ratio*100:.1f}% parameter reduction, "
                    f"target {target_was*100:.1f}% ({status})"
                )

        # Footer guidance (MACs-first)
        formatted.append(f"\nTARGET: Achieve MACs within -{undershoot_tol_pct:.1f}%/+{overshoot_tol_pct:.1f}% of the MACs budget.")
        formatted.append(f"LEARN THE PATTERN: What channel ratio produces MACs in the -{undershoot_tol_pct:.1f}%/+{overshoot_tol_pct:.1f}% band?")
        formatted.append("Focus ONLY on MACs target adherence for this step (accuracy is handled elsewhere).")

        return "\n".join(formatted)


    async def _llm_create_vit_strategy_variation(self, original_strategy, previous_entry, target_ratio, model_name, dataset, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        LLM creates intelligent variation when ViT strategy was already tried
        REASON: Avoid infinite loops by creating smart variations of repeated strategies
        """

        previous_achieved = previous_entry.get('achieved_ratio', 0)

        # --- MACs context for variation logic (MACs-first workflow) ---
        prev_target_macs   = previous_entry.get('target_macs')
        prev_achieved_macs = previous_entry.get('achieved_macs')
        macs_error_pct = None  # + = over budget (too heavy), − = under budget (too light)
        if prev_target_macs and prev_achieved_macs:
            macs_error_pct = (float(prev_achieved_macs) - float(prev_target_macs)) / max(float(prev_target_macs), 1e-9) * 100.0

        original_ratios = original_strategy.get('isomorphic_group_ratios', {})
        original_mlp = original_ratios.get('mlp_multiplier', 0.6)
        original_qkv = original_ratios.get('qkv_multiplier', 0.3)

        overshoot_tol_pct = macs_overshoot_tolerance_pct
        undershoot_tol_pct = macs_undershoot_tolerance_pct

        # Pre-format strings for safe insertion in prompt
        target_macs_str   = f"{prev_target_macs/1e9:.3f}G" if prev_target_macs else "N/A"
        achieved_macs_str = f"{prev_achieved_macs/1e9:.3f}G" if prev_achieved_macs else "N/A"
        macs_error_str    = f"{macs_error_pct:+.2f}%" if macs_error_pct is not None else "N/A"

        prompt = f"""You are a ViT pruning expert creating a variation of a strategy that was already tried.

        SITUATION: The original ViT strategy was already attempted and needs adjustment.

        MACs REQUIREMENT: Achieved MACs must be within -{undershoot_tol_pct:.1f}%/+{overshoot_tol_pct:.1f}% of the target MACs.

        ORIGINAL STRATEGY:
        - MLP multiplier: {original_mlp:.3f}
        - QKV multiplier: {original_qkv:.3f}
        - Importance: {original_strategy.get('importance_criterion', 'unknown')}
        - Round-to: {original_strategy.get('round_to', 'unknown')}

        PREVIOUS RESULT:
        - Target MACs: {target_macs_str}
        - Achieved MACs: {achieved_macs_str}
        - MACs error: {macs_error_str}  (positive = over budget / too heavy; negative = under budget / too light)

        TASK: Create a modified ViT strategy that will bring achieved MACs into the allowed band [-{undershoot_tol_pct:.1f}%, +{overshoot_tol_pct:.1f}%] relative to target MACs.
        
        ANALYSIS GUIDANCE:
        - If MACs error > +{overshoot_tol_pct:.1f}% (too heavy / over budget): Increase multipliers strategically
        - If MACs error < -{undershoot_tol_pct:.1f}% (too light beyond allowed under-budget): Decrease multipliers carefully
        - Consider changing importance criterion or round_to if needed
        - Make smart adjustments based on {model_name} characteristics and ViT architecture

        VARIATION STRATEGY:
        - Prioritize safe changes: increase/decrease MLP more than QKV (attention is critical)
        - Avoid dramatic changes that could cause catastrophic failure
        - Consider the parameter distribution: MLP ≈ 60-70%, Attention ≈ 30-40%

        OUTPUT FORMAT (JSON only):
        {{
        "importance_criterion": "taylor|l1norm|l2norm",
        "pruning_ratio": {target_ratio:.3f},
        "round_to": X,
        "global_pruning": true,
        "parameter_tuning_order": ["importance_criterion", "pruning_ratio", "round_to"],
        "rationale": "Variation strategy (MACs-first): Previous MACs error = {macs_error_str}. Policy allows -{undershoot_tol_pct:.1f}%/+{overshoot_tol_pct:.1f}%. Therefore I [increased/decreased] mlp_multiplier from {original_mlp:.3f}→[new] and qkv_multiplier from {original_qkv:.3f}→[new] to move MACs into the allowed band, prioritizing accuracy.",    "architecture_type": "vit",
        "isomorphic_group_ratios": {{
            "qkv_multiplier": YOUR_ADJUSTED_VALUE,
            "mlp_multiplier": YOUR_ADJUSTED_VALUE,
            "proj_multiplier": 0.0,
            "head_multiplier": 0.0
        }}
        }}

        CRITICAL: Make intelligent adjustments based on whether MACs were over budget (>+{overshoot_tol_pct:.1f}%) or too far under budget (< -{undershoot_tol_pct:.1f}%)."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a ViT pruning expert creating intelligent strategy variations."),
                HumanMessage(content=prompt)
            ])
            
            strategy_dict = parse_llm_json_response(response.content)
            
            if strategy_dict and 'isomorphic_group_ratios' in strategy_dict:
                ratios = strategy_dict['isomorphic_group_ratios']
                new_mlp = ratios.get('mlp_multiplier', 0)
                new_qkv = ratios.get('qkv_multiplier', 0)
                
                # print(f"[🤖] LLM ViT variation strategy:")
                # print(f"   Original: mlp={original_mlp:.3f}, qkv={original_qkv:.3f}")
                # print(f"   Variation: mlp={new_mlp:.3f}, qkv={new_qkv:.3f}")
                # print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                # print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")
                
                return strategy_dict
            else:
                # print(f"[⚠️] LLM ViT variation failed, using simple adjustment")
                return self._simple_vit_strategy_variation(original_strategy, previous_achieved, target_ratio)
                    
        except Exception as e:
            # print(f"[⚠️] LLM ViT variation failed: {e}")
            return self._simple_vit_strategy_variation(original_strategy, previous_achieved, target_ratio)

    def _simple_vit_strategy_variation(self, original_strategy, previous_achieved, target_ratio, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Simple mathematical variation when LLM fails for ViT"""
        
        original_ratios = original_strategy.get('isomorphic_group_ratios', {})
        original_mlp = original_ratios.get('mlp_multiplier', 0.6)
        original_qkv = original_ratios.get('qkv_multiplier', 0.3)
        
        if previous_achieved < target_ratio:
            # undershoot, increase multipliers (more aggressive on MLP)
            new_mlp = min(2.0, original_mlp * 1.3)  # 30% increase
            new_qkv = min(1.5, original_qkv * 1.2)  # 20% increase (more conservative)
            direction = "increased (undershoot)"
        else:
            # overshoot, decrease multipliers
            new_mlp = max(0.1, original_mlp * 0.8)  # 20% decrease
            new_qkv = max(0.1, original_qkv * 0.8)  # 20% decrease
            direction = "decreased (overshoot)"
        
        varied_strategy = original_strategy.copy()
        varied_strategy['isomorphic_group_ratios'] = {
            'qkv_multiplier': new_qkv,
            'mlp_multiplier': new_mlp,
            'proj_multiplier': 0.0,
            'head_multiplier': 0.0
        }
        varied_strategy['rationale'] = f"Simple ViT variation: mlp {original_mlp:.3f}→{new_mlp:.3f}, qkv {original_qkv:.3f}→{new_qkv:.3f} ({direction}). Tolerance: -{macs_undershoot_tolerance_pct:.1f}%/+{macs_overshoot_tolerance_pct:.1f}%"
        
        print(f"[🔧] Simple ViT variation: mlp {original_mlp:.3f}→{new_mlp:.3f}, qkv {original_qkv:.3f}→{new_qkv:.3f} ({direction})")
        
        return varied_strategy


    def _format_history_for_llm_learning(self, history, target_ratio, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        ENHANCED: CNN learning format with clear channel ratio pattern analysis
        """
        
        if not history:
            return "No historical data available for learning."
        
        overshoot_tolerance = macs_overshoot_tolerance_pct / 100.0
        undershoot_tolerance = macs_undershoot_tolerance_pct / 100.0
        
        formatted = []
        formatted.append(f"LEARNING FROM {len(history)} PREVIOUS ATTEMPTS:")
        formatted.append("=" * 60)
        
        # Track patterns for summary
        overshoots = 0
        undershoots = 0
        catastrophic_failures = 0  # Based on target distance, not accuracy
        successful_attempts = 0
        
        for i, entry in enumerate(history, 1):
            strategy = entry.get('strategy_used', {})
            channel_ratio = (strategy.get('channel_ratio_used') or 
                            strategy.get('channel_pruning_ratio') or 
                            strategy.get('channel_ratio'))
            achieved_ratio = entry.get('achieved_ratio', 0)
            target_was = entry.get('target_ratio', target_ratio)
            
            # Calculate deviation from target
            if achieved_ratio > 0:
                deviation = achieved_ratio - target_was
                distance_from_target = abs(deviation)
                
                # CORRECTED: Catastrophic = far from target, not low accuracy
                if distance_from_target > max(overshoot_tolerance, undershoot_tolerance):  # Use larger tolerance for catastrophic threshold
                    catastrophic_failures += 1
                    status = f"💥 CATASTROPHIC (missed target by {distance_from_target*100:.1f}%)"
                elif distance_from_target <= overshoot_tolerance:  # Within overshoot tolerance
                    successful_attempts += 1
                    status = "🎯 SUCCESS (within tolerance)"
                elif deviation > 0:
                    overshoots += 1
                    status = f"📈 OVERSHOOT (+{deviation*100:.1f}%)"
                else:
                    undershoots += 1
                    status = f"📉 UNDERSHOOT ({deviation*100:.1f}%)"
            else:
                catastrophic_failures += 1
                status = "💥 CATASTROPHIC (no parameter reduction achieved)"
            
            formatted.append(f"Attempt {i}: {status}")
            formatted.append(f"  Strategy: channel_ratio={channel_ratio}, importance={strategy.get('importance_criterion', 'unknown')}, round_to={strategy.get('round_to', 'unknown')}")
            formatted.append(f"  Result: {achieved_ratio*100:.1f}% parameter reduction (target: {target_was*100:.1f}%)")
            formatted.append(f"  Distance from target: {abs(achieved_ratio - target_was)*100:.1f}%")
            formatted.append("")
        
        # NEW: Channel ratio pattern analysis
        formatted.append("🔍 CHANNEL RATIO PATTERN ANALYSIS:")
        
        # Group by channel ratio to show clear patterns
        ratio_groups = {}
        for entry in history:
            strategy = entry.get('strategy_used', {})
            channel_ratio = (strategy.get('channel_ratio_used') or 
                            strategy.get('channel_pruning_ratio') or 
                            strategy.get('channel_ratio'))
            achieved_ratio = entry.get('achieved_ratio', 0)
            
            if channel_ratio is not None:
                if channel_ratio not in ratio_groups:
                    ratio_groups[channel_ratio] = []
                ratio_groups[channel_ratio].append(achieved_ratio * 100)
        
        # Show the dominant pattern
        for ratio, results in sorted(ratio_groups.items()):
            avg_result = sum(results) / len(results)
            formatted.append(f"  📊 Channel ratio {ratio} → Average: {avg_result:.1f}% reduction ({len(results)} attempts)")
            if len(results) >= 3:  # Only show details for frequently used ratios
                formatted.append(f"      Range: {min(results):.1f}% to {max(results):.1f}%")
        
        # The key learning insight
        if ratio_groups:
            most_used_ratio = max(ratio_groups.keys(), key=lambda k: len(ratio_groups[k]))
            most_used_avg = sum(ratio_groups[most_used_ratio]) / len(ratio_groups[most_used_ratio])
            most_used_count = len(ratio_groups[most_used_ratio])
            
            formatted.append(f"")
            formatted.append(f"💡 KEY PATTERN:")
            formatted.append(f"  🎯 TARGET: {target_ratio*100:.1f}% parameter reduction")
            formatted.append(f"  📈 OBSERVED: Ratio {most_used_ratio} → {most_used_avg:.1f}% reduction ({most_used_count} attempts)")
            formatted.append(f"  ❌ PROBLEM: {abs(most_used_avg - target_ratio*100):.1f}% away from target")
            formatted.append(f"")
            formatted.append(f"❓ LEARNING QUESTION:")
            formatted.append(f"  If channel_ratio {most_used_ratio} gives {most_used_avg:.1f}% reduction,")
            formatted.append(f"  what channel_ratio would give {target_ratio*100:.1f}% reduction?")
            


            if most_used_avg > target_ratio * 100:
                # Calculate what ratio SHOULD work
                suggested_ratio = most_used_ratio * (target_ratio * 100) / most_used_avg
                formatted.append(f"  💡 MATHEMATICAL SOLUTION:")
                formatted.append(f"     Current: {most_used_ratio} → {most_used_avg:.1f}% reduction")
                formatted.append(f"     Needed:  ? → {target_ratio*100:.1f}% reduction")
                formatted.append(f"     Formula: new_ratio = {most_used_ratio} × ({target_ratio*100:.1f}/{most_used_avg:.1f})")
                formatted.append(f"     Result:  new_ratio = {suggested_ratio:.3f}")
                formatted.append(f"  🎯 RECOMMENDATION: Try channel_ratio ≈ {suggested_ratio:.3f}")
            else:
                # For undershooting
                suggested_ratio = most_used_ratio * (target_ratio * 100) / most_used_avg
                formatted.append(f"  💡 MATHEMATICAL SOLUTION:")
                formatted.append(f"     Current: {most_used_ratio} → {most_used_avg:.1f}% reduction")
                formatted.append(f"     Needed:  ? → {target_ratio*100:.1f}% reduction")
                formatted.append(f"     Formula: new_ratio = {most_used_ratio} × ({target_ratio*100:.1f}/{most_used_avg:.1f})")
                formatted.append(f"     Result:  new_ratio = {suggested_ratio:.3f}")
                formatted.append(f"  🎯 RECOMMENDATION: Try channel_ratio ≈ {suggested_ratio:.3f}")


                formatted.append(f"  🎯 RECOMMENDATION: Try channel_ratio ≈ {suggested_ratio:.3f}")
                
                # ADD THIS NEW SECTION HERE:
                formatted.append(f"")
                formatted.append(f"🚨 CRITICAL UNDERSTANDING:")
                formatted.append(f"  USER'S GOAL: {target_ratio*100:.1f}% parameter reduction (NEVER CHANGE THIS)")
                formatted.append(f"  YOUR JOB: Find the RIGHT channel_pruning_ratio to achieve this goal")
                formatted.append(f"  PROBLEM: channel_ratio = {most_used_ratio} gives {most_used_avg:.1f}% reduction")
                formatted.append(f"  SOLUTION: Use channel_ratio = {suggested_ratio:.3f} to get {target_ratio*100:.1f}% reduction")
                formatted.append(f"")
                formatted.append(f"⚠️  DO NOT use channel_ratio = {target_ratio:.3f} just because target is {target_ratio*100:.1f}%!")
                formatted.append(f"⚠️  These are different things - channel_ratio is a TOOL to achieve the target!")
        
        formatted.append("")
        
        # Add learning summary focused on target hitting
        formatted.append("LEARNING SUMMARY:")
        formatted.append(f"  Target: {target_ratio*100:.0f}% parameter reduction")
        formatted.append(f"  Successful attempts (target met, ≤{macs_overshoot_tolerance_pct:.1f}% overshoot): {successful_attempts}/{len(history)}")
        formatted.append(f"  Overshoots: {overshoots}, Undershoots: {undershoots}")
        formatted.append(f"  Catastrophic failures (>2% from target): {catastrophic_failures}")
        
        # Add pattern insights focused on target achievement
        if overshoots > undershoots:
            formatted.append("  PATTERN: Tendency to overshoot target - try lower channel ratios")
        elif undershoots > overshoots:
            formatted.append("  PATTERN: Tendency to undershoot target - try higher channel ratios")
        else:
            formatted.append("  PATTERN: Mixed results - analyze other parameter effects")
        
        if catastrophic_failures > 0:
            formatted.append(f"  WARNING: {catastrophic_failures} attempts missed target by >2% - identify problematic parameter ranges")
        
        formatted.append("")
        formatted.append("LEARNING TASK:")
        formatted.append(f"Your ONLY goal is to achieve at least {target_ratio*100:.0f}% parameter reduction (up to {macs_overshoot_tolerance_pct:.1f}% overshoot allowed).")
        formatted.append("Accuracy is handled separately - focus purely on parameter reduction prediction.")

        formatted.append("")
        formatted.append("MATHEMATICAL LEARNING TASK:")
        formatted.append("Use the proportional relationship: new_ratio = old_ratio × (target_reduction / achieved_reduction)")
        formatted.append("Calculate the exact channel_ratio needed based on the observed pattern.")
        formatted.append("Show your mathematical reasoning in the rationale.")
        
        return "\n".join(formatted)

    async def _llm_calculate_full_strategy_with_history(self, target_ratio, model_name, history, dataset, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        Use LLM to determine complete strategy based on historical results
        This is the LEARNING version that gets smarter over time
        """
        
        # Format history for LLM analysis
        history_text = self._format_history_for_llm_learning(history, target_ratio, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
        
        prompt = f"""You are an expert in CNN pruning for {model_name} on {dataset}.

        CRITICAL LEARNING TASK: Analyze historical pruning attempts and determine the optimal strategy to achieve {target_ratio*100:.0f}% parameter reduction.

        HISTORICAL LEARNING DATA:
        {history_text}

        LEARNING OBJECTIVES:
        1. IDENTIFY PATTERNS: What channel ratios actually achieved what parameter reductions?
        2. LEARN RELATIONSHIPS: How does {model_name} respond to different channel pruning ratios?
        3. AVOID FAILURES: Which strategies led to catastrophic accuracy loss (<1% accuracy)?
        4. OPTIMIZE PARAMETERS: What combinations of importance_criterion + round_to worked best?
        5. PREDICT ACCURATELY: Based on the pattern, what channel ratio will hit {target_ratio*100:.0f}% exactly?

        ARCHITECTURE INSIGHTS FOR {model_name.upper()}:
        - The mathematical sqrt formula FAILS for real architectures due to skip connections
        - You must learn the ACTUAL relationship from the historical data above
        - Each architecture has unique scaling patterns that only emerge from real experiments

        LEARNING STRATEGY:
        - If previous attempts overshoot the target → suggest lower channel ratio
        - If previous attempts undershoot the target → suggest higher channel ratio
        - If accuracy collapsed → identify what caused it and avoid those parameters
        - If accuracy was preserved → learn what made that strategy successful

        ADVANCED LEARNING:
        - Look for convergence patterns in the data
        - Identify if certain importance_criterion values work better for this model/dataset
        - Learn optimal round_to values from previous hardware efficiency results
        - Extrapolate or interpolate based on the trend in the historical data

        🚨 CRITICAL DISTINCTION:
        - target_ratio = {target_ratio:.3f} (USER'S GOAL - never change this)
        - channel_pruning_ratio = ??? (TOOL to achieve the goal - this is what you calculate)

        WRONG THINKING: "User wants 20% so I'll use channel_ratio = 0.200"
        RIGHT THINKING: "User wants 20% but channel_ratio = 0.200 gives 31%, so I need channel_ratio ≈ 0.126"

        Your job is to find the RIGHT channel_pruning_ratio that produces the target parameter reduction.

        OUTPUT FORMAT (JSON only):
        {{
        "channel_pruning_ratio": 0.XXXX,
        "importance_criterion": "taylor|l1norm|l2norm",
        "round_to": X,
        "global_pruning": true,
        "parameter_tuning_order": ["importance_criterion", "channel_pruning_ratio", "round_to"],
        "rationale": "CRITICAL: channel_pruning_ratio is NOT the same as target_ratio! Based on {len(history)} attempts, I learned that channel_ratio X produces Y% parameter reduction. To achieve {target_ratio*100:.0f}% parameter reduction, I calculated channel_ratio = [show math] because [reasoning from data]."
        }}

        Learn from the data, don't guess. Base your predictions on observed patterns."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a CNN pruning expert learning from experimental data to improve your predictions."),
                HumanMessage(content=prompt)
            ])
            
            strategy_dict = parse_llm_json_response(response.content)
            
            if strategy_dict and 'channel_pruning_ratio' in strategy_dict:
                channel_ratio = float(strategy_dict['channel_pruning_ratio'])
                
                print(f"[🧠] LLM learned from {len(history)} attempts:")
                print(f"   Predicted channel ratio: {channel_ratio:.4f}")
                print(f"   Learned importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Learned round-to: {strategy_dict.get('round_to', 'unknown')}")
                
                # Show learning summary
                learning_summary = strategy_dict.get('rationale', '')[:200]
                print(f"   Learning insight: {learning_summary}...")
                
                return strategy_dict
            else:
                print(f"[⚠️] LLM failed to learn from history")
                raise ValueError("LLM historical learning parsing failed")
                
        except Exception as e:
            print(f"[❌] LLM historical learning failed: {e}")
            print(f"[🔄] Falling back to baseline strategy")
            
            # Fallback to baseline if learning fails
            return await self._llm_calculate_full_strategy_baseline(target_ratio, model_name, dataset)
        
    async def _llm_create_strategy_variation(self, original_strategy, previous_entry, target_ratio, model_name, dataset, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        LLM creates intelligent variation when strategy was already tried
        """
        
        previous_achieved = previous_entry.get('achieved_ratio', 0)
        original_channel = original_strategy.get('channel_pruning_ratio', 0.3)
        
        prompt = f"""You are a CNN pruning expert creating a variation of a strategy.

        SITUATION: The original strategy was already tried and needs adjustment.

        ORIGINAL STRATEGY:
        - Channel ratio: {original_channel:.4f}
        - Importance: {original_strategy.get('importance_criterion', 'unknown')}
        - Round-to: {original_strategy.get('round_to', 'unknown')}

        PREVIOUS RESULT:
        - Achieved: {previous_achieved*100:.1f}% parameter reduction
        - Target: {target_ratio*100:.1f}% parameter reduction

        TASK: Create a modified strategy that will get closer to the {target_ratio*100:.1f}% target.

        ANALYSIS:
        - If previous overshoot: Reduce channel ratio
        - If previous undershoot: Increase channel ratio  
        - Consider changing importance criterion or round_to if needed
        - Make smart adjustments based on {model_name} characteristics

        OUTPUT FORMAT (JSON only):
        {{
        "channel_pruning_ratio": 0.XXXX,
        "importance_criterion": "taylor|l1norm|l2norm",
        "round_to": X,
        "global_pruning": true,
        "parameter_tuning_order": ["importance_criterion", "channel_pruning_ratio", "round_to"],
        "rationale": "Variation strategy: explain the adjustments made and why"
        }}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a CNN pruning expert creating intelligent strategy variations."),
                HumanMessage(content=prompt)
            ])
            
            strategy_dict = parse_llm_json_response(response.content)
            
            if strategy_dict and 'channel_pruning_ratio' in strategy_dict:
                channel_ratio = float(strategy_dict['channel_pruning_ratio'])
                
                print(f"[🤖] LLM variation strategy:")
                print(f"   Original: {original_channel:.4f} → Variation: {channel_ratio:.4f}")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")
                
                return strategy_dict
            else:
                print(f"[⚠️] LLM variation failed, using simple adjustment")
                return self._simple_strategy_variation(original_strategy, previous_achieved, target_ratio, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
                
        except Exception as e:
            print(f"[⚠️] LLM variation failed: {e}")
            return self._simple_strategy_variation(original_strategy, previous_achieved, target_ratio, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)

    # COMPANION METHOD: _simple_strategy_variation (fallback when LLM fails)

    def _simple_strategy_variation(self, original_strategy, previous_achieved, target_ratio, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Simple mathematical variation when LLM fails"""
        
        original_channel = original_strategy.get('channel_pruning_ratio', 0.3)
        
        if previous_achieved > target_ratio:
            # overshoot, reduce by 10%
            new_channel = original_channel * 0.9
            direction = "reduced (overshoot)"
        else:
            # undershoot, increase by 10%
            new_channel = original_channel * 1.1
            direction = "increased (undershoot)"
        
        new_channel = max(0.1, min(0.7, new_channel))
        
        varied_strategy = original_strategy.copy()
        varied_strategy['channel_pruning_ratio'] = new_channel
        varied_strategy['rationale'] = f"Simple variation: {original_channel:.4f} → {new_channel:.4f} ({direction}). Tolerance: -{macs_undershoot_tolerance_pct:.1f}%/+{macs_overshoot_tolerance_pct:.1f}%"
        
        print(f"[🔧] Simple variation: {original_channel:.4f} → {new_channel:.4f} ({direction}). Tolerance: -{macs_undershoot_tolerance_pct:.1f}%/+{macs_overshoot_tolerance_pct:.1f}%")
        
        return varied_strategy


    """
    # Check if this exact strategy was tried before
    already_tried, previous_entry = self._is_strategy_already_tried(strategy, history, tolerance=0.02
    if already_tried:
        print(f"[🔄] Strategy already tried, requesting LLM variation")
        strategy = await self._llm_create_strategy_variation(
            strategy, previous_entry, target_ratio, model_name, dataset
        )
        approach += "_with_variation"
    """

    # WHAT THIS METHOD DOES:

    """
    1. ANALYZES PREVIOUS RESULT:
    - If previous attempt overshoot target → suggests lower channel ratio
    - If previous attempt undershoot target → suggests higher channel ratio

    2. LLM INTELLIGENCE:
    - Considers architecture-specific characteristics
    - May also adjust importance_criterion or round_to
    - Provides reasoning for the changes

    3. FALLBACK HANDLING:
    - If LLM fails → uses simple mathematical adjustment (±10%)
    - Always stays within safety bounds (0.1 to 0.7)

    4. EXAMPLE LLM REASONING:
    "Previous attempt with 0.35 channel ratio achieved 54% reduction but target was 50%.
    For ResNet50, I'll reduce to 0.31 because bottleneck blocks amplify pruning effects.
    Keeping taylor importance for ImageNet accuracy preservation."
    """

    # ENHANCED HISTORY FORMATTING FOR LEARNING

    def _determine_round_to(self, dataset):
        if dataset.lower() == 'imagenet':
            return random.choice([1, 2, 4, 8, 16, None])  # All possible workflow values
        else:
            return random.choice([1, 2, 4, 8, 16, None])  # All possible workflow values

    async def _call_llm_with_prompt(self, enhanced_prompt, state):
        """Call LLM with the enhanced prompt and robust JSON parsing"""
        
        # Add JSON formatting instructions to the prompt
        json_instructions = """

        CRITICAL JSON OUTPUT REQUIREMENTS:
        ================================
        - Output ONLY valid JSON with NO comments
        - Do NOT use # or // comments in JSON
        - No markdown code blocks (no ```)
        - Use double quotes for all strings
        - No trailing commas
        - No explanatory text before or after JSON

        Example format for ViT models:
        {
        "importance_criterion": "taylor",
        "pruning_ratio": 0.3,
        "round_to": 2,
        "global_pruning": true,
        "rationale": "Explanation without comments",
        "isomorphic_group_ratios": {
            "qkv_multiplier": 0.5,
            "mlp_multiplier": 1.0,
            "proj_multiplier": 0.0,
            "head_multiplier": 0.0
        }
        }

        Example format for CNNs:
        {
        "importance_criterion": "taylor",
        "channel_pruning_ratio": 0.3,
        "round_to": 2,
        "global_pruning": true,
        "rationale": "Using Taylor importance with 0.3 channel ratio to target 50% parameter reduction."
        }

        """
        
        final_prompt = enhanced_prompt + json_instructions
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=final_prompt),
                HumanMessage(content=state['query'])
            ])
            
            # print(f"[DEBUG] Raw Analysis Agent response:\n{response.content}")
            
            # Use robust JSON parsing
            strategy_dict = parse_llm_json_response(response.content)
            
            if not strategy_dict:
                # print(f"[❌] Analysis Agent JSON parsing failed - using fallback")
                return self.safety_validator.create_safe_fallback(
                    state.get('dataset'), 
                    state.get('target_pruning_ratio')
                )
            
            # print(f"[DEBUG] Parsed importance criterion: {strategy_dict.get('importance_criterion')}")
            print(f"[✅] Successfully parsed Analysis Agent response")
            
            return strategy_dict
            
        except Exception as e:
            # print(f"[⚠️] LLM call failed: {e}")
            import traceback
            # print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return self.safety_validator.create_safe_fallback(
                state.get('dataset'), 
                state.get('target_pruning_ratio')
            )

    async def _retry_with_corrections_feedback(self, original_prompt, corrected_strategy, state):
        """Re-prompt LLM with feedback about corrections that were needed"""
        
        corrections = corrected_strategy.get('corrections_applied', [])
        feedback_prompt = f"""
        {original_prompt}

        IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT:
        Your previous suggestion required these safety corrections:
        {chr(10).join(f"- {correction}" for correction in corrections)}

        Please provide a NEW recommendation that avoids these issues and stays within safety limits.
        Remember: The goal is to suggest ratios that achieve the target while maintaining model viability.

        CRITICAL FOR IMAGENET: Always use "taylor" importance criterion!
        """
        
        try:
            retry_response = await self.llm.ainvoke([
                SystemMessage(content=feedback_prompt),
                HumanMessage(content="Please provide corrected ratios based on the feedback.")
            ])
            
            # Parse and validate the retry
            clean_json = re.sub(r'```json\s*', '', retry_response.content)
            clean_json = re.sub(r'\s*```.*$', '', clean_json, flags=re.DOTALL)
            retry_strategy = json.loads(clean_json.strip())
            
            # Force Taylor for ImageNet in retry as well
            dataset = state.get('dataset', 'cifar10')
            if dataset.lower() == 'imagenet':
                retry_strategy['importance_criterion'] = "taylor"
                print(f"[🔧] Forced Taylor in retry for ImageNet")
            
            # Re-validate the retry attempt
            final_validated = self.safety_validator.validate_and_correct(
                retry_strategy, state.get('target_pruning_ratio'), 
                state.get('dataset'), state.get('history', [])
            )
            
            print(f"[✅] Retry attempt needed {len(final_validated.get('corrections_applied', []))} corrections")
            return final_validated
            
        except Exception as e:
            # print(f"[⚠️] Retry attempt failed: {e}, using original corrected strategy")
            return corrected_strategy
        
    
    def _is_strategy_already_tried(self, proposed_strategy, history, tolerance=0.002):
        """
        REASON: Prevent Analysis Agent from suggesting same failed strategy repeatedly
        Checks if proposed strategy was already attempted with similar parameters
        """
        
        proposed_channel_ratio = proposed_strategy.get('channel_pruning_ratio')
        proposed_round_to = proposed_strategy.get('round_to')
        proposed_importance = proposed_strategy.get('importance_criterion')
        
        if not proposed_channel_ratio:
            return False, None  # ViT strategies use different validation
        
        for entry in history:
            strategy_used = entry.get('strategy_used', {})
            used_channel_ratio = (strategy_used.get('channel_ratio_used') or strategy_used.get('channel_pruning_ratio') or  strategy_used.get('channel_ratio'))
            used_round_to = strategy_used.get('round_to')
            used_importance = strategy_used.get('importance_criterion')
            
            # Check if parameters are very similar
            if (used_channel_ratio and 
                abs(used_channel_ratio - proposed_channel_ratio) < tolerance and
                used_round_to == proposed_round_to and
                used_importance == proposed_importance):
                
                achieved = entry.get('achieved_ratio', 0)
                target = entry.get('target_ratio', 0)
                
                print(f"[🔄] STRATEGY REPETITION DETECTED:")
                print(f"   Previous: channel={used_channel_ratio:.4f}, round_to={used_round_to}, importance={used_importance}")
                print(f"   Proposed: channel={proposed_channel_ratio:.4f}, round_to={proposed_round_to}, importance={proposed_importance}")
                print(f"   Previous result: {achieved*100:.1f}% (target: {target*100:.1f}%)")
                return True, entry
        
        return False, None

    async def _llm_calculate_full_strategy_baseline(self, target_ratio, model_name, dataset, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        Use LLM to determine complete strategy for first attempt (NO HISTORY)
        """
        
        prompt = f"""You are an expert in CNN pruning for {model_name} on {dataset}.

        TASK: Determine the complete pruning strategy to achieve {target_ratio*100:.0f}% parameter reduction.

        This is the FIRST ATTEMPT - you have no historical data to learn from.
        Make your best educated guess based on architecture knowledge.

        ARCHITECTURE ANALYSIS FOR {model_name.upper()}:
        - ResNet: Has skip connections and bottleneck blocks that amplify pruning effects
        - MobileNet: Uses depthwise separable convolutions that are very sensitive to channel changes
        - EfficientNet: Has compound scaling that affects pruning relationships

        DATASET REQUIREMENTS ({dataset.upper()}):
        - ImageNet: 1000 classes, complex features, need accuracy preservation
        - CIFAR-10: 10 classes, simpler features, can be more aggressive

        OUTPUT FORMAT (JSON only):
        {{
        "channel_pruning_ratio": 0.XXXX,
        "importance_criterion": "taylor|l1norm|l2norm", 
        "round_to": X,
        "global_pruning": true,
        "parameter_tuning_order": ["importance_criterion", "channel_pruning_ratio", "round_to"],
        "rationale": "Architecture-specific reasoning for {model_name} on {dataset}"
        }}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a CNN pruning expert making your first strategy attempt."),
                HumanMessage(content=prompt)
            ])
            
            strategy_dict = parse_llm_json_response(response.content)
            
            if strategy_dict and 'channel_pruning_ratio' in strategy_dict:
                channel_ratio = float(strategy_dict['channel_pruning_ratio'])
                
                print(f"[🤖] LLM baseline strategy for {model_name}:")
                print(f"   Channel ratio: {channel_ratio:.4f}")
                print(f"   Importance: {strategy_dict.get('importance_criterion', 'unknown')}")
                print(f"   Round-to: {strategy_dict.get('round_to', 'unknown')}")
                
                return strategy_dict
            else:
                # print(f"[⚠️] LLM failed to provide valid baseline strategy")
                raise ValueError("LLM baseline strategy parsing failed")
                
        except Exception as e:
            print(f"[❌] LLM baseline strategy calculation failed: {e}")
            print(f"[🔄] Using hardcoded conservative fallback")
            
            # Conservative fallback when LLM completely fails
            if 'resnet' in model_name.lower():
                fallback_channel = target_ratio * 0.6
                fallback_importance = "taylor"
                fallback_round_to = 2 if dataset.lower() == 'imagenet' else 4
            elif 'mobilenet' in model_name.lower():
                fallback_channel = target_ratio * 0.7
                fallback_importance = "taylor"
                fallback_round_to = 2
            else:
                fallback_channel = target_ratio * 0.7
                fallback_importance = "taylor"
                fallback_round_to = 4
            
            return {
                "channel_pruning_ratio": fallback_channel,
                "importance_criterion": fallback_importance,
                "round_to": fallback_round_to,
                "global_pruning": True,
                "parameter_tuning_order": ["importance_criterion", "channel_pruning_ratio", "round_to"],
                "rationale": f"Conservative fallback for {model_name} on {dataset} due to LLM failure: {str(e)}",
                "fallback_used": True
            }
    
    def _create_emergency_fallback_strategy(self, target_ratio, model_name, dataset, error_msg, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        Emergency fallback when both LLM methods fail
        """
        
        print(f"[🚨] Emergency fallback for {model_name} on {dataset}")
        
        # Ultra-conservative approach based on model type
        if 'resnet' in model_name.lower():
            emergency_channel = target_ratio * 0.5  # Very conservative
            emergency_importance = "taylor"
            emergency_round_to = 2
        elif 'mobilenet' in model_name.lower():
            emergency_channel = target_ratio * 0.6
            emergency_importance = "taylor"
            emergency_round_to = 2
        else:
            emergency_channel = target_ratio * 0.55
            emergency_importance = "taylor"
            emergency_round_to = 2
        
        # print(f"[🔧] Emergency fallback: channel={emergency_channel:.4f}, {emergency_importance}, round_to={emergency_round_to}")
        
        return {
            "channel_pruning_ratio": emergency_channel,
            "importance_criterion": emergency_importance,
            "round_to": emergency_round_to,
            "global_pruning": True,
            "parameter_tuning_order": ["importance_criterion", "channel_pruning_ratio", "round_to"],
            "rationale": f"Emergency fallback for {model_name} on {dataset}. LLM failed: {error_msg}. Tolerance: -{macs_undershoot_tolerance_pct:.1f}%/+{macs_overshoot_tolerance_pct:.1f}%",
            "emergency_fallback": True,
            "llm_failed": True
        }
    

    def _detect_convergence_and_accept_best(self, history, target_ratio, current_revision, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        CORRECTED: Find model within tolerance that has highest fine-tuned accuracy
        REASON: User wants best performing model, not just closest to target
        """
        
        if current_revision < 5:  # Need minimum attempts to detect convergence
            return False, None
        
        # Analyze last 5 attempts for convergence
        recent_attempts = history[-5:] if len(history) >= 5 else history
        achieved_ratios = [entry.get('achieved_ratio', 0) for entry in recent_attempts]
        
        if len(achieved_ratios) < 3:
            return False, None
        
        # Calculate variance to detect if results are similar
        mean_achieved = sum(achieved_ratios) / len(achieved_ratios)
        variance = sum((x - mean_achieved) ** 2 for x in achieved_ratios) / len(achieved_ratios)
        std_dev = variance ** 0.5
        
        # Low variance = converged (similar results)
        if std_dev < 0.01:  # Less than 1% standard deviation
            print(f"[🔄] CONVERGENCE DETECTED:")
            print(f"   Recent results: {[f'{x*100:.1f}%' for x in achieved_ratios]}")
            print(f"   Standard deviation: {std_dev*100:.2f}%")
            print(f"   Mean achieved: {mean_achieved*100:.1f}% (target: {target_ratio*100:.1f}%)")
            
            # CORRECTED: Find models within tolerance first
            tolerance = max(macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct) / 100.0
            dataset = self._get_dataset_from_history(history)  # Helper to get dataset
            
            models_within_tolerance = []
            for entry in history:
                achieved = entry.get('achieved_ratio', 0)
                ratio_diff = achieved - target_ratio
                if ratio_diff >= 0 and ratio_diff <= tolerance:  # Must meet/exceed target, within tolerance
                    # Check if it has fine-tuned results
                    if dataset.lower() == 'imagenet':
                        ft_acc = entry.get('fine_tuned_top1_accuracy')
                    else:
                        ft_acc = entry.get('fine_tuned_accuracy')
                    
                    if ft_acc is not None:  # Has been fine-tuned
                        models_within_tolerance.append(entry)
            
            if models_within_tolerance:
                # CORRECTED: Select the one with highest fine-tuned accuracy
                if dataset.lower() == 'imagenet':
                    best_entry = max(models_within_tolerance, 
                                key=lambda x: x.get('fine_tuned_top1_accuracy', 0))
                    best_ft_acc = best_entry.get('fine_tuned_top1_accuracy', 0)
                    acc_type = "Top-1"
                else:
                    best_entry = max(models_within_tolerance, 
                                key=lambda x: x.get('fine_tuned_accuracy', 0))
                    best_ft_acc = best_entry.get('fine_tuned_accuracy', 0)
                    acc_type = "accuracy"
                
                best_achieved = best_entry.get('achieved_ratio', 0)
                print(f"[✅] BEST WITHIN TOLERANCE BY FINE-TUNED ACCURACY:")
                print(f"   Parameter reduction: {best_achieved*100:.1f}%")
                print(f"   Fine-tuned {acc_type}: {best_ft_acc:.2f}%")
                print(f"   Selected from {len(models_within_tolerance)} candidates within tolerance")
                
                return True, best_entry
            
            else:
                # No models within tolerance have been fine-tuned yet
                # This means convergence detected but no valid candidates
                print(f"[⚠️] Convergence detected but no fine-tuned models within tolerance")
                print(f"[🔍] Continuing search for models within tolerance...")
                return False, None
        
        return False, None

    def _get_dataset_from_history(self, history):
        """Helper to extract dataset from history entries"""
        for entry in history:
            dataset = entry.get('dataset')
            if dataset:
                return dataset
        return 'cifar10'  # fallback