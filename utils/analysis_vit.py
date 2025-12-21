from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
from typing import Dict, List, Optional
import os
from llm.prompts import _format_mac_analysis_for_llm



class ViTLearningAnalyzer:
    """
    Enhanced learning system for ViT models that analyzes patterns and provides
    intelligent guidance to prevent undershooting/overshooting
    """

    TOLERANCE_OVERSHOOT  = 0.01   #  +1% above target is still OK (strict)
    TOLERANCE_UNDERSHOOT = 0.05   #  -5% below target is still OK (lenient)
    
    def __init__(self):
        self.min_data_points = 2  # Minimum attempts needed for pattern analysis

    def _get_high_accuracy_threshold(self, dataset: str) -> float:
        """
        Return a strict default accuracy threshold for 'catastrophic-failure'
        detection when the caller doesn’t pass one explicitly.
        """
        if dataset.lower() == "imagenet":
            return 1.0          # allow ≤ 1 pp absolute drop
        elif dataset.lower() in {"cifar10", "cifar-10"}:
            return 85.0
        return 90.0


    def analyze_vit_multiplier_patterns(self, history, dataset, target_macs=None, accuracy_threshold=None, baseline_macs=None, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):    
        """
        Comprehensive analysis of ViT multiplier patterns to guide LLM based on MAC constraints
        """

        if not history or len(history) < self.min_data_points:
            return self._create_baseline_guidance(target_macs, dataset)

        
        # Extract and analyze all attempts
        attempts = self._extract_vit_attempts(history, baseline_macs)
        
        if len(attempts) < self.min_data_points:
            return self._create_baseline_guidance(target_macs, dataset)

        if accuracy_threshold is None:
            accuracy_threshold = self._get_high_accuracy_threshold(dataset)
        
        # Analyze patterns (pass MAC targets and tolerance into analysis)
        pattern_analysis = self._analyze_multiplier_effectiveness(attempts, target_macs, accuracy_threshold, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
        trend_analysis = self._analyze_trends(attempts,target_macs)
        failure_analysis = self._analyze_failures(attempts, target_macs, dataset, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
        
        # Generate intelligent recommendations based on MAC constraints
        recommendations = self._generate_smart_recommendations(
            pattern_analysis, trend_analysis, failure_analysis, target_macs, dataset, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
        )
        
        return {
            'pattern_analysis': pattern_analysis,
            'trend_analysis': trend_analysis,
            'failure_analysis': failure_analysis,
            'recommendations': recommendations,
            'has_sufficient_data': True,
            'total_attempts': len(attempts)
        }
    
    
    def _extract_vit_attempts(self, history, baseline_macs=None):
        """Extract ViT-specific attempts from history - MAC-BASED VERSION"""
        attempts = []
        
        for entry in history:
            strategy = entry.get('strategy_used', {})
            
            # ✅ FIX: Get MAC allocation from the correct nested structure
            ratios = strategy.get('isomorphic_group_ratios', strategy)  # Fallback if nested
            if not ratios:
                # print(f"[⚠️] Skipping entry - no isomorphic ratios found")
                continue

            mlp_multiplier = ratios.get('mlp_multiplier')
            qkv_multiplier = ratios.get('qkv_multiplier')
            achieved_ratio = entry.get('achieved_ratio', 0)
            target_macs = entry.get('target_macs')
            achieved_macs = entry.get('achieved_macs') or entry.get('final_macs') or entry.get('macs_after')

        
            if mlp_multiplier is None or qkv_multiplier is None:
                # print(f"[⚠️] Skipping entry with incomplete multiplier data")
                continue
            
            # ✅ FIX: Get accuracy correctly (unchanged)
            accuracy = entry.get('zero_shot_top1_accuracy', entry.get('zero_shot_accuracy', 0))
            if accuracy is None:
                accuracy = 0
            

            # Calculate MAC deviation if available  
            mac_deviation = achieved_macs - target_macs if (achieved_macs and target_macs) else 0
            mac_deviation_pct = (mac_deviation / target_macs) * 100 if target_macs else 0
            mac_efficiency = achieved_macs / baseline_macs if baseline_macs and achieved_macs else 1.0

            attempts.append({
                'mlp_multiplier': float(mlp_multiplier),
                'qkv_multiplier': float(qkv_multiplier),
                'achieved_ratio': float(achieved_ratio),    
                'achieved_macs': float(achieved_macs) if achieved_macs else 0,
                'target_macs': float(target_macs) if target_macs else 0,
                'mac_deviation': float(mac_deviation),
                'mac_deviation_pct': float(mac_deviation_pct),
                'mac_efficiency': float(mac_efficiency),
                'revision': entry.get('revision', 0),
                'importance': strategy.get('importance_criterion', 'unknown'),
                'round_to': strategy.get('round_to', 'unknown'),
                'accuracy': float(accuracy)
            })
        
        # ✅ FIX: Sort by revision to get chronological order (unchanged)
        attempts.sort(key=lambda x: x['revision'])
        # print(f"[📊] Extracted {len(attempts)} valid attempts in chronological order")
        
        return attempts


    def _analyze_multiplier_effectiveness(self, attempts: List[dict], target_macs: float, accuracy_threshold: Optional[float], macs_overshoot_tolerance_pct: float, macs_undershoot_tolerance_pct: float) -> Dict:
        """
        Evaluate how different (mlp,qkv) MAC allocation combos perform and
        flag high-accuracy near-misses.
        """
        # ------------------------------------------------------------------
        # 1. Resolve the threshold   (user-supplied → dataset default)
        # ------------------------------------------------------------------
        if accuracy_threshold is None:
            dataset_hint = attempts[0].get("dataset", "cifar10") if attempts else "cifar10"
            accuracy_threshold = self._get_high_accuracy_threshold(dataset_hint)
        # ------------------------------------------------------------------
        # 2. MAC-based stats (updated from ratio-based)
        analysis = {
            'avg_mlp_multiplier': 0,  # ✅ Clean and consistent
            'avg_qkv_multiplier': 0,  # ✅ Clean and consistent
            'avg_achieved_ratio': 0,
            'avg_achieved_macs': 0,
            'avg_mac_deviation': 0,
            'avg_mac_deviation_pct': 0,
            'avg_mac_efficiency': 0,
            'systematic_undershoot': False,
            'systematic_overshoot': False,
            'closest_attempt': None,
            'most_aggressive': None,
            'most_conservative': None,
            'has_excellent_near_miss': False,
            'high_accuracy_undershoots': []
        }

        if not attempts:
            return analysis
        # Calculate multiplier-based averages
        analysis['avg_mlp_multiplier'] = sum(a['mlp_multiplier'] for a in attempts) / len(attempts)
        analysis['avg_qkv_multiplier'] = sum(a['qkv_multiplier'] for a in attempts) / len(attempts)
        analysis['avg_achieved_ratio'] = sum(a['achieved_ratio'] for a in attempts) / len(attempts)
        analysis['avg_achieved_macs'] = sum(a['achieved_macs'] for a in attempts) / len(attempts)
        analysis['avg_mac_deviation'] = sum(a['mac_deviation'] for a in attempts) / len(attempts)
        analysis['avg_mac_deviation_pct'] = sum(a['mac_deviation_pct'] for a in attempts) / len(attempts)
        analysis['avg_mac_efficiency'] = sum(a.get('mac_efficiency', 1.0) for a in attempts) / len(attempts)

        analysis['all_attempts'] = attempts  # Store attempts for later use

        # Analyze undershoot/overshoot patterns using MAC deviation
        undershoots = [a for a in attempts if a.get('mac_deviation', 0) > 0]   # too heavy -> need more pruning
        overshoots  = [a for a in attempts if a.get('mac_deviation', 0) < 0]   # too light -> pruned too hard

        analysis['systematic_undershoot'] = len(undershoots) >= len(attempts) * 0.8
        analysis['systematic_overshoot'] = len(overshoots) >= len(attempts) * 0.8

        # Find key attempts based on MAC metrics
        analysis['closest_attempt'] = min(attempts, key=lambda x: abs(x['mac_deviation']))
        analysis['most_aggressive'] = max(attempts, key=lambda x: x['mlp_multiplier'] + x['qkv_multiplier'])
        analysis['most_conservative'] = min(attempts, key=lambda x: x['mlp_multiplier'] + x['qkv_multiplier'])

        # ------------------------------------------------------------------
        # 3. Detect high-accuracy small undershoots using MAC thresholds
        # ------------------------------------------------------------------
        if analysis['systematic_undershoot']:
            # Convert tolerance to MAC deviation threshold  
            mac_tolerance_threshold = target_macs * (macs_overshoot_tolerance_pct / 100) if target_macs else 0.1
            
            high_acc_attempts = [
                a for a in attempts
                if 0 < a['mac_deviation'] < mac_tolerance_threshold  # Small MAC undershoot
                and a.get('accuracy', 0) > accuracy_threshold
            ]
            analysis['high_accuracy_undershoots'] = high_acc_attempts
            analysis['has_excellent_near_miss'] = bool(high_acc_attempts)

            # ---- MAC-based debug prints -------------------------
            if analysis['has_excellent_near_miss']:
                print(f"[🎯] DETECTED: {len(high_acc_attempts)} excellent near-miss attempts")
                best_near_miss = max(high_acc_attempts, key=lambda x: x.get('accuracy', 0))
                print(f"[🎯] Best: {best_near_miss.get('accuracy', 0):.2f}% accuracy, "
                        f"{best_near_miss.get('mac_deviation', 0):.3f}G MAC deviation, "
                        f"{best_near_miss.get('mac_deviation_pct', 0):.2f}% MAC deviation")

        return analysis

    
    
    def _analyze_trends(self, attempts, target_macs):
        """Analyze trends in MAC allocation adjustments"""
        if len(attempts) < 3:
            return {'insufficient_data': True}
        
        # Sort by revision to see progression
        sorted_attempts = sorted(attempts, key=lambda x: x['revision'])
        
        trend_analysis = {
            'mlp_multiplier_trend': self._calculate_trend([a['mlp_multiplier'] for a in sorted_attempts]),
            'qkv_multiplier_trend': self._calculate_trend([a['qkv_multiplier'] for a in sorted_attempts]),
            'achieved_mac_trend': self._calculate_trend([a['achieved_macs'] for a in sorted_attempts]),
            'mac_deviation_trend': self._calculate_trend([a['mac_deviation'] for a in sorted_attempts]),
            'mac_deviation_pct_trend': self._calculate_trend([a['mac_deviation_pct'] for a in sorted_attempts]),
            'mac_efficiency_trend': self._calculate_trend([a.get('mac_efficiency', 1.0) for a in sorted_attempts]),
            'is_converging': False,
            'direction_consistency': self._check_direction_consistency(sorted_attempts),
            'target_approach_rate': 0.0
        }
        
        # Check if we're getting closer to MAC target
        recent_mac_deviations = [abs(a['mac_deviation']) for a in sorted_attempts[-3:]]
        trend_analysis['is_converging'] = len(recent_mac_deviations) >= 2 and recent_mac_deviations[-1] < recent_mac_deviations[0]
        
        # Calculate how quickly we're approaching the target MAC
        if len(recent_mac_deviations) >= 2:
            improvement = recent_mac_deviations[0] - recent_mac_deviations[-1]
            trend_analysis['target_approach_rate'] = improvement / recent_mac_deviations[0] if recent_mac_deviations[0] > 0 else 0.0
        
        # Additional MAC-specific trend analysis
        if target_macs:
            # Analyze how close we're getting to target MAC in absolute terms
            recent_achievements = [a['achieved_macs'] for a in sorted_attempts[-3:]]
            target_distances = [abs(mac - target_macs) for mac in recent_achievements]
            
            trend_analysis['target_mac_convergence'] = len(target_distances) >= 2 and target_distances[-1] < target_distances[0]
            trend_analysis['avg_recent_mac_distance'] = sum(target_distances) / len(target_distances)
        
        return trend_analysis

    
    def _calculate_trend(self, values):
        """Calculate if values are increasing, decreasing, or stable"""
        if len(values) < 2:
            return 'insufficient_data'
        
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_change = sum(differences) / len(differences)
        
        if abs(avg_change) < 0.01:
            return 'stable'
        elif avg_change > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _check_direction_consistency(self, sorted_attempts):
        """
        Evaluate whether successive MAC allocation tweaks move
        the achieved MAC in the correct direction.
        """
        if len(sorted_attempts) < 3:
            return 'insufficient_data'

        # MAC tolerance thresholds (in GigaOps)
        MAC_TOLERANCE_UNDERSHOOT = getattr(self, 'MAC_TOLERANCE_UNDERSHOOT', -0.5)
        MAC_TOLERANCE_OVERSHOOT = getattr(self, 'MAC_TOLERANCE_OVERSHOOT', 0.5)

        correct_moves = 0
        total_moves   = 0

        for i in range(1, len(sorted_attempts)):
            prev_attempt   = sorted_attempts[i - 1]
            curr_attempt   = sorted_attempts[i]

            prev_mac_dev   = prev_attempt['mac_deviation']
            prev_total_mac = prev_attempt['mlp_multiplier'] + prev_attempt['qkv_multiplier']
            curr_total_mac = curr_attempt['mlp_multiplier'] + curr_attempt['qkv_multiplier']

            # ---------- right-direction test for MAC allocation ----------
            if prev_mac_dev < MAC_TOLERANCE_UNDERSHOOT:  # Significantly under target
                if curr_total_mac > prev_total_mac:
                    correct_moves += 1

            elif prev_mac_dev > MAC_TOLERANCE_OVERSHOOT:  # Significantly over target
                if curr_total_mac < prev_total_mac:
                    correct_moves += 1

            total_moves += 1

        if total_moves == 0:
            return 'no_moves'

        ratio = correct_moves / total_moves
        if ratio >= 0.80:
            return 'highly_consistent'
        elif ratio >= 0.60:
            return 'moderately_consistent'
        else:
            return 'inconsistent'

    def _analyze_failures(self, attempts, target_macs, dataset, macs_overshoot_tolerance_pct: float, macs_undershoot_tolerance_pct: float):
        """Analyze specific failure patterns"""
        tolerance = 0.02  # For ratio-based failures
        
        failures = []
        catastrophic_failures = []
        
        for attempt in attempts:
            # Check MAC-based failures with asymmetric tolerance
            mac_deviation_pct = attempt['mac_deviation_pct']
            
            if mac_deviation_pct > macs_overshoot_tolerance_pct:  # MAC undershoot (too heavy, exceeded overshoot limit)
                failures.append({
                    'type': 'mac_undershoot',
                    'severity': mac_deviation_pct - macs_overshoot_tolerance_pct,
                    'multipliers': (attempt['mlp_multiplier'], attempt['qkv_multiplier']),
                    'attempt': attempt
                })
            elif mac_deviation_pct < -macs_undershoot_tolerance_pct:  # MAC overshoot (too light, exceeded undershoot limit)
                failures.append({
                    'type': 'mac_overshoot',
                    'severity': abs(mac_deviation_pct) - macs_undershoot_tolerance_pct,
                    'multipliers': (attempt['mlp_multiplier'], attempt['qkv_multiplier']),
                    'attempt': attempt
                })
            
            # Check ratio-based failures (if ratio data exists)
            if 'achieved_ratio' in attempt:
                ratio_deviation = attempt['achieved_ratio'] - (attempt.get('target_ratio', 0.2))
                if ratio_deviation < 0:  # Ratio undershoot
                    failures.append({
                        'type': 'ratio_undershoot',
                        'severity': abs(ratio_deviation),
                        'multipliers': (attempt['mlp_multiplier'], attempt['qkv_multiplier']),
                        'attempt': attempt
                    })
                elif ratio_deviation > tolerance:  # Ratio overshoot beyond tolerance
                    failures.append({
                        'type': 'ratio_overshoot',
                        'severity': ratio_deviation - tolerance,
                        'multipliers': (attempt['mlp_multiplier'], attempt['qkv_multiplier']),
                        'attempt': attempt
                    })
            
            # Check for catastrophic accuracy failures
            accuracy = attempt.get('accuracy', 0)
            if dataset.lower() == 'imagenet':
                if accuracy < 10:  # Less than 10% is catastrophic for ImageNet
                    catastrophic_failures.append(attempt)
            else:
                if accuracy < 30:  # Less than 30% is catastrophic for CIFAR-10
                    catastrophic_failures.append(attempt)
        
        return {
            'total_failures': len(failures),
            'mac_undershoot_failures': [f for f in failures if f['type'] == 'mac_undershoot'],
            'mac_overshoot_failures': [f for f in failures if f['type'] == 'mac_overshoot'],
            'ratio_undershoot_failures': [f for f in failures if f['type'] == 'ratio_undershoot'],
            'ratio_overshoot_failures': [f for f in failures if f['type'] == 'ratio_overshoot'],
            'catastrophic_failures': catastrophic_failures,
            'worst_undershoot': max(failures, key=lambda x: x['severity']) if failures else None,
            'avg_undershoot_severity': sum(f['severity'] for f in failures if 'undershoot' in f['type']) / max(1, len([f for f in failures if 'undershoot' in f['type']]))
        }

    def _generate_smart_recommendations(self, pattern_analysis, trend_analysis, failure_analysis, target_macs, dataset, macs_overshoot_tolerance_pct: float, macs_undershoot_tolerance_pct: float):
        """Generate intelligent recommendations based on MAC-based analyses"""
        recommendations = {
            'suggested_mlp_multiplier': 0.0,
            'suggested_qkv_multiplier': 0.0,
            'confidence': 'low',
            'reasoning': [],
            'mathematical_calculation': {},
            'safety_constraints': [],
            'learning_insights': []
        }
        
        # Calculate MAC tolerance threshold
        mac_tolerance_threshold = target_macs * (macs_overshoot_tolerance_pct / 100) if target_macs else 0.1
        
        # CRITICAL: Handle systematic undershooting (achieved MAC < target MAC)
        if pattern_analysis['systematic_undershoot']:
            undershoot_severity = abs(pattern_analysis['avg_mac_deviation'])
            
            # CHECK FOR HIGH-ACCURACY SMALL UNDERSHOOTS FIRST
            if pattern_analysis['has_excellent_near_miss']:
                high_acc_attempts = pattern_analysis['high_accuracy_undershoots']
                
                recommendations['reasoning'].append(f"HIGH-ACCURACY SMALL MAC UNDERSHOOT DETECTED: Found {len(high_acc_attempts)} attempts with excellent accuracy but tiny MAC undershoot")
                
                # Find the best high-accuracy attempt
                best_high_acc = max(high_acc_attempts, key=lambda x: x.get('accuracy', 0))
                
                best_achieved_mac = best_high_acc.get('achieved_macs', 0)
                best_acc = best_high_acc.get('accuracy', 0)
                best_mlp_multiplier = best_high_acc.get('mlp_multiplier', pattern_analysis['avg_mlp_multiplier'])
                best_qkv_multiplier = best_high_acc.get('qkv_multiplier', pattern_analysis['avg_qkv_multiplier'])
                best_mac_deviation = best_high_acc.get('mac_deviation', 0)
                
                # Very small correction needed - preserve the excellent strategy
                mac_correction_factor = 1.05  # Only 5% increase to preserve accuracy
                
                # Apply safety constraints for MAC allocation
                if dataset.lower() == 'imagenet':
                    max_mlp_multiplier = 15.0  # 15% of total MAC for MLP layers (ImageNet)
                    max_qkv_multiplier = 8.0   # 8% of total MAC for attention layers (ImageNet)
                else:
                    max_mlp_multiplier = 50.0  # 50% of total MAC for MLP layers (CIFAR-10)
                    max_qkv_multiplier = 40.0  # 40% of total MAC for attention layers (CIFAR-10)
                
                # Apply gentle correction
                corrected_mlp_multiplier = min(best_mlp_multiplier * mac_correction_factor, max_mlp_multiplier)
                corrected_qkv_multiplier = min(best_qkv_multiplier * mac_correction_factor, max_qkv_multiplier)
                
                recommendations['suggested_mlp_multiplier'] = corrected_mlp_multiplier
                recommendations['suggested_qkv_multiplier'] = corrected_qkv_multiplier
                recommendations['confidence'] = 'very_high'
                
                recommendations['reasoning'].append(f"TINY MAC ADJUSTMENT STRATEGY: Best attempt achieved {best_achieved_mac:.2f}G MAC with {best_acc:.1f}% accuracy")
                recommendations['reasoning'].append(f"MAC deviation was only {abs(best_mac_deviation):.3f}G - applying minimal {mac_correction_factor}x correction")
                
                recommendations['mathematical_calculation'] = {
                    'strategy_type': 'high_accuracy_tiny_mac_adjustment',
                    'best_accuracy_found': best_acc,
                    'best_achieved_mac_g': best_achieved_mac,
                    'tiny_mac_deviation': best_mac_deviation,
                    'mac_correction_factor_applied': mac_correction_factor,
                    'original_mlp_multiplier': best_mlp_multiplier,
                    'original_qkv_multiplier': best_qkv_multiplier,
                    'corrected_mlp_multiplier': corrected_mlp_multiplier,
                    'corrected_qkv_multiplier': corrected_qkv_multiplier,
                    'preservation_priority': 'accuracy_preservation'
                }
                
                recommendations['learning_insights'].append(f"EXCELLENT RESULT FOUND: {best_acc:.1f}% accuracy shows MAC allocation strategy is nearly perfect")
                recommendations['learning_insights'].append(f"Strategy: importance={best_high_acc.get('importance', 'unknown')}, round_to={best_high_acc.get('round_to', 'unknown')} works excellently")
                recommendations['learning_insights'].append(f"Only {mac_correction_factor}x MAC allocation increase needed - DO NOT change strategy drastically")
                
                return recommendations
            
            # REGULAR SYSTEMATIC UNDERSHOOT HANDLING
            recommendations['reasoning'].append(f"SYSTEMATIC MAC UNDERSHOOT DETECTED: Average MAC deviation {pattern_analysis['avg_mac_deviation']:.3f}G")
            
            # Calculate correction factor based on MAC undershoot severity
            if undershoot_severity > mac_tolerance_threshold * 2:  # Severe undershoot
                mac_correction_factor = 2.0  # Aggressive correction
                recommendations['confidence'] = 'high'
                recommendations['reasoning'].append(f"Severe MAC undershoot ({undershoot_severity:.3f}G) requires aggressive correction factor {mac_correction_factor}")
            elif undershoot_severity > mac_tolerance_threshold:  # Moderate undershoot
                mac_correction_factor = 1.5
                recommendations['confidence'] = 'medium'
                recommendations['reasoning'].append(f"Moderate MAC undershoot requires correction factor {mac_correction_factor}")
            else:
                mac_correction_factor = 1.2
                recommendations['confidence'] = 'medium'
                recommendations['reasoning'].append(f"Minor MAC undershoot requires correction factor {mac_correction_factor}")
            
            # Apply correction to average MAC multiplierages
            base_mlp_multiplier = pattern_analysis['avg_mlp_multiplier'] * mac_correction_factor
            base_qkv_multiplier = pattern_analysis['avg_qkv_multiplier'] * mac_correction_factor
            
            # Apply safety constraints
            if dataset.lower() == 'imagenet':
                max_mlp_multiplier = 15.0  # 15% of total MAC for MLP layers (ImageNet)
                max_qkv_multiplier = 8.0   # 8% of total MAC for attention layers (ImageNet)
            else:
                max_mlp_multiplier = 50.0  # 50% of total MAC for MLP layers (CIFAR-10)
                max_qkv_multiplier = 40.0  # 40% of total MAC for attention layers (CIFAR-10)
            
            # Apply safety constraints
            recommendations['suggested_mlp_multiplier'] = min(base_mlp_multiplier, max_mlp_multiplier)
            recommendations['suggested_qkv_multiplier'] = min(base_qkv_multiplier, max_qkv_multiplier)
            
            recommendations['mathematical_calculation'] = {
                'avg_mlp_multiplier_before': pattern_analysis['avg_mlp_multiplier'],
                'avg_qkv_multiplier_before': pattern_analysis['avg_qkv_multiplier'],
                'mac_correction_factor': mac_correction_factor,
                'base_mlp_multiplier_after_correction': base_mlp_multiplier,
                'base_qkv_multiplier_after_correction': base_qkv_multiplier,
                'safety_capped_mlp_multiplier': recommendations['suggested_mlp_multiplier'],
                'safety_capped_qkv_multiplier': recommendations['suggested_qkv_multiplier']
            }
            
            recommendations['learning_insights'].append(f"Historical average achieved only {pattern_analysis['avg_achieved_macs']:.2f}G vs target {target_macs:.2f}G")
            recommendations['learning_insights'].append(f"MAC allocation multiplierages need {mac_correction_factor}x increase to reach target")
        
        # Handle systematic overshoot with accuracy awareness (achieved MAC > target MAC)
        elif pattern_analysis['systematic_overshoot']:
            avg_mac_overshoot = pattern_analysis['avg_mac_deviation']
            
            # CHECK FOR HIGH-ACCURACY OVERSHOOTS (WITHIN REASONABLE RANGE)
            good_overshoots = []
            for attempt in pattern_analysis.get('all_attempts', []):  # Need to pass this from calling function
                mac_deviation = attempt.get('mac_deviation', 0)
                accuracy = attempt.get('accuracy', 0)
                
                if mac_deviation > (target_macs * (macs_overshoot_tolerance_pct / 100)) and mac_deviation <= (target_macs * (macs_overshoot_tolerance_pct / 100) * 2):  # Overshoot beyond overshoot tolerance but not excessive
                    if ((target_macs and target_macs <= 5.0 and accuracy > 1.0) or  # ImageNet-like (low MAC target)
                        (target_macs and target_macs > 5.0 and accuracy > 80)):      # CIFAR-10-like (higher MAC target)
                        good_overshoots.append(attempt)
            
            if good_overshoots:
                # GENTLE REDUCTION FOR HIGH-ACCURACY OVERSHOOTS
                recommendations['reasoning'].append(f"HIGH-ACCURACY MAC OVERSHOOT DETECTED: Found {len(good_overshoots)} attempts with good accuracy but slight overshoot")
                
                best_overshoot = max(good_overshoots, key=lambda x: x.get('accuracy', 0))
                
                best_achieved_mac = best_overshoot.get('achieved_mac_g', 0)
                best_acc = best_overshoot.get('accuracy', 0)
                best_mlp_multiplier = best_overshoot.get('mlp_multiplier', pattern_analysis['avg_mlp_multiplier'])
                best_qkv_multiplier = best_overshoot.get('qkv_multiplier', pattern_analysis['avg_qkv_multiplier'])
                
                # Gentle reduction to bring into tolerance
                target_mac_with_tolerance = target_macs + (mac_tolerance_threshold * 0.5) if target_macs else best_achieved_mac * 0.95
                mac_reduction_ratio = (best_achieved_mac - target_mac_with_tolerance) / best_achieved_mac if best_achieved_mac > 0 else 0.05
                mac_reduction_factor = max(0.85, 1.0 - mac_reduction_ratio)  # Don't reduce by more than 15%
                
                recommendations['suggested_mlp_multiplier'] = best_mlp_multiplier * mac_reduction_factor
                recommendations['suggested_qkv_multiplier'] = best_qkv_multiplier * mac_reduction_factor
                recommendations['confidence'] = 'high'
                
                recommendations['reasoning'].append(f"GENTLE MAC REDUCTION STRATEGY: Best overshoot achieved {best_achieved_mac:.2f}G with {best_acc:.1f}% accuracy")
                recommendations['reasoning'].append(f"Using {mac_reduction_factor:.2f}x reduction to bring MAC into tolerance while preserving accuracy")
                
                recommendations['learning_insights'].append(f"Good accuracy ({best_acc:.1f}%) shows MAC allocation strategy works - need gentle reduction to hit tolerance")
                recommendations['learning_insights'].append(f"Preserving high-performing strategy with minimal MAC adjustment")
                
            else:
                # AGGRESSIVE OVERSHOOT - original handling
                mac_reduction_factor = 0.7 if avg_mac_overshoot > mac_tolerance_threshold * 2 else 0.8
                
                recommendations['suggested_mlp_multiplier'] = pattern_analysis['avg_mlp_multiplier'] * mac_reduction_factor
                recommendations['suggested_qkv_multiplier'] = pattern_analysis['avg_qkv_multiplier'] * mac_reduction_factor
                recommendations['confidence'] = 'medium'
                recommendations['reasoning'].append(f"Systematic aggressive MAC overshoot requires {mac_reduction_factor} reduction factor")
                recommendations['learning_insights'].append(f"Aggressive MAC overshoot detected - significant reduction needed")

        # Handle mixed results - use closest attempt as baseline
        else:
            closest = pattern_analysis['closest_attempt']
            if closest:
                # Small adjustment based on how close we were in MAC terms
                mac_adjustment = 0.90 if closest['mac_deviation'] < 0 else 1.10
                
                recommendations['suggested_mlp_multiplier'] = closest['mlp_multiplier'] * mac_adjustment
                recommendations['suggested_qkv_multiplier'] = closest['qkv_multiplier'] * mac_adjustment
                recommendations['confidence'] = 'medium'
                recommendations['reasoning'].append(f"Using closest MAC attempt (deviation: {closest['mac_deviation']:.3f}G) with {mac_adjustment} adjustment")
        
        return recommendations


    def _create_baseline_guidance(self, target_macs, dataset):
        """Create baseline MAC allocation guidance when insufficient historical data"""

        baseline_importance = 'taylor'

        if dataset.lower() == 'imagenet':
            # Conservative baseline for ImageNet - allocate smaller multiplierages of total MAC
            baseline_mlp_multiplier = 8.0   # 8% of total MAC for MLP layers
            baseline_qkv_multiplier = 5.0   # 5% of total MAC for attention layers
        else:
            # More aggressive baseline for CIFAR-10 - can allocate higher multiplierages
            baseline_mlp_multiplier = 40.0  # 40% of total MAC for MLP layers
            baseline_qkv_multiplier = 30.0  # 30% of total MAC for attention layers
        
        # Calculate expected MAC allocation in GigaOps
        expected_mlp_mac = target_macs * (baseline_mlp_multiplier / 100)
        expected_qkv_mac = target_macs * (baseline_qkv_multiplier / 100)
        
        return {
            'pattern_analysis': None,
            'trend_analysis': None,
            'failure_analysis': None,
            'recommendations': {
                'suggested_mlp_multiplier': baseline_mlp_multiplier,
                'suggested_qkv_multiplier': baseline_qkv_multiplier,
                'suggested_importance_criterion': baseline_importance,
                'confidence': 'low',
                'reasoning': [f"Research-based baseline for {dataset}: Using Taylor importance (SOTA with isomorphic pruning)"],
                'mathematical_calculation': {
                    'baseline_approach': True,
                    'target_mac_g': target_macs,
                    'suggested_mlp_multiplier': baseline_mlp_multiplier,
                    'suggested_qkv_multiplier': baseline_qkv_multiplier,
                    'expected_mlp_mac_g': expected_mlp_mac,
                    'expected_qkv_mac_g': expected_qkv_mac,
                    'total_allocated_multiplier': baseline_mlp_multiplier + baseline_qkv_multiplier
                },
                'learning_insights': [
                    f"Conservative MAC allocation for {dataset}: MLP {baseline_mlp_multiplier}%, QKV {baseline_qkv_multiplier}%",
                    f"Expected MAC usage: MLP {expected_mlp_mac:.2f}G, QKV {expected_qkv_mac:.2f}G"
                ]
            },
            'has_sufficient_data': False,
            'total_attempts': 0
        }

    def create_enhanced_learning_prompt(self, model_name, history, dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0, strategic_guidance=None):
        """Create an enhanced prompt with comprehensive MAC-based learning analysis"""

        round_to_instruction = """Use round_to=2 as the standard choice because:
    1. PRECISION: Finer granularity allows more precise MAC targeting
    2. ACCURACY PRESERVATION: Smaller pruning increments reduce risk of removing critical channel combinations
    3. FINE-GRAINED CONTROL: Enables hitting exact MAC budgets more accurately
    4. ITERATIVE REFINEMENT: Easier to fine-tune results with smaller steps"""

        # Get comprehensive MAC-based analysis
        learning_analysis = self.analyze_vit_multiplier_patterns(
            history, dataset,
            target_macs=target_macs,
            baseline_macs=baseline_macs, 
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
        )
        
        target_ratio = 1.0 - (target_macs / baseline_macs) if baseline_macs else 0.5

        # Format the analysis for the LLM
        analysis_text = _format_mac_analysis_for_llm(learning_analysis, target_ratio, target_macs, dataset)

        
        # Handle strategic guidance if present
        strategic_text = ""
        if strategic_guidance:
            avoid_combinations = strategic_guidance.get('avoid_combinations', [])
            if avoid_combinations:
                strategic_text = f"""
🚨 STRATEGIC GUIDANCE FROM MASTER AGENT:
FORBIDDEN COMBINATIONS: {avoid_combinations}
YOU MUST NOT USE ANY OF THESE (importance_criterion, round_to) COMBINATIONS.
"""
    
        # Calculate MAC tolerance range for guidance
        mac_tolerance_range = f"{target_macs - (target_macs * macs_undershoot_tolerance_pct / 100):.2f}G - {target_macs + (target_macs * macs_overshoot_tolerance_pct / 100):.2f}G" if target_macs else "Not specified"
        
        prompt = f"""You are a ViT pruning expert with advanced MAC-based learning capabilities.

        RESEARCH-BASED IMPORTANCE CRITERION PRIORITY:
        Your system uses isomorphic pruning (global_pruning=true) which groups isomorphic structures.
        Combined with Taylor criterion, this achieves SOTA performance:
        - For ViTs: Isomorphic + Taylor dramatically outperforms magnitude-based methods (DeiT-Tiny: 74.52% → 77.50%)
        - Taylor uses gradient information vs. L1/L2 which are unreliable for heterogeneous transformer structures
        - Your isomorphic grouping addresses the parameter scale divergence problem in ViTs

        RECOMMENDED: Start with "taylor" as importance_criterion based on research evidence.


CRITICAL ROUND_TO INSTRUCTION: {round_to_instruction}

CRITICAL MAC-BASED LEARNING TASK: Based on comprehensive historical analysis, determine optimal ViT MAC allocation for {target_macs:.2f}G target MAC operations on {dataset}.

TARGET SPECIFICATIONS:
- Target MAC Operations: {target_macs:.2f}G
- Baseline MAC Operations: {f"{baseline_macs:.2f}G" if baseline_macs else "Not specified"}
- MAC Tolerance: +{macs_overshoot_tolerance_pct}%/-{macs_undershoot_tolerance_pct}% ({mac_tolerance_range})
- Efficiency Target: {f"{((target_macs / baseline_macs) * 100):.1f}% of baseline" if baseline_macs else "Not calculated"}

{strategic_text}

{analysis_text}

TASK: Calculate specific MAC allocation multiplierages that will achieve {target_macs:.2f}G MAC operations based on the learning analysis above.

OUTPUT FORMAT (JSON only):
{{
    "importance_criterion": "taylor|l1norm|l2norm",
    "target_mac_g": {target_macs:.3f},
    "baseline_mac_g": {(baseline_macs if baseline_macs else 0.0):.3f},
    "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct},
    "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct},
    "round_to": 2,  // Use 2 unless historical analysis shows repeated failures with round_to=2
    "global_pruning": true,
    "rationale": "MAC-BASED LEARNING CALCULATION: [Explain how you used the historical MAC analysis to determine allocation multiplierages]. MATHEMATICAL REASONING: [Show your MAC allocation calculation]. CONFIDENCE: [State your confidence level and why].",
    "architecture_type": "vit",
    "learning_applied": true,
    "isomorphic_group_ratios": {{
        "mlp_multiplier": YOUR_CALCULATED_mlp_multiplierAGE,
        "qkv_multiplier": YOUR_CALCULATED_qkv_multiplierAGE,
        "proj_multiplier": 8.0,
        "head_multiplier": 2.0
    }},
    "expected_mac_breakdown": {{
        "mlp_mac_g": "YOUR_CALCULATED_mlp_multiplierAGE * {target_macs:.3f} / 100",
        "qkv_mac_g": "YOUR_CALCULATED_qkv_multiplierAGE * {target_macs:.3f} / 100",
        "proj_mac_g": "{(8.0 * target_macs / 100):.3f}",
        "head_mac_g": "{(2.0 * target_macs / 100):.3f}"
    }},
    "confidence_level": "high|medium|low",
    "learning_insights": [
        "Key insight 1 from historical MAC analysis",
        "Key insight 2 about MAC allocation effectiveness"
    ]
}}

CRITICAL INSTRUCTIONS:
1. Use the MAC-based learning analysis above to make intelligent decisions
2. Ensure mlp_multiplier + qkv_multiplier + proj_multiplier + head_multiplier ≤ 100%
3. Focus on achieving {target_macs:.2f}G +{macs_overshoot_tolerance_pct}%/-{macs_undershoot_tolerance_pct}% MAC operations
4. Do not repeat failed MAC allocation patterns from history
5. Prioritize MAC efficiency while maintaining accuracy"""

        return prompt
