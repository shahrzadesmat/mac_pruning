class CNNLearningAnalyzer:
    """
    Enhanced learning system for CNN models that analyzes patterns and provides
    intelligent guidance to prevent undershooting/overshooting for channel pruning
    """

    TOLERANCE_OVERSHOOT = 0.01   # +1% (strict)
    TOLERANCE_UNDERSHOOT = 0.05  # -5% (lenient)
    
    def __init__(self):
        self.min_data_points = 2  # Minimum attempts needed for pattern analysis
        

    def _get_high_accuracy_threshold(self, dataset: str) -> float:
        """
        Return a strict accuracy threshold for 'catastrophic failure' detection,
        based on dataset size/difficulty.
        """
        if dataset.lower() == "imagenet":
            # ImageNet models can lose only a little accuracy
            return 1.0
        elif dataset.lower() in {"cifar10", "cifar-10"}:
            # CIFAR-10 tolerates a larger absolute drop
            return 85.0
        else:
            # default fallback
            return 90.0

    def _calculate_trend(self, values):
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Remove None values
        clean_values = [v for v in values if v is not None]
        if len(clean_values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        n = len(clean_values)
        x_sum = sum(range(n))
        y_sum = sum(clean_values)
        xy_sum = sum(i * v for i, v in enumerate(clean_values))
        x2_sum = sum(i * i for i in range(n))
        
        # Calculate slope
        denominator = n * x2_sum - x_sum * x_sum
        if abs(denominator) < 1e-10:
            return 'flat'
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        
        # Classify trend
        if abs(slope) < 1e-6:
            return 'flat'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _check_direction_consistency(self, sorted_attempts, target_value, tolerance_overshoot=1.0, tolerance_undershoot=5.0, value_key='macs_error_pct', adjustment_key='channel_ratio'):
        """Generic method to check if adjustments consistently move toward target."""
        if len(sorted_attempts) < 3:
            return 'insufficient_data'

        correct_moves = 0
        total_moves = 0

        for i in range(1, len(sorted_attempts)):
            prev = sorted_attempts[i - 1]
            curr = sorted_attempts[i]

            prev_error = prev.get(value_key)
            if prev_error is None:
                continue

            adjustment_increased = curr[adjustment_key] > prev[adjustment_key]

            # Positive error = over target, need increase; negative = under target, need decrease
            if prev_error > tolerance_overshoot and adjustment_increased:
                correct_moves += 1
            elif prev_error < -tolerance_undershoot and not adjustment_increased:
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

    def _detect_oscillation_pattern(self, sorted_attempts, value_key='macs_error_pct', threshold=0.02):
        """Detect if the system is oscillating around target values."""
        if len(sorted_attempts) < 4:
            return {'oscillating': False, 'reason': 'insufficient_data'}

        values = [a.get(value_key) for a in sorted_attempts if a.get(value_key) is not None]
        if len(values) < 4:
            return {'oscillating': False, 'reason': 'insufficient_valid_values'}

        # Check for sign changes (crossing zero/target)
        sign_changes = 0
        for i in range(1, len(values)):
            if (values[i-1] > 0) != (values[i] > 0):  # Sign changed
                sign_changes += 1

        # Check if errors are consistently above threshold (not converging)
        recent_errors = [abs(v) for v in values[-3:]]
        avg_recent_error = sum(recent_errors) / len(recent_errors)

        oscillating = (sign_changes >= len(values) * 0.4 and avg_recent_error > threshold)
        
        return {
            'oscillating': oscillating,
            'sign_changes': sign_changes,
            'avg_recent_error': avg_recent_error,
            'pattern_strength': sign_changes / len(values) if len(values) > 0 else 0
        }

    def _calculate_convergence_rate(self, sorted_attempts, value_key='macs_error_pct'):
        """Measure how quickly errors are decreasing."""
        if len(sorted_attempts) < 3:
            return {'converging': False, 'reason': 'insufficient_data'}

        errors = [abs(a.get(value_key, float('inf'))) for a in sorted_attempts 
                if a.get(value_key) is not None]
        
        if len(errors) < 3:
            return {'converging': False, 'reason': 'insufficient_valid_errors'}

        # Calculate improvement rate over recent attempts
        recent_errors = errors[-3:]
        if len(recent_errors) >= 2:
            improvement = recent_errors[0] - recent_errors[-1]
            rate = improvement / len(recent_errors)
        else:
            improvement = 0
            rate = 0

        # Check if trend is consistently downward
        decreasing_count = 0
        for i in range(1, len(recent_errors)):
            if recent_errors[i] < recent_errors[i-1]:
                decreasing_count += 1

        converging = (improvement > 0.001 and 
                    decreasing_count >= len(recent_errors) * 0.6)

        return {
            'converging': converging,
            'improvement_rate': rate,
            'total_improvement': improvement,
            'recent_error': recent_errors[-1],
            'decreasing_trend_strength': decreasing_count / max(1, len(recent_errors) - 1)
        }


    def analyze_cnn_channel_patterns(self, history, target_ratio, dataset,
                                    baseline_macs=None, target_macs=None, 
                                    macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """
        Comprehensive analysis of CNN channel pruning patterns to guide LLM (MAC-first).
        target_ratio is kept for backward-compatibility only.
        """
        if not history or len(history) < self.min_data_points:
            return self._create_baseline_guidance(
                target_macs, dataset,
                baseline_macs=baseline_macs,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
            )

        # Extract and analyze all attempts
        attempts = self._extract_cnn_attempts(history, baseline_macs=baseline_macs, default_target_macs=target_macs)

        if len(attempts) < self.min_data_points:
            return self._create_baseline_guidance(
                target_macs, dataset,
                baseline_macs=baseline_macs,
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
            )

        # Analyze patterns (MAC-first)
        pattern_analysis = self._analyze_channel_effectiveness(
            attempts, target_macs=target_macs, 
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
        )
        trend_analysis = self._analyze_channel_trends(
            attempts, target_macs=target_macs,
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
        )
        failure_analysis = self._analyze_channel_failures(
            attempts, dataset=dataset, 
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
        )

        # Generate intelligent recommendations (FIXED: pass MAC parameters)
        recommendations = self._generate_cnn_recommendations(
            pattern_analysis, trend_analysis, failure_analysis, 
            target_macs, dataset, baseline_macs, 
            macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct
        )

        return {
            'pattern_analysis': pattern_analysis,
            'trend_analysis': trend_analysis,
            'failure_analysis': failure_analysis,
            'recommendations': recommendations,
            'has_sufficient_data': True,
            'total_attempts': len(attempts)
        }


    def _extract_cnn_attempts(self, history, baseline_macs=None, default_target_macs=None):
        """Extract CNN-specific attempts from history (MACs-first; supports legacy fields)."""
        attempts = []
        baseline_ops = baseline_macs if baseline_macs is not None else None
        default_target_ops = default_target_macs if default_target_macs is not None else None

        for entry in history:
            strategy = entry.get('strategy_used', {}) or {}

            # Channel ratio knob (various keys)
            channel_ratio = (strategy.get('channel_pruning_ratio') or
                            strategy.get('channel_ratio_used') or
                            strategy.get('channel_ratio'))

            importance = strategy.get('importance_criterion')
            round_to = strategy.get('round_to')

            # Prefer absolute MACs; fall back to legacy ratios if necessary
            achieved_ops = (entry.get('achieved_macs') or entry.get('final_macs') or entry.get('macs_after'))
            target_ops = entry.get('target_macs')

            if target_ops is None:
                if default_target_ops is not None:
                    target_ops = default_target_ops
                elif baseline_ops is not None:
                    legacy_tr = entry.get('target_ratio')
                    if legacy_tr is not None:
                        target_ops = baseline_ops * (1.0 - float(legacy_tr))

            if achieved_ops is None and baseline_ops is not None:
                legacy_ar = entry.get('achieved_ratio')
                if legacy_ar is not None:
                    achieved_ops = baseline_ops * (1.0 - float(legacy_ar))

            # Accuracy
            if (entry.get('dataset', '') or dataset if (dataset := entry.get('dataset', '')) else '').lower() == 'imagenet':
                accuracy = entry.get('zero_shot_top1_accuracy', entry.get('zero_shot_accuracy', 0))
            else:
                accuracy = entry.get('zero_shot_accuracy', 0)
            if accuracy is None:
                accuracy = 0.0

            if channel_ratio is None or achieved_ops is None:
                print(f"[⚠️] Skipping entry with incomplete CNN data (channel_ratio or achieved MACs missing)")
                continue

            macs_error_pct = None
            if target_ops:
                macs_error_pct = (float(achieved_ops) - float(target_ops)) / max(float(target_ops), 1e-9) * 100.0

            attempts.append({
                'channel_ratio': float(channel_ratio),
                'importance_criterion': importance,
                'round_to': round_to,
                'achieved_macs': float(achieved_ops),
                'target_macs': float(target_ops) if target_ops is not None else None,
                'macs_error_pct': macs_error_pct,  # signed percent; + => too heavy (not enough pruning)
                'revision': entry.get('revision', 0),
                'accuracy': float(accuracy)
            })

        # Sort by revision to get chronological order
        attempts.sort(key=lambda x: x['revision'])
        print(f"[📊] Extracted {len(attempts)} valid CNN attempts in chronological order")
        return attempts


    def _analyze_channel_effectiveness(self, attempts, target_macs=None, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0, accuracy_threshold=None):
        """Analyze how effective different channel ratios are (MACs-first)."""
        analysis = {
            'avg_channel_ratio': 0.0,
            'avg_achieved_macs': 0.0,
            'avg_macs_error_pct': 0.0,
            'systematic_undershoot': False,
            'systematic_overshoot': False,
            'closest_attempt': None,
            'most_aggressive': None,
            'most_conservative': None,
            'has_excellent_near_miss': False,
            'high_accuracy_undershoots': [],
            'channel_ratio_effectiveness': {}
        }

        if accuracy_threshold is None:
            accuracy_threshold = self._get_high_accuracy_threshold("imagenet")

        if not attempts:
            return analysis

        # Averages
        analysis['avg_channel_ratio'] = sum(a['channel_ratio'] for a in attempts) / len(attempts)
        analysis['avg_achieved_macs'] = (sum(a['achieved_macs'] for a in attempts) / len(attempts)) / 1e9
        valid_errs = [a['macs_error_pct'] for a in attempts if a['macs_error_pct'] is not None]
        analysis['avg_macs_error_pct'] = (sum(valid_errs) / len(valid_errs)) if valid_errs else 0.0

        # Systematic patterns
        undershoots = sum(1 for a in attempts if (a['macs_error_pct'] is not None and a['macs_error_pct'] > macs_overshoot_tolerance_pct))
        overshoots  = sum(1 for a in attempts if (a['macs_error_pct'] is not None and a['macs_error_pct'] < -macs_undershoot_tolerance_pct))
        analysis['systematic_undershoot'] = undershoots >= len(attempts) * 0.8
        analysis['systematic_overshoot']  = overshoots  >= len(attempts) * 0.8

        # Extremes and closest
        with_err = [a for a in attempts if a['macs_error_pct'] is not None]
        analysis['closest_attempt'] = min(with_err, key=lambda x: abs(x['macs_error_pct'])) if with_err else None
        analysis['most_aggressive'] = max(attempts, key=lambda x: x['channel_ratio'])
        analysis['most_conservative'] = min(attempts, key=lambda x: x['channel_ratio'])

        # Group by channel ratio
        channel_groups = {}
        for a in attempts:
            ratio_key = round(a['channel_ratio'], 2)
            channel_groups.setdefault(ratio_key, []).append(a)

        for ratio, group in channel_groups.items():
            g_errs = [x['macs_error_pct'] for x in group if x['macs_error_pct'] is not None]
            avg_err = (sum(g_errs) / len(g_errs)) if g_errs else None
            analysis['channel_ratio_effectiveness'][ratio] = {
                'avg_achieved_macs': (sum(x['achieved_macs'] for x in group) / len(group)) / 1e9,
                'avg_macs_error_pct': avg_err,
                'avg_accuracy': sum(x['accuracy'] for x in group) / len(group),
                'count': len(group),
                'success_rate': sum(1 for x in group if (x['macs_error_pct'] is not None and -macs_undershoot_tolerance_pct <= x['macs_error_pct'] <= macs_overshoot_tolerance_pct)) / len(group)
            }

        # High-accuracy small undershoots (on the heavy side but within tolerance band)
        if analysis['systematic_undershoot']:
            high_acc_attempts = [
                a for a in attempts
                if (a['macs_error_pct'] is not None and 0 < a['macs_error_pct'] <= macs_overshoot_tolerance_pct)
                and a.get('accuracy', 0) > accuracy_threshold
            ]
            analysis['high_accuracy_undershoots'] = high_acc_attempts
            analysis['has_excellent_near_miss'] = bool(high_acc_attempts)
            if analysis['has_excellent_near_miss']:
                print(f"[🎯] DETECTED: {len(high_acc_attempts)} excellent CNN near-miss attempts")
                best_near = max(high_acc_attempts, key=lambda x: x.get('accuracy', 0))
                print(f"[🎯] Best: {best_near.get('accuracy', 0):.2f}% accuracy, {best_near.get('macs_error_pct', 0):.2f}% MACs error")

        return analysis

    def _analyze_channel_trends(self, attempts, target_macs=None, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Analyze trends in channel ratio adjustments (MACs-first)."""
        if len(attempts) < 3:
            return {'insufficient_data': True}

        sorted_attempts = sorted(attempts, key=lambda x: x['revision'])

        trend_analysis = {
            'channel_ratio_trend': self._calculate_trend([a['channel_ratio'] for a in sorted_attempts]),
            'achievement_trend': self._calculate_trend([a['achieved_macs'] for a in sorted_attempts]),
            'deviation_trend': self._calculate_trend([a['macs_error_pct'] for a in sorted_attempts if a['macs_error_pct'] is not None]),
            'accuracy_trend': self._calculate_trend([a['accuracy'] for a in sorted_attempts]),
            'is_converging': False,
            'direction_consistency': self._check_cnn_direction_consistency(
                sorted_attempts, 
                macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
                macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
            ),
            'optimal_channel_ratio_estimate': None
        }

        recent_errs = [abs(a['macs_error_pct']) for a in sorted_attempts[-3:] if a['macs_error_pct'] is not None]
        trend_analysis['is_converging'] = len(recent_errs) >= 2 and recent_errs[-1] < recent_errs[0]

        if len(sorted_attempts) >= 2 and target_macs is not None:
            trend_analysis['optimal_channel_ratio_estimate'] = self._estimate_optimal_channel_ratio(sorted_attempts, target_macs=target_macs)

        return trend_analysis


    def _check_cnn_direction_consistency(self, sorted_attempts, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Check if channel ratio adjustments consistently move MACs error toward 0%."""
        if len(sorted_attempts) < 3:
            return 'insufficient_data'

        correct_moves = 0
        total_moves = 0

        for i in range(1, len(sorted_attempts)):
            prev = sorted_attempts[i - 1]
            curr = sorted_attempts[i]

            prev_err = prev.get('macs_error_pct')
            if prev_err is None:
                continue

            channel_ratio_increased = curr['channel_ratio'] > prev['channel_ratio']

            # Positive error → model too heavy (not enough pruning) → should increase channel ratio
            if prev_err > macs_overshoot_tolerance_pct and channel_ratio_increased:
                correct_moves += 1
            # Negative error → model too light (too much pruning) → should decrease channel ratio
            elif prev_err < -macs_undershoot_tolerance_pct and not channel_ratio_increased:
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


    def _estimate_optimal_channel_ratio(self, sorted_attempts, target_macs=None, baseline_macs=None):
        """
        Estimate optimal channel ratio by linear interpolation/extrapolation on (channel_ratio → achieved_macs).
        Falls back to ratio-based estimate if MACs are unavailable.
        """
        # Try MACs-based line using the two most recent distinct channel ratios
        for i in range(len(sorted_attempts) - 1, 0, -1):
            a2, a1 = sorted_attempts[i], sorted_attempts[i - 1]
            if abs(a2['channel_ratio'] - a1['channel_ratio']) <= 1e-3:
                continue
            y1, y2 = a1.get('achieved_macs'), a2.get('achieved_macs')
            if y1 is None or y2 is None:
                continue
            if target_macs is None:
                break
            target_ops = target_macs
            if abs(y2 - y1) < 1e-6:
                continue
            # linear solve: x at y=target
            x1, x2 = a1['channel_ratio'], a2['channel_ratio']
            est = x1 + (target_ops - y1) * (x2 - x1) / (y2 - y1)
            return max(0.05, min(0.8, float(est)))

        # Legacy fallback: estimate using achieved_ratio if present
        for i in range(len(sorted_attempts) - 1, 0, -1):
            a2, a1 = sorted_attempts[i], sorted_attempts[i - 1]
            if abs(a2['channel_ratio'] - a1['channel_ratio']) <= 1e-3:
                continue
            y1, y2 = a1.get('achieved_ratio'), a2.get('achieved_ratio')
            if y1 is None or y2 is None or abs(y2 - y1) < 1e-6:
                continue
            x1, x2 = a1['channel_ratio'], a2['channel_ratio']
            # Solve for target_ratio (unknown here). If you want, pass it and use it—kept generic.
            return max(0.05, min(0.8, float(x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1))))  # heuristic target=0.5 if unknown

        return None


    def _analyze_channel_failures(self, attempts, dataset, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Analyze specific failure patterns for CNN models (MACs-first)."""
        failures = []
        catastrophic_failures = []

        for a in attempts:
            err = a.get('macs_error_pct')
            if err is not None:
                if err > macs_overshoot_tolerance_pct:  # too heavy → undershoot of pruning
                    failures.append({
                        'type': 'undershoot',
                        'severity': err - macs_overshoot_tolerance_pct,
                        'channel_ratio': a['channel_ratio'],
                        'importance': a.get('importance_criterion'),
                        'round_to': a.get('round_to'),
                        'attempt': a
                    })
                elif err < -macs_undershoot_tolerance_pct:  # too light → overshoot of pruning
                    failures.append({
                        'type': 'overshoot',
                        'severity': (-macs_undershoot_tolerance_pct - err),
                        'channel_ratio': a['channel_ratio'],
                        'importance': a.get('importance_criterion'),
                        'round_to': a.get('round_to'),
                        'attempt': a
                    })

            # Catastrophic accuracy failures
            if dataset.lower() == 'imagenet':
                if a.get('accuracy', 0) < 10:
                    catastrophic_failures.append(a)
            else:
                if a.get('accuracy', 0) < 30:
                    catastrophic_failures.append(a)

        return {
            'total_failures': len(failures),
            'undershoot_failures': [f for f in failures if f['type'] == 'undershoot'],
            'overshoot_failures': [f for f in failures if f['type'] == 'overshoot'],
            'catastrophic_failures': catastrophic_failures,
            'worst_undershoot': max(failures, key=lambda x: x['severity']) if failures else None,
            'avg_undershoot_severity': (
                sum(f['severity'] for f in failures if f['type'] == 'undershoot') /
                max(1, len([f for f in failures if f['type'] == 'undershoot']))
            ),
            'problematic_channel_ratios': self._identify_problematic_ratios(catastrophic_failures)
        }

    
    def _identify_problematic_ratios(self, catastrophic_failures):
        """Identify channel ratios that consistently lead to catastrophic failures (robust to key variants)"""
        problematic = {}
        for failure in catastrophic_failures or []:
            # Support both flat and nested storage of channel ratio
            ratio = (
                failure.get('channel_ratio')
                or (failure.get('attempt', {}) or {}).get('channel_ratio')
                or (failure.get('strategy_used', {}) or {}).get('channel_pruning_ratio')
            )
            if ratio is None:
                continue
            ratio = round(float(ratio), 2)
            problematic[ratio] = problematic.get(ratio, 0) + 1

        # Return ratios that caused failures multiple times
        return {ratio: count for ratio, count in problematic.items() if count >= 2}


    def _generate_cnn_recommendations(self, pattern_analysis, trend_analysis, failure_analysis, 
                                    target_macs, dataset, baseline_macs=None, 
                                    macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Generate intelligent CNN recommendations based on all analyses (MAC-based)"""
        recommendations = {
            'suggested_channel_ratio': 0.0,
            'suggested_importance_criterion': 'taylor',
            'suggested_round_to': 4,
            'confidence': 'low',
            'reasoning': [],
            'mathematical_calculation': {},
            'safety_constraints': [],
            'learning_insights': []
        }

        pa = pattern_analysis or {}
        ta = trend_analysis or {}
        fa = failure_analysis or {}

        # --- Read MAC-based metrics when available; fall back to legacy ratio-based ones ---
        avg_macs_err_pct = pa.get('avg_macs_error_pct')          # percent, signed
        avg_deviation_frac_legacy = pa.get('avg_deviation')      # fraction, e.g. 0.024
        has_near_miss = pa.get('has_excellent_near_miss', False)
        high_acc_undershoots = pa.get('high_accuracy_undershoots', [])
        avg_channel_ratio = pa.get('avg_channel_ratio', 0.3)

        # Systematic flags are already computed upstream
        sys_under = pa.get('systematic_undershoot', False)
        sys_over  = pa.get('systematic_overshoot', False)

        # Helper: express an "undershoot severity" in fraction units to keep thresholds consistent
        if avg_macs_err_pct is not None:
            undershoot_severity_frac = max(0.0, float(avg_macs_err_pct)) / 100.0
        else:
            undershoot_severity_frac = abs(float(avg_deviation_frac_legacy or 0.0))

        # =========================
        # SYSTEMATIC UNDERSHOOT
        # =========================
        if sys_under:
            # --- High-accuracy tiny undershoots (prefer MAC fields if present) ---
            if has_near_miss and high_acc_undershoots:
                best_high_acc = max(high_acc_undershoots, key=lambda x: x.get('accuracy', 0))
                best_acc = best_high_acc.get('accuracy', 0)

                # Pull achieved/target either as MACs or legacy ratio
                best_ach_macs = best_high_acc.get('achieved_macs')
                best_tgt_macs = best_high_acc.get('target_macs')
                best_err_pct  = best_high_acc.get('macs_error_pct')
                best_ach_ratio_legacy = best_high_acc.get('achieved_ratio')
                best_dev_legacy       = best_high_acc.get('deviation')

                best_channel = best_high_acc.get('channel_ratio', avg_channel_ratio)

                # Very small correction to preserve accuracy
                correction_factor = 1.05

                # Safety caps by dataset
                max_safe_channel = 0.6 if dataset.lower() == 'imagenet' else 0.8
                corrected_channel = min(best_channel * correction_factor, max_safe_channel)

                recommendations['suggested_channel_ratio'] = corrected_channel
                recommendations['suggested_importance_criterion'] = best_high_acc.get('importance_criterion', 'taylor')
                recommendations['suggested_round_to'] = best_high_acc.get('round_to', 4)
                recommendations['confidence'] = 'very_high'

                # Reasoning strings (MAC-first; legacy fallback)
                if best_ach_macs is not None and best_tgt_macs is not None and best_err_pct is not None:
                    recommendations['reasoning'].append(
                        f"TINY MAC ADJUSTMENT: Excellent {best_acc:.1f}% accuracy, "
                        f"achieved {best_ach_macs/1e9:.2f}G vs target {best_tgt_macs/1e9:.2f}G "
                        f"(error {best_err_pct:.2f}%). Apply ×{correction_factor:.2f} to channel ratio."
                    )
                else:
                    recommendations['reasoning'].append(
                        f"TINY ADJUSTMENT: Excellent {best_acc:.1f}% accuracy, "
                        f"achieved {best_ach_ratio_legacy*100:.1f}% reduction "
                        f"(deviation {abs(best_dev_legacy)*100:.1f}%). Apply ×{correction_factor:.2f}."
                    )

                recommendations['mathematical_calculation'] = {
                    'strategy_type': 'high_accuracy_tiny_mac_adjustment',
                    'best_accuracy_found': best_acc,
                    'original_channel_ratio': best_channel,
                    'correction_factor': correction_factor,
                    'corrected_channel_ratio': corrected_channel,
                    # MAC-first context if available:
                    'achieved_macs': (best_ach_macs / 1e9) if best_ach_macs is not None else None,
                    'target_macs': (best_tgt_macs / 1e9) if best_tgt_macs is not None else None,
                    'macs_error_pct': best_err_pct,
                    # Legacy context (kept):
                    'best_achieved_ratio': best_ach_ratio_legacy,
                    'tiny_deviation': best_dev_legacy,
                    'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                    'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct
                }
                return recommendations

            # --- Regular systematic undershoot handling ---
            recommendations['reasoning'].append(
                f"SYSTEMATIC MAC UNDERSHOOT: average "
                f"{(avg_macs_err_pct if avg_macs_err_pct is not None else (avg_deviation_frac_legacy*100 if avg_deviation_frac_legacy is not None else 0)):.1f}% "
                f"{'MAC error' if avg_macs_err_pct is not None else 'deviation'}"
            )

            # Choose correction factor by severity (fraction domain thresholds)
            if undershoot_severity_frac > 0.04:       # > 4%
                correction_factor = 1.4
                recommendations['confidence'] = 'high'
            elif undershoot_severity_frac > 0.02:     # > 2%
                correction_factor = 1.25
                recommendations['confidence'] = 'medium'
            else:
                correction_factor = 1.15
                recommendations['confidence'] = 'medium'
            recommendations['reasoning'].append(f"Applying correction factor ×{correction_factor:.2f} to average channel ratio.")

            # Estimated optimal or corrected average
            estimated_optimal = ta.get('optimal_channel_ratio_estimate')
            if estimated_optimal:
                base_channel = float(estimated_optimal)
                recommendations['reasoning'].append(f"Using estimated optimal ratio: {base_channel:.3f}")
            else:
                base_channel = float(avg_channel_ratio) * correction_factor
                recommendations['reasoning'].append(f"Corrected average ratio: {avg_channel_ratio:.3f} × {correction_factor:.2f} = {base_channel:.3f}")

            # Safety caps
            max_safe_channel = 0.6 if dataset.lower() == 'imagenet' else 0.8
            recommendations['suggested_channel_ratio'] = min(base_channel, max_safe_channel)

            # Choose importance from best-performing bucket if available
            effectiveness = pa.get('channel_ratio_effectiveness', {})
            if effectiveness:
                best_ratio_key = max(effectiveness.keys(), key=lambda k: effectiveness[k].get('avg_accuracy', 0))
                # try to pick an importance that appeared with that ratio in undershoot failures (as proxy to data)
                best_attempts = [a for a in fa.get('undershoot_failures', []) if round(a.get('channel_ratio', -1), 2) == round(float(best_ratio_key), 2)]
                if best_attempts:
                    recommendations['suggested_importance_criterion'] = best_attempts[0].get('importance', 'taylor')

            recommendations['mathematical_calculation'] = {
                'avg_channel_before': avg_channel_ratio,
                'correction_factor': correction_factor,
                'estimated_optimal': estimated_optimal,
                'base_channel_after_correction': base_channel,
                'safety_capped_channel': recommendations['suggested_channel_ratio'],
                'target_macs': target_macs,
                'baseline_macs': baseline_macs,
                'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct
            }

        # =========================
        # SYSTEMATIC OVERSHOOT
        # =========================
        elif sys_over:
            # Use avg MAC error if present; negative means too much pruning (achieved MACs below target)
            if avg_macs_err_pct is not None:
                avg_overshoot_frac = max(0.0, -float(avg_macs_err_pct) / 100.0)
            else:
                # legacy: positive deviation means exceeded reduction target; treat similarly
                avg_overshoot_frac = max(0.0, float(avg_deviation_frac_legacy or 0.0))

            reduction_factor = 0.8 if avg_overshoot_frac > 0.05 else 0.9
            recommendations['suggested_channel_ratio'] = float(avg_channel_ratio) * reduction_factor
            recommendations['suggested_importance_criterion'] = 'taylor'
            recommendations['suggested_round_to'] = 4
            recommendations['confidence'] = 'medium'
            recommendations['reasoning'].append(f"SYSTEMATIC MAC OVERSHOOT: applying reduction factor ×{reduction_factor:.2f} to average channel ratio.")
            recommendations['mathematical_calculation'] = {
                'avg_overshoot_frac': avg_overshoot_frac,
                'reduction_factor': reduction_factor,
                'avg_channel_before': avg_channel_ratio,
                'suggested_channel_after': recommendations['suggested_channel_ratio'],
                'target_macs': target_macs,
                'baseline_macs': baseline_macs,
                'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct
            }

        # =========================
        # MIXED RESULTS → use closest attempt
        # =========================
        else:
            closest = pa.get('closest_attempt')
            if closest:
                estimated_optimal = ta.get('optimal_channel_ratio_estimate')
                if estimated_optimal:
                    recommendations['suggested_channel_ratio'] = float(estimated_optimal)
                    recommendations['reasoning'].append(f"Using mathematically estimated optimal ratio: {float(estimated_optimal):.3f}")
                else:
                    # If MAC error is available, positive error => too heavy => increase ratio; negative => decrease
                    macs_err_pct = closest.get('macs_error_pct')
                    if macs_err_pct is not None:
                        adjustment = 1.10 if macs_err_pct > 0 else 0.95
                    else:
                        # legacy deviation: negative (undershoot of reduction) → increase; positive → decrease
                        deviation = closest.get('deviation', 0)
                        adjustment = 1.10 if deviation < 0 else 0.95
                    new_ratio = float(closest.get('channel_ratio', avg_channel_ratio)) * adjustment
                    recommendations['suggested_channel_ratio'] = new_ratio
                    recommendations['reasoning'].append(f"Closest attempt adjusted by ×{adjustment:.2f} to move toward MAC target.")

                recommendations['suggested_importance_criterion'] = closest.get('importance_criterion', 'taylor')
                recommendations['suggested_round_to'] = closest.get('round_to', 4)
                recommendations['confidence'] = 'medium'
                recommendations['mathematical_calculation'] = {
                    'strategy_type': 'mixed_results_adjustment',
                    'closest_attempt_used': True,
                    'estimated_optimal': estimated_optimal,
                    'target_macs': target_macs,
                    'baseline_macs': baseline_macs,
                    'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                    'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct
                }

        # =========================
        # Learning insights
        # =========================
        try:
            attempts_count = len(pattern_analysis.get('all_attempts', []))
            if attempts_count:
                recommendations['learning_insights'].append(f"Analyzed {attempts_count} CNN attempts with channel pruning")
        except Exception:
            pass

        if ta.get('direction_consistency') == 'highly_consistent':
            recommendations['learning_insights'].append("Direction adjustments are highly consistent—learning signal is reliable")
        elif ta.get('is_converging'):
            recommendations['learning_insights'].append("Results show convergence—near optimal configuration")

        return recommendations


    
    def _create_baseline_guidance(
        self,
        target_macs,                      
        dataset,
        baseline_macs=None,               
        macs_overshoot_tolerance_pct=1.0,
        macs_undershoot_tolerance_pct=5.0
        ):
        """Create baseline guidance when insufficient historical data (MAC-based)."""

        if dataset.lower() == 'imagenet':
            # Conservative defaults for ImageNet (accuracy-preserving)
            baseline_channel = 0.5
            baseline_importance = 'taylor'
            baseline_round_to = 8
        else:
            # Moderate defaults for CIFAR-10
            baseline_channel = 0.6
            baseline_importance = 'l1norm'
            baseline_round_to = 4

        return {
            'pattern_analysis': None,
            'trend_analysis': None,
            'failure_analysis': None,
            'recommendations': {
                'suggested_channel_ratio': baseline_channel,
                'suggested_importance_criterion': baseline_importance,
                'suggested_round_to': baseline_round_to,
                'confidence': 'low',
                'reasoning': [f"Baseline guidance for {dataset} CNN (insufficient historical data)"],
                'mathematical_calculation': {
                    'baseline_approach': True,
                    'target_channel_pruning_estimate': baseline_channel,
                    'baseline_macs': baseline_macs,
                    'target_macs': target_macs,
                    'macs_overshoot_tolerance_pct': macs_overshoot_tolerance_pct,
                    'macs_undershoot_tolerance_pct': macs_undershoot_tolerance_pct,
                    'expected_achieved_macs': target_macs
                }
            },
            'has_sufficient_data': False,
            'total_attempts': 0
        }


    def _detect_cnn_local_optimum_trap(
        self,
        history,
        target_macs,                     # FIXED: changed from target_ratio to target_macs  
        dataset,
        accuracy_threshold=None,
        baseline_macs=None,              # NEW: optional baseline MACs in G
        macs_overshoot_tolerance_pct=1.0,
        macs_undershoot_tolerance_pct=5.0
        ):
        """
        Detect if CNN is stuck in a local optimum (MAC-based).
        Looks for: one or more excellent, on-budget runs followed by recent failures.
        """

        # Resolve thresholds
        if accuracy_threshold is None:
            accuracy_threshold = self._get_high_accuracy_threshold(dataset)
        if not history or len(history) < 5:
            return None

        # Convert G→ops for internal comparisons
        baseline_macs_ops = baseline_macs if baseline_macs is not None else None
        target_macs_ops   = target_macs if target_macs   is not None else None

        excellent_results = []
        recent_catastrophic = 0

        for i, entry in enumerate(history):
            # Accuracy (dataset-aware)
            accuracy = entry.get('zero_shot_top1_accuracy', entry.get('zero_shot_accuracy', 0)) or 0.0

            # Achieved MACs (ops)
            achieved_ops = (
                entry.get('achieved_macs')
                or entry.get('final_macs')
                or entry.get('macs_after')
            )

            # Target MACs for this attempt (ops)
            entry_target_ops = entry.get('target_macs')
            if entry_target_ops is None:
                # Fallbacks: use function-level target
                if target_macs_ops is not None:
                    entry_target_ops = target_macs_ops
                elif baseline_macs_ops is not None:
                    legacy_tr = entry.get('target_ratio')
                    if legacy_tr is not None:
                        entry_target_ops = baseline_macs_ops * (1.0 - float(legacy_tr))

            # Compute MACs error %
            macs_err_pct = None
            within_tolerance = False
            if achieved_ops is not None and entry_target_ops is not None and entry_target_ops > 0:
                macs_err_pct = (float(achieved_ops) - float(entry_target_ops)) / float(entry_target_ops) * 100.0
                within_tolerance = (-macs_undershoot_tolerance_pct <= macs_err_pct <= macs_overshoot_tolerance_pct)
            else:
                # Legacy fallback using ratios if MACs unavailable
                achieved_ratio = entry.get('achieved_ratio')
                tr = entry.get('target_ratio')
                if achieved_ratio is not None and tr is not None:
                    ratio_diff = float(achieved_ratio) - float(tr)
                    within_tolerance = (-macs_undershoot_tolerance_pct/100.0 <= ratio_diff <= macs_overshoot_tolerance_pct/100.0)
                    
            # Excellent = high accuracy AND near target MACs
            if accuracy > accuracy_threshold and within_tolerance:
                excellent_results.append({
                    'index': i,
                    'accuracy': float(accuracy),
                    'achieved_macs_ops': achieved_ops,
                    'target_macs_ops': entry_target_ops,
                    'macs_error_pct': macs_err_pct,
                    'entry': entry
                })

            # Count catastrophic among last 5 attempts
            if i >= len(history) - 5 and accuracy < accuracy_threshold:
                recent_catastrophic += 1

        # Trap condition: we HAVE an excellent on-budget run, but recent attempts failed repeatedly
        if excellent_results and recent_catastrophic >= 3:
            best_excellent = max(excellent_results, key=lambda x: x['accuracy'])

            # Pretty prints
            print(f"[🕳️] CNN LOCAL OPTIMUM TRAP DETECTED:")
            print(f"Excellent result: {best_excellent['accuracy']:.2f}% accuracy")
            if best_excellent['achieved_macs_ops'] and best_excellent['target_macs_ops']:
                print(f"   MACs: achieved {best_excellent['achieved_macs_ops']/1e9:.2f}G vs target {best_excellent['target_macs_ops']/1e9:.2f}G "
                    f"(error {best_excellent['macs_error_pct']:.2f}%)")
            print(f"Recent failures: {recent_catastrophic}/5 attempts failed")

            return {
                'trap_type': 'cnn_local_optimum',
                'excellent_result': {
                    'index': best_excellent['index'],
                    'accuracy': best_excellent['accuracy'],
                    'achieved_macs': (best_excellent['achieved_macs_ops'] / 1e9) if best_excellent['achieved_macs_ops'] else None,
                    'target_macs': (best_excellent['target_macs_ops'] / 1e9) if best_excellent['target_macs_ops'] else None,
                    'macs_error_pct': best_excellent['macs_error_pct']
                },
                'recent_failures': recent_catastrophic,
                'pattern': 'excellent_on_budget_with_recent_catastrophic_attempts'
            }

        return None

    def _generate_cnn_escape_strategy(self, trap_info, target_macs):
        """
        Generate escape strategy for CNN local optimum (MAC-based)
        """
        excellent_result = trap_info.get('excellent_result', {}) or {}
        best_entry = excellent_result.get('entry', {}) or {}
        best_strategy = best_entry.get('strategy_used', {}) or {}

        # Working knobs from the best-known good attempt
        working_channel_ratio = float(best_strategy.get('channel_pruning_ratio', 0.2) or 0.2)
        working_importance = best_strategy.get('importance_criterion', 'taylor') or 'taylor'
        working_round_to = best_strategy.get('round_to', 8) or 8

        # Pull MAC info if available (from detector), fall back to legacy ratio if present
        acc = float(excellent_result.get('accuracy', best_entry.get('zero_shot_top1_accuracy',
                        best_entry.get('zero_shot_accuracy', 0)) or 0.0))
        achieved_ratio_legacy = best_entry.get('achieved_ratio', excellent_result.get('achieved'))

        achieved_macs = excellent_result.get('achieved_macs')
        target_macs_result = excellent_result.get('target_macs')
        macs_error_pct = excellent_result.get('macs_error_pct')

        # Build human-readable recommendation bullets (MAC-first if we can)
        recs = [
            f"CNN LOCAL OPTIMUM: An excellent result {acc:.1f}% accuracy exists"
        ]
        if achieved_macs is not None and target_macs_result is not None and macs_error_pct is not None:
            recs.append(f"ON-BUDGET: achieved {achieved_macs:.2f}G vs target {target_macs_result:.2f}G (error {macs_error_pct:.2f}%)")
        elif achieved_ratio_legacy is not None:
            recs.append(f"(Legacy) Achieved {achieved_ratio_legacy*100:.1f}% parameter reduction")

        recs.extend([
            f"AVOID: Tiny adjustments to working channel_ratio {working_channel_ratio:.3f}",
            "STRATEGY: Try meaningfully different channel ratios",
            f"PRESERVE: Working importance='{working_importance}' and round_to={working_round_to}",
            "EXPLORE: Different channel ratio ranges, not micro-adjustments"
        ])

        # Exploration bands around the working channel ratio (avoid micro changes)
        more_conservative_low  = max(0.05, working_channel_ratio * 0.6)
        more_conservative_high = max(0.07, working_channel_ratio * 0.9)

        strategic_guidance = {
            'guidance_type': 'cnn_escape_local_optimum',
            'trap_detected': True,
            'excellent_baseline': {
                'accuracy': acc,
                'achieved_ratio': achieved_ratio_legacy,          # legacy (may be None)
                'achieved_macs': achieved_macs,               # MAC-first
                'target_macs': target_macs_result,            # MAC-first
                'macs_error_pct': macs_error_pct,                 # MAC-first
                'working_channel_ratio': working_channel_ratio,
                'working_strategy': best_strategy
            },
            'exploration_strategy': 'diversified_channel_ratio_search',
            'strategic_recommendations': recs,
            'exploration_directions': [
                {
                    'direction': 'more_conservative_channel',
                    'suggested_channel_range': [more_conservative_low, more_conservative_high],
                    'rationale': 'Reduce channel pruning aggressiveness to preserve accuracy while staying near MAC target'
                },
                {
                    'direction': 'different_importance_same_ratio',
                    'preserve_channel_ratio': True,
                    'suggested_importance': ['l1norm', 'l2norm'] if working_importance == 'taylor' else ['taylor'],
                    'rationale': 'Probe sensitivity to importance criterion at the same structural budget'
                }
            ],
            'forbidden_actions': [
                'micro_adjustments_to_excellent_channel_ratio',
                f'channel_ratios_within_0.02_of_{working_channel_ratio:.3f}',
                'repeating_any_failed_recent_combination'
            ],
            # MAC-based target
            'target_macs': target_macs
        }

        return {
            'strategic_guidance': strategic_guidance,
            'continue': True,
            'rationale': 'CNN escape from local optimum: diversify channel ratio search while preserving the known-good settings; prefer MAC-on-budget exploration.',
            'guidance_type': 'cnn_escape_local_optimum'
        }

    
    def create_enhanced_cnn_learning_prompt(
        self,
        target_ratio,                       # kept for backward-compat (legacy)
        model_name,
        history,
        dataset,
        strategic_guidance=None,
        baseline_macs=None,
        target_macs=None, 
        macs_overshoot_tolerance_pct=1.0, 
        macs_undershoot_tolerance_pct=5.0
        ):
        """Create an enhanced prompt with comprehensive CNN learning analysis (MAC-first)."""

        print(" macs_overshoot_tolerance_pct value in analysis_cnn.py is:",  macs_overshoot_tolerance_pct)
        print(" macs_undershoot_tolerance_pct value in analysis_cnn.py is:",  macs_undershoot_tolerance_pct)

        # Get comprehensive analysis - FIXED: pass MAC parameters
        learning_analysis = self.analyze_cnn_channel_patterns(
            history, target_ratio, dataset,
            baseline_macs=baseline_macs,
            target_macs=target_macs,
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
        )

        # Format the analysis for the LLM - FIXED: pass MAC parameters
        analysis_text = self._format_cnn_analysis_for_llm(
            learning_analysis, target_macs, dataset, 
            baseline_macs=baseline_macs,
            macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
            macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct
        )

        # Strategic guidance text
        strategic_text = ""
        if strategic_guidance:
            avoid_combinations = strategic_guidance.get('avoid_combinations', [])
            if avoid_combinations:
                    strategic_text = f"""
        🚨 STRATEGIC GUIDANCE FROM MASTER AGENT:
        FORBIDDEN COMBINATIONS: {avoid_combinations}
        YOU MUST NOT USE ANY OF THESE (importance_criterion, round_to) COMBINATIONS.
        """

        base_str = f"{baseline_macs:.3f}G" if baseline_macs is not None else "N/A"
        tgt_str = f"{target_macs:.3f}G" if target_macs is not None else "N/A"
        
        # Calculate MAC efficiency target
        efficiency_target = (target_macs / baseline_macs * 100) if (baseline_macs and target_macs) else "N/A"
        efficiency_str = f"{efficiency_target:.1f}%" if isinstance(efficiency_target, (int, float)) else "N/A"

        prompt = f"""You are a CNN pruning expert with advanced MAC-based learning capabilities.

        CRITICAL MAC-BASED LEARNING TASK: Based on comprehensive historical analysis, determine optimal CNN channel pruning parameters to meet a MAC budget of {tgt_str} (baseline {base_str}, efficiency target {efficiency_str}, tolerance +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) on {dataset}.

        TARGET SPECIFICATIONS:
        - Target MAC Operations: {tgt_str}
        - Baseline MAC Operations: {base_str}
        - MAC Efficiency Target: {efficiency_str} of baseline
        - MAC Tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%
        - Dataset: {dataset}

        {strategic_text}

        {analysis_text}

        TASK: Calculate a specific channel_pruning_ratio (and related parameters) that will achieve the target MAC budget within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% of {tgt_str}.

        OUTPUT FORMAT (JSON only):
        {{
            "importance_criterion": "taylor|l1norm|l2norm",
            "channel_pruning_ratio": YOUR_CALCULATED_VALUE_BASED_ON_MAC_ANALYSIS,
            "round_to": INTEGER_VALUE,
            "global_pruning": true,
            "baseline_macs": {baseline_macs if baseline_macs is not None else "null"},
            "target_macs": {target_macs if target_macs is not None else "null"},
            "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct:.1f},
            "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct:.1f},
            "expected_achieved_macs": YOUR_CALCULATED_EXPECTED_MAC_VALUE,
            "mac_efficiency_target": {efficiency_target if efficiency_target != "N/A" else "null"},
            "rationale": "CNN MAC-BASED LEARNING CALCULATION: [Explain how you used the historical MAC analysis to determine channel ratio]. [Show mathematical reasoning linking channel_pruning_ratio to expected_achieved_macs]. [Justify importance criterion and round_to based on historical MAC performance]. [Explain how this achieves {tgt_str} +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%].",
            "architecture_type": "cnn",
            "learning_applied": true,
            "confidence_level": "high|medium|low",
            "learning_insights": [
                "Key insight 1 from historical MAC analysis",
                "Key insight 2 about channel ratio effectiveness vs MAC operations",
                "Key insight 3 about accuracy preservation at target MAC budget"
            ],
            "mac_calculation_details": {{
                "channel_ratio_to_macs_relationship": "YOUR_ANALYSIS_OF_HISTORICAL_PATTERN",
                "expected_mac_reduction": YOUR_CALCULATED_MAC_REDUCTION_VALUE,
                "accuracy_risk_assessment": "low|medium|high"
            }},
            "legacy": {{
                "target_ratio": {target_ratio:.4f}
            }}
        }}

        CRITICAL MAC-BASED INSTRUCTIONS:
        1. Use the MAC-based learning analysis above to make intelligent decisions2. Focus on achieving {tgt_str} +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% MAC operations, NOT parameter reduction percentages
        2. Focus on achieving {tgt_str} within +{macs_overshoot_tolerance_pct:.1f}% / -{macs_undershoot_tolerance_pct:.1f}% MAC operations, NOT parameter reduction percentages
        3. Prioritize channel ratios that historically achieved good MAC efficiency with preserved accuracy
        4. Do not repeat failed channel ratio patterns from MAC analysis
        5. Ensure expected_achieved_macs is within tolerance range: {target_macs - (target_macs * macs_undershoot_tolerance_pct / 100) if target_macs else 0:.2f}G - {target_macs + (target_macs * macs_overshoot_tolerance_pct / 100) if target_macs else 0:.2f}G
        6. Consider MAC efficiency and accuracy trade-offs from historical data"""

        return prompt
    
    def _format_cnn_analysis_for_llm(self, learning_analysis, target_macs, dataset, 
                                    baseline_macs=None, macs_overshoot_tolerance_pct=1.0, macs_undershoot_tolerance_pct=5.0):
        """Format the CNN learning analysis for the LLM (MAC-first with legacy fallback)."""
        def _fmt_pct(x):
            return f"{x:.1f}%" if isinstance(x, (int, float)) else "N/A%"
        def _fmt_g(x):
            return f"{x:.2f}G" if isinstance(x, (int, float)) else "N/A"

        # No history → show baseline guidance (updated with MAC context)
        if not learning_analysis.get('has_sufficient_data', False):
            rec = learning_analysis.get('recommendations', {})
            calc = rec.get('mathematical_calculation', {})
            
            # Enhanced MAC context for baseline guidance
            macs_context = ""
            if target_macs or baseline_macs:
                macs_context = f"""
    - Target MACs: {_fmt_g(target_macs)} (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
    - Baseline MACs: {_fmt_g(baseline_macs)}"""
                if baseline_macs and target_macs:
                    efficiency = (target_macs / baseline_macs) * 100
                    macs_context += f"""
    - MAC Efficiency Target: {efficiency:.1f}% of baseline"""
            
            return f"""
    📊 BASELINE CNN MAC GUIDANCE (No Historical Data):
    - Suggested channel ratio: {rec.get('suggested_channel_ratio', 0):.3f}
    - Suggested importance: {rec.get('suggested_importance_criterion', 'unknown')}
    - Suggested round_to: {rec.get('suggested_round_to', 'unknown')}{macs_context}
    - Reasoning: {', '.join(rec.get('reasoning', []))}
    """

        pattern = learning_analysis.get('pattern_analysis', {}) or {}
        failure = learning_analysis.get('failure_analysis', {}) or {}
        recommendations = learning_analysis.get('recommendations', {}) or {}
        total_attempts = learning_analysis.get('total_attempts', 0)

        # Prefer MAC metrics if present
        avg_achieved_macs = pattern.get('avg_achieved_macs')
        avg_macs_error_pct = pattern.get('avg_macs_error_pct')
        avg_channel_ratio = pattern.get('avg_channel_ratio', None)

        # Legacy fallbacks
        avg_achieved_ratio = pattern.get('avg_achieved_ratio')
        avg_deviation = pattern.get('avg_deviation')

        # Systematic flags (already computed by analyzer)
        sys_under = pattern.get('systematic_undershoot')
        sys_over = pattern.get('systematic_overshoot')

        # Build pattern header with MAC-first, then legacy if needed
        header_lines = []
        if avg_achieved_macs is not None or avg_macs_error_pct is not None:
            header_lines.append(f"- Average achieved MACs: {_fmt_g(avg_achieved_macs)} (target: {_fmt_g(target_macs)})")
            header_lines.append(f"- Average MAC error: {_fmt_pct(avg_macs_error_pct)} (tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)")
            
            # Add MAC efficiency if baseline is available
            if baseline_macs and avg_achieved_macs:
                efficiency = (avg_achieved_macs / baseline_macs) * 100
                target_efficiency = (target_macs / baseline_macs) * 100 if target_macs else 0
                header_lines.append(f"- Average MAC efficiency: {efficiency:.1f}% of baseline (target: {target_efficiency:.1f}%)")
        else:
            # Legacy fallback - but we need target_ratio, so we'll estimate from MAC if possible
            target_ratio_est = 1 - (target_macs / baseline_macs) if (baseline_macs is not None and target_macs is not None) else 0.3
            header_lines.append(f"- Average achieved: {avg_achieved_ratio*100:.1f}% (estimated target: {target_ratio_est*100:.1f}%)" if avg_achieved_ratio is not None else "- Average achieved: N/A")
            header_lines.append(f"- Average deviation: {avg_deviation*100:.1f}%" if avg_deviation is not None else "- Average deviation: N/A")

        if avg_channel_ratio is not None:
            header_lines.append(f"- Average channel ratio used: {avg_channel_ratio:.3f}")
        header_lines.append(f"- Systematic MAC undershoot: {sys_under}")
        header_lines.append(f"- Systematic MAC overshoot: {sys_over}")

        # Enhanced near-miss detection display
        has_near_miss = pattern.get('has_excellent_near_miss', False)
        high_acc_undershoots = pattern.get('high_accuracy_undershoots', [])
        if has_near_miss and high_acc_undershoots:
            best_near_miss = max(high_acc_undershoots, key=lambda x: x.get('accuracy', 0))
            header_lines.append(f"🎯 EXCELLENT NEAR-MISS FOUND: {best_near_miss.get('accuracy', 0):.1f}% accuracy")
            if best_near_miss.get('achieved_macs') and best_near_miss.get('target_macs'):
                achieved_g = best_near_miss['achieved_macs'] / 1e9
                target_g = best_near_miss['target_macs'] / 1e9
                error_pct = best_near_miss.get('macs_error_pct', 0)
                header_lines.append(f"   MAC performance: {achieved_g:.2f}G vs {target_g:.2f}G (error: {error_pct:.2f}%)")
            header_lines.append(f"   Strategy: channel_ratio={best_near_miss.get('channel_ratio', 0):.3f}, importance={best_near_miss.get('importance_criterion', 'unknown')}")

        # Effectiveness block: support MAC or legacy fields per-ratio
        eff_text = ""
        effectiveness = pattern.get('channel_ratio_effectiveness', {}) or {}
        for ratio, data in effectiveness.items():
            success_rate = data.get('success_rate', 0) * 100
            if 'avg_achieved_macs' in data or 'avg_macs_error_pct' in data:
                eff_text += (
                    f"  📊 Channel ratio {float(ratio):.3f} → "
                    f"Avg MACs: {_fmt_g(data.get('avg_achieved_macs'))}, "
                    f"MAC error: {_fmt_pct(data.get('avg_macs_error_pct'))}, "
                    f"Accuracy: {data.get('avg_accuracy', 0):.1f}%, "
                    f"Success: {success_rate:.0f}% "
                    f"({data.get('count', 0)} attempts)\n"
                )
            else:
                eff_text += (
                    f"  📊 Channel ratio {float(ratio):.3f} → "
                    f"Avg reduction: {data.get('avg_achieved', 0)*100:.1f}%, "
                    f"Accuracy: {data.get('avg_accuracy', 0):.1f}%, "
                    f"Success: {success_rate:.0f}% "
                    f"({data.get('count', 0)} attempts)\n"
                )

        # Failure stats with MAC context
        avg_undershoot_sev_pct = failure.get('avg_undershoot_severity_pct')
        if avg_undershoot_sev_pct is None:
            # Legacy field is a fraction; convert to %
            legacy_frac = failure.get('avg_undershoot_severity')
            avg_undershoot_sev_pct = legacy_frac * 100 if isinstance(legacy_frac, (int, float)) else None

        # Add MAC tolerance context to failures
        mac_tolerance_range = ""
        if target_macs:
            overshoot_g = target_macs * (macs_overshoot_tolerance_pct / 100.0)
            undershoot_g = target_macs * (macs_undershoot_tolerance_pct / 100.0)
            mac_tolerance_range = f" (tolerance: -{undershoot_g:.2f}G…+{overshoot_g:.2f}G)"


        text = f"""
    📊 COMPREHENSIVE CNN MAC-BASED LEARNING ANALYSIS FROM {total_attempts} ATTEMPTS:

    🔍 CHANNEL RATIO MAC PATTERN ANALYSIS:
    {chr(10).join(header_lines)}

    📈 CHANNEL RATIO MAC EFFECTIVENESS:
    {eff_text if eff_text else "  (No per-ratio breakdown available)"}\n

    ❌ MAC-BASED FAILURE ANALYSIS:
    - Total MAC failures: {failure.get('total_failures', 0)}
    - MAC undershoot failures: {len(failure.get('undershoot_failures', []))} (achieved > target{mac_tolerance_range})
    - MAC overshoot failures: {len(failure.get('overshoot_failures', []))} (achieved < target{mac_tolerance_range})
    - Catastrophic accuracy failures: {len(failure.get('catastrophic_failures', []))}
    - Average MAC undershoot severity: {_fmt_pct(avg_undershoot_sev_pct)}

    💡 INTELLIGENT CNN MAC RECOMMENDATIONS:
    - Suggested channel ratio: {recommendations.get('suggested_channel_ratio', 0):.3f}
    - Suggested importance: {recommendations.get('suggested_importance_criterion', 'unknown')}
    - Suggested round_to: {recommendations.get('suggested_round_to', 'unknown')}
    - Confidence level: {recommendations.get('confidence', 'low')}
    - Reasoning: {', '.join(recommendations.get('reasoning', []))}

    🧮 MAC-BASED MATHEMATICAL CALCULATION:
    {self._format_cnn_math_calculation(recommendations.get('mathematical_calculation', {}))}

    🎯 KEY CNN MAC LEARNING INSIGHTS:
    {chr(10).join('- ' + s for s in recommendations.get('learning_insights', []))}

    ⚠️ CRITICAL MAC-BASED INSTRUCTIONS:
    Primary Goal: Achieve {_fmt_g(target_macs)} +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% MAC operations
    - Target Range: {_fmt_g((target_macs * (1 - macs_undershoot_tolerance_pct / 100)) if target_macs else 0)} - {_fmt_g((target_macs * (1 + macs_overshoot_tolerance_pct / 100)) if target_macs else 0)}    
    - Use historical MAC patterns above to guide channel ratio selection
    - Prioritize channel ratios with high MAC success rates and good accuracy
    - Do not repeat failed channel-ratio patterns that missed MAC targets
    - Focus on MAC efficiency, not parameter reduction percentages
    """
        return text


    def _format_cnn_math_calculation(self, calc):
        """Format mathematical calculation details for CNN (MAC-first with legacy fallback)."""
        if not calc:
            return "No specific calculation available"

        def _fmt_pct(x):
            return f"{x:.1f}%" if isinstance(x, (int, float)) else "N/A%"
        def _fmt_g(x):
            return f"{x:.2f}G" if isinstance(x, (int, float)) else "N/A"

        # Handle high-accuracy tiny MAC adjustment strategy
        if calc.get('strategy_type') == 'high_accuracy_tiny_mac_adjustment':
            lines = [
                "Strategy: High-Accuracy Tiny MAC Adjustment",
                f"- Best accuracy found: {calc.get('best_accuracy_found', 0):.2f}%",
                f"- Original channel ratio: {calc.get('original_channel_ratio', 0):.3f}",
                f"- Correction factor: {calc.get('correction_factor', 1.0):.2f}",
                f"- Corrected channel ratio: {calc.get('corrected_channel_ratio', 0):.3f}"
            ]
            
            # Add MAC context if available
            if calc.get('achieved_macs') is not None:
                lines.extend([
                    f"- Best achieved MACs: {_fmt_g(calc.get('achieved_macs'))}",
                    f"- Target MACs: {_fmt_g(calc.get('target_macs'))}",
                    f"- MAC error: {_fmt_pct(calc.get('macs_error_pct'))}"
                ])
            elif calc.get('best_achieved_ratio') is not None:
                # Legacy fallback
                lines.extend([
                    f"- (Legacy) Best achieved ratio: {calc.get('best_achieved_ratio', 0)*100:.1f}%",
                    f"- (Legacy) Tiny deviation: {calc.get('tiny_deviation', 0)*100:.1f}%"
                ])
            
            return "\n".join(lines)

        # Baseline path (enhanced with MAC context)
        if calc.get('baseline_approach'):
            lines = [f"Strategy: Baseline MAC Allocation"]
            lines.append(f"- Target channel ratio: {calc.get('target_channel_pruning_estimate', 0):.3f}")
            
            # MAC information if available
            if any(k in calc for k in ('baseline_macs', 'target_macs', 'macs_overshoot_tolerance_pct', 'macs_undershoot_tolerance_pct', 'expected_achieved_macs')):
                lines.extend([
                    f"- Baseline MACs: {_fmt_g(calc.get('baseline_macs'))}",
                    f"- Target MACs: {_fmt_g(calc.get('target_macs'))} (+{calc.get('macs_overshoot_tolerance_pct', 1.0):.1f}%/-{calc.get('macs_undershoot_tolerance_pct', 5.0):.1f}%)",
                    f"- Expected achieved MACs: {_fmt_g(calc.get('expected_achieved_macs'))}"
                ])
                
                # Calculate and show MAC efficiency
                baseline_macs = calc.get('baseline_macs')
                target_macs = calc.get('target_macs')
                if baseline_macs and target_macs:
                    efficiency = (target_macs / baseline_macs) * 100
                    lines.append(f"- MAC efficiency target: {efficiency:.1f}% of baseline")
            else:
                # Legacy fallback
                lines.append(f"- Expected parameter reduction: {calc.get('expected_parameter_reduction', 0)*100:.1f}%")
            
            return "\n".join(lines)

        # Systematic correction strategy
        lines = []
        strategy_type = "Systematic MAC Correction"
        
        # Add strategy identification
        if calc.get('avg_channel_before') is not None:
            lines.append(f"Strategy: {strategy_type}")
            lines.append(f"- Previous average channel ratio: {calc.get('avg_channel_before', 0):.3f}")
        
        if calc.get('correction_factor') is not None:
            correction = calc.get('correction_factor', 1.0)
            correction_type = "increase" if correction > 1.0 else "decrease" if correction < 1.0 else "maintain"
            lines.append(f"- Correction factor applied: {correction:.2f} ({correction_type})")
            
            # Explain why this correction was chosen
            if correction > 1.4:
                lines.append("  → Aggressive correction for severe MAC undershoot")
            elif correction > 1.2:
                lines.append("  → Moderate correction for MAC undershoot")
            elif correction < 0.9:
                lines.append("  → Reduction for MAC overshoot")
        
        if calc.get('estimated_optimal') is not None:
            lines.append(f"- Estimated optimal ratio: {calc.get('estimated_optimal'):.3f}")
            lines.append("  → Based on historical MAC performance trend")
        
        if calc.get('base_channel_after_correction') is not None:
            lines.append(f"- Calculated channel ratio: {calc.get('base_channel_after_correction', 0):.3f}")
        
        if calc.get('safety_capped_channel') is not None:
            safety_cap = calc.get('safety_capped_channel', 0)
            base_calc = calc.get('base_channel_after_correction', safety_cap)
            if abs(safety_cap - base_calc) > 0.001:
                lines.append(f"- Safety-capped channel ratio: {safety_cap:.3f} (reduced for safety)")
            else:
                lines.append(f"- Final channel ratio: {safety_cap:.3f}")

        # MAC context section (enhanced)
        mac_lines = []
        if any(k in calc for k in ('baseline_macs', 'target_macs', 'expected_achieved_macs', 'macs_overshoot_tolerance_pct', 'macs_undershoot_tolerance_pct')):
            mac_lines.append("MAC Context:")
            mac_lines.append(f"- Baseline MACs: {_fmt_g(calc.get('baseline_macs'))}")
            mac_lines.append(f"- Target MACs: {_fmt_g(calc.get('target_macs'))} (+{calc.get('macs_overshoot_tolerance_pct', 1.0):.1f}%/-{calc.get('macs_undershoot_tolerance_pct', 5.0):.1f}%)")
            
            # Calculate asymmetric tolerance range
            target_macs = calc.get('target_macs')
            overshoot_pct = calc.get('macs_overshoot_tolerance_pct', 1.0)
            undershoot_pct = calc.get('macs_undershoot_tolerance_pct', 5.0)
            if target_macs:
                overshoot_g = target_macs * (overshoot_pct / 100)
                undershoot_g = target_macs * (undershoot_pct / 100)
                mac_lines.append(f"- Acceptable range: {target_macs - undershoot_g:.2f}G - {target_macs + overshoot_g:.2f}G")
            
            if calc.get('expected_achieved_macs') is not None:
                expected = calc.get('expected_achieved_macs')
                mac_lines.append(f"- Expected achieved MACs: {_fmt_g(expected)}")
                
                # Calculate expected error
                if target_macs:
                    expected_error = ((expected - target_macs) / target_macs) * 100
                    mac_lines.append(f"- Expected MAC error: {expected_error:+.1f}%")
            
            if calc.get('expected_macs_error_pct') is not None:
                mac_lines.append(f"- Expected MAC error: {_fmt_pct(calc.get('expected_macs_error_pct'))}")
            
            # Calculate MAC efficiency
            baseline = calc.get('baseline_macs')
            target = calc.get('target_macs')
            if baseline and target:
                efficiency = (target / baseline) * 100
                mac_lines.append(f"- Target MAC efficiency: {efficiency:.1f}% of baseline")

        # Combine all sections
        if mac_lines:
            if lines:
                lines.append("")  # Add separator
            lines.extend(mac_lines)

        # Legacy fallbacks (only if no MAC info present)
        if not any(k in calc for k in ('baseline_macs', 'target_macs', 'expected_achieved_macs')):
            if calc.get('expected_parameter_reduction') is not None:
                if lines:
                    lines.append("")
                lines.append("Legacy Context:")
                lines.append(f"- Expected parameter reduction: {calc.get('expected_parameter_reduction', 0)*100:.1f}%")

        return "\n".join(lines) if lines else "No specific calculation available"