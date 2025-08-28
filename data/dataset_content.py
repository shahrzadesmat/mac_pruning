# Helper function to generate MAC-based dataset-specific prompt content
def get_dataset_specific_content(dataset, num_classes, input_size, baseline_macs, target_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, accuracy_threshold):
    """Generate MAC-based dataset-specific content for prompts"""

    safe_baseline_macs = baseline_macs if baseline_macs is not None else 10.0
    safe_target_macs = target_macs if target_macs is not None else 5.0

    safe_macs_overshoot_tolerance_pct = macs_overshoot_tolerance_pct if macs_overshoot_tolerance_pct is not None else 1.0
    safe_macs_undershoot_tolerance_pct = macs_undershoot_tolerance_pct if macs_undershoot_tolerance_pct is not None else 5.0
    
    # Then use these safe variables in your calculations
    if safe_baseline_macs > 0:
        mac_efficiency_target = (safe_target_macs / safe_baseline_macs) * 100
        mac_reduction_needed = ((safe_baseline_macs - safe_target_macs) / safe_baseline_macs) * 100
    else:
        mac_efficiency_target = 50.0
        mac_reduction_needed = 50.0
        
    tolerance_range = f"{safe_target_macs - (safe_target_macs * safe_macs_undershoot_tolerance_pct / 100):.3f}G - {safe_target_macs + (safe_target_macs * safe_macs_overshoot_tolerance_pct / 100):.3f}G"
    
    if dataset.lower() == 'imagenet':
        return {
            'importance_criterion_override': 'taylor',  # Force Taylor for ImageNet MAC efficiency
            'complexity_level': 'high',
            'dataset_guidance': f"""
            ImageNet-specific MAC guidance (UPDATED MAC SAFETY LIMITS):
            - Target MAC budget: {target_macs:.3f}G (efficiency: {mac_efficiency_target:.1f}% of baseline)
            - Use VERY conservative MAC allocation (max 15% MLP MAC, 8% QKV MAC)
            - Taylor importance criterion REQUIRED for MAC-aware accuracy preservation
            - round_to=2 or null for optimal MAC measurement balance
            - Expect 1-2% Top-1 accuracy drop as acceptable at target MAC budget
            - Fine-tuning requires 3-5 epochs minimum for MAC-efficient models
            - CRITICAL: Never exceed 15% MLP MAC or 8% QKV MAC allocation
            """,
            'mac_allocation_guidance': f'ImageNet requires VERY conservative MAC allocation due to 1000-class complexity at {target_macs:.3f}G target',
            'round_to_guidance': 'ImageNet benefits from round_to=2 for MAC measurement stability and hardware efficiency',
            'conservative_approach': 'very conservative',
            'model_complexity': f'high (1000 classes, complex features, {safe_baseline_macs:.3f}G baseline MAC)',
            'dataset_strategy': f"""
            ImageNet MAC Strategy (UPDATED):
            - Maximum 15% of target MAC budget for MLP layers ({target_macs * 0.15:.3f}G HARD LIMIT)
            - Maximum 8% of target MAC budget for QKV attention ({target_macs * 0.08:.3f}G HARD LIMIT)
            - Preserve pretrained features while achieving {target_macs:.3f}G MAC target
            - Focus minimal MAC reduction on middle layers only
            - Protect early feature extraction MAC contributions
            - Use extremely gradual MAC allocation schedule
            """,
            'initial_mac_suggestion': target_macs * 1.1,  # 10% above target for safety
            'mac_rationale': f'very conservative start for ImageNet 1000-class complexity at {mac_efficiency_target:.1f}% efficiency',
            'mac_increment_size': f'{target_macs * 0.02:.3f}G',  # 2% of target MAC increments
            'round_to_explanation': 'round_to=2 recommended for ImageNet MAC measurement stability and hardware efficiency',
            'ignored_layers_guidance': f"""
            - Preserve final 1000-class classifier MAC operations (CRITICAL for {target_macs:.3f}G target)
            - Preserve first conv layer MAC contributions for low-level features (CRITICAL)
            - Protect all pretrained embedding MAC allocations (CRITICAL)
            - Preserve position embedding MAC operations for ViT models (CRITICAL)
            """,
            'conservatism_note': f'Be EXTREMELY conservative with ImageNet MAC allocation - 15% MLP/{target_macs * 0.15:.3f}G and 8% QKV/{target_macs * 0.08:.3f}G are HARD LIMITS!',
            'evaluation_metrics': 'Top-1 and Top-5 accuracy with MAC efficiency tracking',
            'primary_metric': 'Top-1 accuracy at target MAC budget',
            'recommended_lr': '0.001-0.003',  # More conservative for MAC-efficient models
            'recommended_epochs': '3-5',  # Fewer epochs for MAC-optimized training
            'recommended_batch_size': '32-64',
            'recommended_weight_decay': '1e-4',
            'recommended_optimizer': 'AdamW for ViT, SGD for CNN (MAC-optimized)',
            'data_augmentation_strategy': f"""
            ImageNet augmentation for MAC-efficient models (conservative):
            - Random resized crop (224x224) optimized for {target_macs:.3f}G MAC budget
            - Random horizontal flip (p=0.5)
            - MINIMAL augmentation during MAC-pruned model recovery
            - Standard ImageNet normalization
            - No aggressive augmentation that could mask MAC pruning effects
            """,
            'dataset_recommendations': f"""
            ImageNet-specific MAC recommendations (UPDATED):
            1. Learning rate: Start with 0.001-0.003 to preserve MAC-efficient pretrained features
            2. Optimizer: AdamW for ViT models, SGD with momentum for CNNs (MAC-optimized)
            3. Training duration: 3-5 epochs sufficient for MAC-efficient recovery
            4. Monitor Top-1 and Top-5 accuracy closely at {mac_efficiency_target:.1f}% MAC efficiency
            5. Use cosine annealing scheduler for MAC-optimized training
            6. CRITICAL: Stop if zero-shot accuracy < 40% at target MAC budget
            7. CRITICAL: Use Taylor importance for MAC-aware gradient-based pruning
            8. Validate MAC operations remain within {tolerance_range} throughout training
            """,
            'metrics_to_report': f"""
            1. MAC-aware accuracy metrics:
               - Top-1 accuracy on validation set at {target_macs:.3f}G MAC budget
               - Top-5 accuracy on validation set at target MAC efficiency
               - Accuracy change from original model (%) at {mac_efficiency_target:.1f}% MAC efficiency
               - Accuracy recovery after fine-tuning (%) while maintaining MAC budget
               - MAC allocation percentages (must stay under limits)
               - MAC efficiency: {mac_efficiency_target:.1f}% of baseline operations
            """,
            'dataset_analysis_requirements': f"""
            ImageNet MAC analysis requirements (UPDATED):
            1. Assess Top-1 accuracy vs. {accuracy_threshold}% threshold at {target_macs:.3f}G MAC budget
            2. Evaluate Top-5 accuracy performance at {mac_efficiency_target:.1f}% MAC efficiency
            3. Check for catastrophic accuracy collapse (< 10%) at target MAC budget
            4. Verify MAC allocation stays within safety limits ({tolerance_range})
            5. Compare to ImageNet MAC reduction benchmarks
            6. Analyze inference speed improvements at target MAC budget
            7. Validate MAC efficiency sustainability across batch sizes
            """,
            'success_criteria': f"""
            1. Top-1 accuracy ≥ {accuracy_threshold}% at {target_macs:.3f}G MAC budget
            2. MAC operations within +{safe_macs_overshoot_tolerance_pct:.1f}%/-{safe_macs_undershoot_tolerance_pct:.1f}% of target ({tolerance_range})
            3. Top-5 accuracy degradation < 2% at {mac_efficiency_target:.1f}% MAC efficiency
            4. No catastrophic accuracy collapse (> 10% final accuracy) at target MAC budget
            5. MLP MAC allocation ≤ 15%, QKV MAC allocation ≤ 8% of target budget
            6. MAC efficiency target achieved: {mac_efficiency_target:.1f}% of baseline operations
            """,
            'failure_scenarios': f"""
            1. Top-1 accuracy drops below 40% at {target_macs:.3f}G MAC budget (catastrophic)
            2. Zero-shot accuracy < 10% at target MAC efficiency (severe over-pruning)
            3. MLP MAC > 15% or QKV MAC > 8% of target budget (safety violation)
            4. MAC operations outside {tolerance_range} tolerance (target miss)
            5. Model becomes unstable or produces NaN outputs at target MAC budget
            6. No meaningful inference speedup achieved despite MAC reduction
            """
        }

    else:  # CIFAR-10
        return {
            'complexity_level': 'moderate',
            'dataset_guidance': f"""
            CIFAR-10-specific MAC guidance:
            - Target MAC budget: {target_macs:.3f}G (efficiency: {mac_efficiency_target:.1f}% of baseline)
            - Can use aggressive MAC allocation (50% MLP MAC, 30% QKV MAC)
            - L1/L2 norm importance often sufficient for MAC optimization
            - round_to=1 for maximum MAC allocation flexibility
            - Expect <1% accuracy drop as target at {mac_efficiency_target:.1f}% MAC efficiency
            - Short fine-tuning (1-3 epochs) sufficient for MAC-efficient models
            """,
            'mac_allocation_guidance': f'CIFAR-10 allows aggressive MAC allocation due to simplicity at {target_macs:.3f}G target',
            'round_to_guidance': 'CIFAR-10 benefits from round_to=1 for fine-grained MAC control',
            'conservative_approach': 'moderate',
            'model_complexity': f'moderate (10 classes, simple features, {safe_baseline_macs:.3f}G baseline MAC)',
            'dataset_strategy': f"""
            CIFAR-10 MAC Strategy:
            - Aggressive MAC allocation acceptable up to 50% MLP, 30% QKV
            - Focus on maintaining basic feature extraction while achieving {target_macs:.3f}G target
            - Final classifier can handle significant MAC reduction
            - Faster convergence expected at {mac_efficiency_target:.1f}% MAC efficiency
            """,
            'initial_mac_suggestion': target_macs * 0.9,  # 10% below target for aggressive start
            'mac_rationale': f'aggressive start acceptable for CIFAR-10 at {mac_efficiency_target:.1f}% efficiency',
            'mac_increment_size': f'{target_macs * 0.05:.3f}G',  # 5% of target MAC increments
            'round_to_explanation': 'round_to=1 recommended for CIFAR-10 fine-grained MAC control',
            'ignored_layers_guidance': f"""
            - Preserve final 10-class classifier MAC operations for {target_macs:.3f}G target
            - First conv layer can contribute to MAC reduction more aggressively
            - Focus on maintaining sufficient representational MAC capacity
            """,
            'conservatism_note': f'CIFAR-10 allows more aggressive MAC allocation than ImageNet at {target_macs:.3f}G target.',
            'evaluation_metrics': 'accuracy with MAC efficiency tracking',
            'primary_metric': 'accuracy at target MAC budget',
            'recommended_lr': '0.01-0.05',
            'recommended_epochs': '1-3',
            'recommended_batch_size': '64-128',
            'recommended_weight_decay': '5e-4',
            'recommended_optimizer': 'SGD with momentum (MAC-optimized)',
            'data_augmentation_strategy': f"""
            CIFAR-10 augmentation for MAC-efficient models:
            - Random crops and horizontal flips optimized for {target_macs:.3f}G MAC budget
            - Standard CIFAR-10 normalization  
            - Resize to 224x224 for model compatibility with MAC measurements
            - Moderate augmentation during MAC-efficient fine-tuning
            """,
            'dataset_recommendations': f"""
            CIFAR-10-specific MAC recommendations:
            1. Learning rate: 0.01-0.05 for quick convergence at {mac_efficiency_target:.1f}% MAC efficiency
            2. Optimizer: SGD with momentum (0.9) and weight decay (5e-4) for MAC-optimized models
            3. Training duration: 1-3 epochs typically sufficient for MAC-efficient recovery
            4. Monitor accuracy on validation set at {target_macs:.3f}G MAC budget
            5. Use step or cosine annealing scheduler for MAC-optimized training
            6. Simple augmentation strategy that doesn't interfere with MAC measurements
            7. Validate MAC operations remain within {tolerance_range} during training
            """,
            'metrics_to_report': f"""
            1. MAC-aware accuracy metrics:
               - Accuracy on test set at {target_macs:.3f}G MAC budget
               - Accuracy change from original model (%) at {mac_efficiency_target:.1f}% MAC efficiency
               - Accuracy recovery after fine-tuning (%) while maintaining MAC budget
               - MAC efficiency: {mac_efficiency_target:.1f}% of baseline operations
            """,
            'dataset_analysis_requirements': f"""
            CIFAR-10 MAC analysis requirements:
            1. Assess accuracy vs. {accuracy_threshold}% threshold at {target_macs:.3f}G MAC budget
            2. Check for class-specific degradation at {mac_efficiency_target:.1f}% MAC efficiency
            3. Compare to CIFAR-10 MAC reduction benchmarks
            4. Evaluate overfitting indicators in MAC-efficient models
            5. Analyze training efficiency gains at target MAC budget
            6. Validate MAC operations consistency across epochs
            """,
            'success_criteria': f"""
            1. Accuracy ≥ {accuracy_threshold}% at {target_macs:.3f}G MAC budget
            2. MAC operations within +{safe_macs_overshoot_tolerance_pct:.1f}%/-{safe_macs_undershoot_tolerance_pct:.1f}% of target ({tolerance_range})
            3. Minimal accuracy degradation (<2%) at {mac_efficiency_target:.1f}% MAC efficiency
            4. Good training efficiency improvements at target MAC budget
            5. MAC efficiency target achieved: {mac_efficiency_target:.1f}% of baseline operations
            """,
            'failure_scenarios': f"""
            1. Accuracy drops below 80% at {target_macs:.3f}G MAC budget
            2. MAC operations outside {tolerance_range} tolerance (target miss)
            3. Severe overfitting observed in MAC-efficient model
            4. Training becomes unstable at target MAC budget
            5. No meaningful efficiency gains despite MAC reduction
            """
        }

