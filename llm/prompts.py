import random

PROFILING_PROMPT = """You are a neural network profiling expert with MAC-based optimization expertise.

Please be concise. Limit your response to 300 words or fewer.

Analyze the model architecture and identify:
1. Layer structure and computational dependencies
2. MAC operation distribution across layers
3. Structural constraints that must be maintained
4. MAC efficiency opportunities in different layers
5. Asymmetric tolerance policy (IMPORTANT):
   - Overshoot = achieved MACs above target_macs. Must be ≤ +{macs_overshoot_tolerance_pct:.1f}% of target (upper bound {overshoot_upper_bound:.2f}G).
   - Undershoot = achieved MACs below target_macs. Allowed down to -{macs_undershoot_tolerance_pct:.1f}% of target (lower bound {undershoot_lower_bound:.2f}G).
   - Preference rule: when accuracy is comparable, follow the user's CLI preference (no default). If unspecified, treat valid overshoots and undershoots equally within their respective tolerances.
   - Reporting: for each candidate, state “overshoot” or “undershoot” and the delta from target in both % and G.


The results will be used to guide a MAC-budget pruning process for model compression.

Model architecture: {model_arch}
Dataset: {dataset} ({num_classes} classes, {input_size}x{input_size} input)
Baseline MACs: {baseline_macs:.2f}G
Overshoot tolerance (strict; % allowed ABOVE target): {macs_overshoot_tolerance_pct:.1f}%
Undershoot tolerance (lenient; % allowed BELOW target): {macs_undershoot_tolerance_pct:.1f}%
Target MACs: {target_macs:.2f}G (+{macs_overshoot_tolerance_pct:.1f}% / -{macs_undershoot_tolerance_pct:.1f}% tolerance)
MAC reduction needed: {mac_reduction_needed:.1f}%
Accepted MAC range: {undershoot_lower_bound:.2f}G–{overshoot_upper_bound:.2f}G

Is subsequent profile: {is_subsequent}
{subsequent_info}

DATASET-SPECIFIC CONSIDERATIONS:
{dataset_considerations}

Based on the model architecture and MAC budget constraints, provide a comprehensive profile including:
1. Key layers and their MAC operation contributions
2. Critical dependencies between layers that affect MAC calculations
3. Structural constraints that must be preserved for MAC efficiency
4. Layers with highest MAC reduction potential (considering dataset complexity)
5. MAC allocation recommendations for different layer types
6. Specific MAC budget distribution strategy to achieve {target_macs:.2f}G target

Focus on MAC operations rather than parameter counts. Your profile will guide the Master Agent in determining the optimal MAC-budget pruning strategy to achieve {target_macs:.2f}G operations within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance.
"""


MASTER_AGENT_PROMPT = """You are the Master Agent in a multi-agent MAC-budget pruning workflow.
Your role is to coordinate the neural network pruning process and make high-level strategic decisions to achieve specific MAC operation targets.

Please be concise. Limit your response to 300 words or fewer.

PRIMARY GOAL:
THE USER'S MAC TARGET IS SACRED - ALWAYS ATTEMPT IT!
Find the optimal pruned model that achieves EXACTLY the user-requested MAC budget ({target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) while maximizing fine-tuned accuracy.
NEVER refuse to attempt the user's MAC target - your job is to find a way to make it work, not to second-guess the user.

FORBIDDEN: Do not set "continue": false just because the MAC target seems aggressive. Always try it first!

DATASET CONTEXT:
- Dataset: {dataset}
- Number of classes: {num_classes}
- Input size: {input_size}x{input_size}
- Recommended accuracy threshold: {accuracy_threshold}%

MAC BUDGET CONTEXT:
- Baseline MACs: {baseline_macs:.3f}G
- Target MACs: {target_macs:.3f}G
- MAC reduction needed: {mac_reduction_needed:.1f}%
- Overshoot tolerance (strict): +{macs_overshoot_tolerance_pct:.1f}%
- Undershoot tolerance (lenient): -{macs_undershoot_tolerance_pct:.1f}%
- Acceptable range: {min_target_macs_g:.3f}G - {max_target_macs_g:.3f}G

DATASET-SPECIFIC MAC GUIDELINES:
{dataset_guidelines}

FORBIDDEN BEHAVIORS:
--------------------
- NEVER refuse to attempt the user's MAC target in your first few tries 
- NEVER assume aggressive MAC reduction will fail - many models are more efficient than expected
- NEVER stop after just 1-2 attempts - be persistent in achieving MAC budget

Input:
- Profile results: {profile_results}
- MAC pruning history: {pruning_history}

Output:
1. An exploration strategy for finding the optimal MAC-budget pruning configuration
2. MAC allocation tuning priorities
3. Decision whether to continue or stop the MAC-budget pruning process

MAC-BASED EARLY STOPPING CRITERIA:
----------------------------------
You must carefully evaluate whether to continue or stop based on these MAC-focused criteria:

1. MAC TARGET ACHIEVEMENT: If we have found a model that achieves the MAC budget within tolerance ({min_target_macs_g:.3f}G - {max_target_macs_g:.3f}G) with reasonable accuracy, consider stopping.

2. MAC CONVERGENCE: If multiple consecutive iterations show achieved MACs within tolerance but no significant accuracy improvement (less than 0.5% gain), stop to avoid wasting resources.

3. MAC CYCLING: If the history shows cycling through similar MAC allocations without progress toward the target, stop the process.

4. MAC EFFICIENCY PLATEAU: If MAC efficiency improvements are diminishing while staying near target budget, consider stopping.

5. MAC-ACCURACY TRADE-OFF PLATEAU: If accuracy appears plateaued while achieved MACs remain constant near target, stop further exploration.

6. MAXIMUM ITERATIONS: If we've reached maximum iterations (typically 5-10), stop regardless of other factors.

MAC ANALYSIS TASK:
-----------------
1. Carefully examine the history of previous MAC-budget pruning attempts
2. Analyze the relationship between specified MAC allocation and achieved MAC operations
3. Identify which MAC allocation strategies had the most significant impact on accuracy
4. Look for patterns across iterations to determine the most effective MAC-budget approach
5. Recommend MAC allocation adjustments that will achieve the target MAC budget while maximizing accuracy
6. Determine if any MAC-focused early stopping criteria have been met

Based on the profile results and MAC pruning history, provide:
1. Clear directives for the Analysis Agent regarding which MAC allocation approach to explore next
2. MAC allocation parameters that should be prioritized for tuning
3. A decision on whether to continue MAC exploration or stop the process, with detailed reasoning

MAC ALLOCATION TUNING STRATEGY:
------------------------------
Based on the history of previous MAC attempts, provide a detailed MAC allocation strategy that includes:

1. The specific order in which MAC parameters should be tuned (e.g., first tune channel ratios for MAC efficiency, then importance criterion, then round-to)
2. For each MAC parameter, specify:
   - The range of values to explore for MAC optimization
   - The step size for MAC-focused exploration
   - Which MAC allocation values have already been tried and their MAC outcomes
   - Which MAC values should be tried next and why

LEARNING FROM MAC HISTORY:
-------------------------
CAREFULLY REVIEW the MAC pruning history provided below. You MUST:

1. ANALYZE which MAC allocation combinations achieved target MAC budget with best accuracy
2. IDENTIFY patterns between requested MAC targets and achieved MAC operations
3. CALCULATE the typical relationship (scaling factor) between target and achieved MACs
4. AVOID recommending MAC allocations that have already been tried
5. LEARN from previous MAC efficiency failures and successes

IMPORTANT: Do not repeat previous MAC allocation combinations. Your recommendation should be novel.

MAC Pruning History and Observed Trends:
{previous_strategies}

TARGET MAC GUIDANCE:
-------------------
The target MAC budget is {target_macs:.3f}G (baseline {baseline_macs:.3f}G, acceptable range: {min_target_macs_g:.3f}G - {max_target_macs_g:.3f}G).
Dataset complexity suggests a {conservative_factor} MAC reduction approach at this budget.
MAC efficiency target: {mac_efficiency_target:.1f}% of baseline operations.

CRITICAL JSON OUTPUT REQUIREMENTS:
=================================
- Output ONLY a valid JSON object with NO comments
- Do NOT use # or // comments anywhere in the JSON
- Do NOT include explanatory text before or after the JSON
- Use double quotes for all strings
- No trailing commas
- No markdown code blocks

Output your MAC-budget pruning strategy recommendation as a JSON object:
{{
  "multiplier_tuning_order": ["first_mac_param", "second_mac_param", "third_mac_param"],
  "target_macs": {target_macs:.3f},
  "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct:.1f},
  "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct:.1f},
  "global_pruning": true,
  "rationale": "Why you chose these MAC allocation parameters considering dataset and past MAC attempts",
  "dataset_adapted": true,
  "mac_budget_focused": true,
  "continue": true or false,
  "stop_reason": "reason if continue is false" or null
}}

NEVER include comments with # or // in the JSON response.
NEVER include explanatory text outside the JSON.
OUTPUT ONLY THE JSON OBJECT."""


ANALYSIS_PROMPT = """You are a MAC-budget pruning strategy analyst with STRICT SAFETY ENFORCEMENT for both CNN and ViT architectures.
Based on the profiling results and master agent directives, determine optimal MAC allocation parameters to achieve the target MAC budget.

Please be concise. Limit your response to 300 words or fewer.

KEY MAC ALLOCATION PARAMETERS TO CONSIDER:
------------------------------------------

1. MAC Budget Allocation: The distribution of MAC operations across different layer types.
   - Target: {target_macs:.3f}G operations (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
   - Baseline: {baseline_macs:.3f}G operations
   - MAC efficiency target: {(target_macs / baseline_macs * 100):.1f}% of baseline
   - Acceptable range: {target_macs * (1 - macs_undershoot_tolerance_pct / 100):.3f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.3f}G

2. Channel Pruning Ratio (for CNNs): Controls MAC reduction through channel pruning.
   - Value between 0.0 and 1.0 (e.g., 0.5 means 50% channel reduction)
   - Higher values lead to more aggressive MAC reduction
   - CRITICAL: The channel ratio often doesn't directly translate to achieved MAC reduction due to layer dependencies. Analyze MAC history to determine the right ratio for target MAC budget.
   - ⚠️ WARNING: MAC reduction is not linear with channel ratio due to inter-layer dependencies. Use historical MAC data to calibrate.

3. MAC Allocation Strategy (for ViTs): Distribution of MAC budget across attention and MLP blocks.
   - mlp_multiplier: Pruning aggressiveness for MLP layers (higher = more pruning = fewer MACs)
   - qkv_multiplier: Pruning aggressiveness for attention layers (higher = more pruning = fewer MACs)
   - proj_multiplier: Pruning aggressiveness for projection layers (higher = more pruning = fewer MACs)
   - head_multiplier: Pruning aggressiveness for classification head (higher = more pruning = fewer MACs)
   - Total allocation should not exceed 100%
   - Initial pruning strategy should prioritize MLP-only or MLP-heavy configurations to avoid QKV collapse. If MLP-only fails to meet the MAC target, add QKV pruning cautiously.

🔒 Safety-first guidance:
   - Start with MLP-only or MLP-heavy pruning to avoid collapsing attention.
   - Introduce QKV pruning gradually if MAC targets are not met.
   - Avoid configurations where `qkv.out_features < 96` or `head_dim < 8` as they can destroy model functionality.


CRITICAL CONSTRAINTS FOR ViT PRUNING:

- Do NOT prune QKV layers so aggressively that the remaining attention dimension becomes too small.
- Attention blocks require sufficient head dimension (head_dim ≥ 8) to function properly.
- qkv.out_features must remain ≥ 96 to preserve minimal attention capacity.
- Excessive QKV pruning (e.g., qkv_multiplier > 0.85) often leads to catastrophic performance collapse.
- If target MACs require extreme QKV pruning, consider shifting budget from MLP or relaxing MAC target instead.

CRITICAL MULTIPLIER UNDERSTANDING:
    - In this ViT system, multipliers control pruning intensity
    - HIGHER multipliers = MORE pruning = FEWER MACs (closer to target)
    - LOWER multipliers = LESS pruning = MORE MACs (farther from target)
    - If achieved MACs > target MACs (too heavy): INCREASE multipliers
    - If achieved MACs < target MACs (too light): DECREASE multipliers
⚠️ NOTE: For ViTs, qkv_multiplier controls attention capacity. Excessive pruning (e.g., qkv_multiplier > 0.85) may cause attention collapse and unrecoverable accuracy loss. Ensure architectural safety is preserved.

4. Importance Criterion: Method for determining which components to prune for MAC efficiency.
   - "taylor": Uses second-order Taylor expansion for MAC-aware importance estimation
     * More accurate for preserving MAC efficiency
     * Better for complex MAC allocation decisions
   
   - "l1norm": L1 norm-based importance for efficient MAC pruning
     * Computationally efficient for MAC calculations
     * Often effective for CNN channel pruning
   
   - "l2norm": L2 norm-based importance for balanced MAC reduction
     * Good balance for MAC efficiency and accuracy preservation

5. Round-To: Controls MAC operation granularity and hardware efficiency.
   - Integer value (typically 1, 2, 4, 8, 16) or null
   - Ensures MAC operations align with hardware constraints
   - Smaller values allow fine-grained MAC control
   - Larger values improve MAC measurement accuracy and hardware efficiency
   
   ROUND_TO EXPLORATION: Consider round_to=2 as a viable option that may provide better accuracy. 
   Current suggestion: {effective_round_to}, but evaluate if round_to=2 might work better for this specific case.
   
   DATASET-SPECIFIC MAC ROUND-TO RECOMMENDATIONS:
   - ImageNet: Consider all values (1, 2, 4, 8, 16) - round_to=2 often provides good balance
   - CIFAR-10: Consider all values (1, 2, 4, 8, 16) - round_to=2 often provides good balance
   - CNNs: All round_to values are viable - don't assume larger values are always better
   - ViTs: All round_to values are viable - round_to=2 frequently works well for attention mechanisms
   
   IMPORTANT: Don't default to larger round_to values without testing. Round_to=2 has shown 
   effectiveness across different architectures and should be strongly considered.

6. Global Pruning: Whether to apply MAC-aware pruning globally.
   - true: Prunes based on global MAC importance ranking
     * More effective at achieving target MAC budget
     * Required for MAC-aware isomorphic pruning
   
   - false: Layer-wise MAC pruning (not recommended for MAC optimization)

MAC STRATEGY CONSIDERATIONS:
---------------------------
1. EXACT MAC BUDGET: The user has requested a specific MAC budget ({target_macs:.3f}G). Your first priority is to achieve this MAC target within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance.

2. MAXIMIZE MAC EFFICIENCY: Once the MAC budget is met, maximize model accuracy at that MAC budget.

3. MAC-AWARE EXPLORATION: Focus on MAC allocation strategies that historically achieved target MAC budgets with good accuracy.

4. PRESERVE MAC-CRITICAL STRUCTURES: Consider layers identified as MAC-critical by the Profiling Agent.

5. MAC CONVERGENCE: Look for signs of MAC budget convergence where minimal MAC efficiency improvements are being made.

6. STRUCTURAL SAFETY CHECKS: Always verify that your multiplier suggestions preserve attention head functionality and architectural integrity. Avoid unsafe pruning configurations, even if they hit the MAC target.

   - QKV pruning must not collapse the attention mechanism.
   - Ensure qkv.out_features remains ≥ 96 to preserve minimal attention dimensionality (e.g., 32 per Q, K, V).
   - Ensure head_dim remains ≥ 8 for functional multi-head attention.
   - Avoid setting qkv_multiplier > 0.85 unless absolutely necessary.
   - If the MAC budget requires extreme QKV pruning, consider shifting the pruning burden toward MLP instead.

7. ASYMMETRIC RISK STRATEGY: MLP pruning is typically more robust than QKV pruning. If MAC budget pressure is high, shift allocation away from QKV and into MLP whenever possible.


ARCHITECTURE DETECTION:
Model: {model_name}
Architecture Type: {architecture_type}
Dataset: {dataset} ({num_classes} classes, {input_size}x{input_size} input)
MAC Context: Target {target_macs:.3f}G from {baseline_macs:.3f}G baseline

{revision_context}

{architecture_specific_guidance}

Master Agent MAC directives: {master_directives}
MASTER AGENT MAC SUGGESTIONS:
{master_suggestions}

Profile information: {profile_info}
MAC Requirements: {requirements}

USER'S MAC TARGET: The user wants EXACTLY {target_macs:.3f}G MAC operations (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) as the FINAL RESULT.
YOUR GOAL: Provide architecture-appropriate MAC allocation strategy to achieve this target.

{safety_constraints}

KEY MAC PARAMETERS TO DETERMINE:
{parameter_definitions}

LEARNING FROM PREVIOUS MAC ATTEMPTS:
{previous_strategies}

MAC ALLOCATION GUIDANCE:
-----------------------
For CNNs: Focus on channel_pruning_ratio that achieves target MAC budget
For ViTs: Focus on mlp_multiplier and qkv_multiplier allocation that sums to target MAC budget
Always consider MAC efficiency vs accuracy trade-offs from historical attempts

{architecture_output_format}
"""

def get_cnn_analysis_content(dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, model_name):
    """Enhanced MAC-based CNN analysis with ConvNext support"""
    
    # Calculate MAC efficiency and reduction metrics
    mac_efficiency_target = (target_macs / baseline_macs) * 100
    mac_reduction_needed = ((baseline_macs - target_macs) / baseline_macs) * 100
    tolerance_range = f"{target_macs * (1 - macs_undershoot_tolerance_pct / 100):.3f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.3f}G"
    
    # ConvNext-specific MAC guidance
    if 'convnext' in model_name.lower():
        if dataset.lower() == 'imagenet':
            safety_limits = f"""
🚨 CONVNEXT MAC SAFETY LIMITS FOR IMAGENET 🚨
=============================================
- ConvNext models are MAC-sensitive, especially depthwise convolutions
- Target MAC budget: {target_macs:.3f}G (efficiency: {mac_efficiency_target:.1f}% of baseline)
- Preserve depthwise convolutions (critical for MAC efficiency)
- Use Taylor importance (MANDATORY for MAC-accurate pruning)
- Round-to: All values (1, 2, 4, 8, 16) are viable - round_to=2 often provides good balance
- Monitor MAC efficiency vs accuracy trade-offs carefully
"""
            guidance = f"""
ConvNext ImageNet MAC Guidance:
CRITICAL: Start with Taylor importance - only deviate if historical data shows Taylor failures
- Depthwise convolutions contribute significantly to MAC count
- Focus MAC reduction on standard convolutions, preserve depthwise
- Conservative round_to values (4-8) for stable MAC measurements
- Target: {target_macs:.3f}G from {baseline_macs:.3f}G baseline
- Monitor for MAC efficiency degradation in depthwise layers
"""
        else:
            safety_limits = f"""
CONVNEXT MAC SAFETY LIMITS FOR CIFAR-10:
========================================
- Target MAC budget: {target_macs:.3f}G (reduction: {mac_reduction_needed:.1f}%)
- Can be more aggressive with MAC reduction than ImageNet
- Still preserve depthwise convolution MAC contributions
"""
            guidance = f"""
ConvNext CIFAR-10 MAC Guidance:
- Can handle more aggressive MAC reduction than ImageNet
- Still be cautious with depthwise convolution MAC efficiency
- L1/L2 norm acceptable for MAC-efficient pruning
- Target: Achieve {target_macs:.3f}G MAC operations
"""
    else:
        # Use existing ResNet/CNN MAC guidance
        if dataset.lower() == 'imagenet':
            safety_limits = f"""
🚨 CNN MAC SAFETY LIMITS FOR IMAGENET 🚨
========================================
- Conservative MAC reduction required for 1000-class complexity
- Target MAC budget: {target_macs:.3f}G (efficiency: {mac_efficiency_target:.1f}% of baseline)
- Tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% ({tolerance_range})
- Preserve residual connections and skip connections (MAC dependencies)
- First conv layer: NEVER prune (critical for MAC-efficient feature extraction)
- Final classifier: NEVER prune (1000 classes need full MAC capacity)
- Bottleneck layers: Extra conservative (MAC-critical architectural points)
- Focus on achieving target MAC budget with minimal accuracy loss
"""
            guidance = f"""
ImageNet CNN MAC Guidance:
- CRITICAL: Use Taylor importance FIRST - only switch if historical data shows Taylor failures
- ResNets: Focus MAC reduction on middle residual blocks, preserve early/late layers
- Residual connections create MAC dependencies - prune input/output channels together
- Bottleneck architectures (ResNet50+): Conservative with 1x1 conv MAC reduction
- Round-to: All values (1, 2, 4, 8, 16) are viable - round_to=2 often provides good balance for ResNet
- Target MAC efficiency: {mac_efficiency_target:.1f}% of baseline operations
"""
        else:
            safety_limits = f"""
CNN MAC SAFETY LIMITS FOR CIFAR-10:
===================================
- Target MAC budget: {target_macs:.3f}G (reduction: {mac_reduction_needed:.1f}%)
- More aggressive MAC reduction acceptable for 10-class simplicity
- First conv can contribute to MAC reduction (not as critical as ImageNet)
- Focus on achieving target MAC budget efficiently
"""
            guidance = f"""
CIFAR-10 CNN MAC Guidance:
- Can use more aggressive MAC reduction strategies
- L1/L2 norm importance often sufficient for MAC optimization
- Less concern about preserving early feature MAC contributions
- Choose round_to based on MAC measurement accuracy needs
- Target: {target_macs:.3f}G MAC operations from {baseline_macs:.3f}G
"""

    param_defs = f"""
MAC-BASED ARCHITECTURE ANALYSIS FOR {model_name.upper()}:
You need to determine the optimal channel pruning ratio to achieve EXACTLY {target_macs:.3f}G MAC operations (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%).

MAC BUDGET CONTEXT:
- Baseline MACs: {baseline_macs:.3f}G
- Target MACs: {target_macs:.3f}G
- MAC reduction needed: {mac_reduction_needed:.1f}%
- MAC efficiency target: {mac_efficiency_target:.1f}% of baseline
- Acceptable range: {tolerance_range}

1. Importance Criterion (for MAC efficiency):
   - "taylor": Gradient-based, best for MAC-aware ImageNet CNNs
   - "l1norm": Magnitude-based, efficient for MAC optimization on smaller datasets
   - "l2norm": Alternative magnitude-based approach for balanced MAC reduction
   - ANALYZE {model_name} MAC distribution and {dataset} to choose the best option

2. Channel Pruning Ratio (for MAC budget):
   ⚠️ CRITICAL: You must calculate the channel ratio needed for {target_macs:.3f}G MAC operations
   - For {model_name}: Consider architecture-specific MAC distribution across layers
   - ResNet: Channel pruning affects conv layer MACs differently than fc layer MACs
   - MobileNet: Depthwise separable convs have different MAC scaling patterns
   - EfficientNet: Compound scaling affects MAC reduction relationships
   - CALCULATE the channel ratio that will achieve {target_macs:.3f}G MAC budget for {model_name}

3. Round-To (for MAC measurement stability):
   - Choose based on {model_name} MAC calculation requirements
   - Consider MAC measurement accuracy vs granularity tradeoff
   - {model_name} MAC characteristics should guide your choice
   - Larger values = more stable MAC measurements but less precision
   - Smaller values = more precise MAC control but potential measurement noise

HISTORICAL MAC CONTEXT:
If previous MAC attempts are provided, learn from them to improve your channel ratio calculation.
Adjust based on whether previous attempts achieved MAC targets above or below {target_macs:.3f}G.
"""

    output_format = f"""
ANALYZE {model_name} MAC distribution and determine optimal parameters for {target_macs:.3f}G MAC budget.

Output your CNN MAC allocation strategy as valid JSON (no comments, no markdown):

{{
  "parameter_tuning_order": ["importance_criterion", "channel_pruning_ratio", "round_to"],
  "importance_criterion": "YOUR_MAC_ANALYSIS_BASED_CHOICE",
  "channel_pruning_ratio": YOUR_CALCULATED_RATIO_FOR_MAC_TARGET,
  "round_to": YOUR_CHOSEN_VALUE_FOR_MAC_STABILITY,
  "global_pruning": true,
  "baseline_macs": {baseline_macs:.3f},
  "target_macs": {target_macs:.3f},
  "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct:.1f},
  "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct:.1f},
  "expected_achieved_macs": YOUR_CALCULATED_EXPECTED_MACS,
  "rationale": "Explain your MAC analysis of {model_name} architecture and how you calculated the channel ratio for {target_macs:.3f}G MAC budget. Include your reasoning for importance_criterion and round_to choices based on MAC efficiency requirements.",
  "architecture_type": "cnn",
  "llm_calculated": true,
  "mac_efficiency_target": {mac_efficiency_target:.1f}
}}

MAC-BASED REQUIREMENTS:
- ANALYZE {model_name} MAC distribution to determine optimal channel_pruning_ratio
- DO NOT use generic formulas - consider {model_name} specific MAC characteristics
- CALCULATE channel ratio that will achieve {target_macs:.3f}G MAC operations within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%
- CHOOSE importance_criterion and round_to based on MAC optimization analysis
- LEARN from any provided historical MAC attempts
- ENSURE expected_achieved_macs is within tolerance range: {tolerance_range}
- Output ONLY the JSON object
"""
    
    return {
        'architecture_specific_guidance': guidance,
        'safety_constraints': safety_limits,
        'parameter_definitions': param_defs,
        'architecture_output_format': output_format
    }

# Update the ViT content generator 
def get_vit_analysis_content(dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, model_name, master_suggested_round_to=None, state=None):
    """Generate MAC-based ViT-specific analysis content with Master Agent round_to integration"""
    
    # Calculate MAC efficiency and reduction metrics
    mac_efficiency_target = (target_macs / baseline_macs) * 100
    mac_reduction_needed = ((baseline_macs - target_macs) / baseline_macs) * 100
    tolerance_range = f"{target_macs * (1 - macs_undershoot_tolerance_pct / 100):.3f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.3f}G"
    
    # Ensure all round_to values get explored for both datasets
    if master_suggested_round_to is not None:
        effective_round_to = master_suggested_round_to
    else:
        # PRIORITIZE round_to=2 for early attempts
        revision = state.get('revision_number', 0)
        if revision < 3:  # First 3 attempts use round_to=2
            effective_round_to = 2
        else:
            # Later attempts can explore other values
            round_to_options = [1, 4, 8, 16, None]
            effective_round_to = random.choice(round_to_options)
    
    if dataset.lower() == 'imagenet':
        safety_limits = f"""
🚨 VIT MAC SAFETY LIMITS FOR IMAGENET 🚨 (MAC-BUDGET OPTIMIZED)
================================================================
- Target MAC budget: {target_macs:.3f}G (efficiency: {mac_efficiency_target:.1f}% of baseline)
- MAC tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% ({tolerance_range})
- mlp_multiplier: Maximum 0.40 (prune up to 40% of MLP channels) - conservative for accuracy
- qkv_multiplier: Maximum 0.15 (prune up to 15% of attention channels) - attention preservation
- proj_multiplier: Keep at 0.0 (do not prune projection layers)
- head_multiplier: Keep at 0.0 (do not prune attention heads)
- Round-to: Master Agent suggested {effective_round_to} for MAC calculation stability

MAC ALLOCATION VERIFICATION FOR {target_macs:.3f}G TARGET:
- Max safe MLP MAC: {target_macs * 0.15:.3f}G ({(target_macs * 0.15 / baseline_macs * 100):.1f}% of baseline)
- Max safe QKV MAC: {target_macs * 0.08:.3f}G ({(target_macs * 0.08 / baseline_macs * 100):.1f}% of baseline)
- Total core allocation: {target_macs * 0.23:.3f}G ({(target_macs * 0.23 / baseline_macs * 100):.1f}% of baseline)

CRITICAL: These MAC limits allow achieving {target_macs:.3f}G target while maintaining model viability.
"""
        guidance = f"""
ImageNet ViT MAC Guidance (UPDATED):
- Use Taylor importance (MANDATORY for MAC-aware accuracy preservation)
- Balance MLP and attention MAC allocation for optimal efficiency
- Start conservative with MAC allocation, then optimize based on MAC measurements
- Focus on MLP layers for primary MAC reduction, attention for fine-tuning
- Monitor zero-shot accuracy - if <10%, MAC allocation too aggressive
- Target MAC efficiency: {mac_efficiency_target:.1f}% of baseline operations
- Respect Master Agent's round_to suggestion for MAC measurement stability
"""
    else:
        safety_limits = f"""
VIT MAC SAFETY LIMITS FOR CIFAR-10:
===================================
- Target MAC budget: {target_macs:.3f}G (reduction: {mac_reduction_needed:.1f}%)
- MAC tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% ({tolerance_range})
- MLP MAC allocation: Maximum 50% of total target MAC budget (aggressive allowed)
- QKV MAC allocation: Maximum 30% of total target MAC budget (moderate-aggressive)
- More aggressive MAC allocation allowed for simpler dataset
- Round-to: Master Agent suggested {effective_round_to} for MAC stability

MAC ALLOCATION VERIFICATION FOR {target_macs:.3f}G TARGET:
- Max safe MLP MAC: {target_macs * 0.50:.3f}G ({(target_macs * 0.50 / baseline_macs * 100):.1f}% of baseline)
- Max safe QKV MAC: {target_macs * 0.30:.3f}G ({(target_macs * 0.30 / baseline_macs * 100):.1f}% of baseline)
"""
        guidance = f"""
CIFAR-10 ViT MAC Guidance:
- L1/L2 norm importance acceptable for MAC optimization efficiency
- Can use more aggressive MAC allocation than ImageNet
- Less critical to preserve pretrained MAC patterns
- Focus on achieving {target_macs:.3f}G MAC target with good accuracy recovery
- Use Master Agent's round_to suggestion for MAC measurement accuracy
"""

    param_defs = f"""
MAC-BASED PARAMETER DEFINITIONS:

1. Importance Criterion (for MAC efficiency): 
   - "taylor": REQUIRED for ImageNet ViTs (best MAC-aware accuracy preservation)
   - "l1norm"/"l2norm": OK for CIFAR-10 (MAC optimization efficiency acceptable)
   - MASTER AGENT SUGGESTED: Follow Master Agent's MAC-based recommendation

2. MAC Allocation Strategy (KEY PARAMETERS):
   - mlp_multiplier: Percentage of target MAC budget allocated to MLP layers
   - qkv_multiplier: Percentage of target MAC budget allocated to attention layers
   - proj_multiplier: Always 8% (preserve output projection MAC efficiency)
   - head_multiplier: Always 2% (preserve attention head MAC operations)
   - Total allocation should achieve {target_macs:.3f}G target

3. Round-To (for MAC measurement): Master Agent suggested {effective_round_to} for ViT MAC compatibility
   - 1: Maximum MAC flexibility, fine-grained control
   - 2: Standard for ViTs (attention head MAC compatibility)
   - 4: Hardware MAC efficiency balance

4. Multiplier Explanation:
   - Multipliers are DIRECT PRUNING RATIOS (NOT scaling factors after the fix)
   - mlp_multiplier: Direct pruning ratio for MLP layers (e.g., 0.4 means prune 40% of MLP channels)
   - qkv_multiplier: Direct pruning ratio for attention layers (e.g., 0.15 means prune 15% of attention channels)
   - Example: mlp_multiplier=0.40 → 40% MLP pruning (direct)
   - Example: qkv_multiplier=0.15 → 15% attention pruning (direct)
   - CRITICAL: Higher multipliers = MORE pruning = FEWER MACs; Lower multipliers = LESS pruning = MORE MACs
   - Valid range: 0.0 (no pruning) to 0.99 (maximum pruning)

MAC BUDGET CONTEXT:
- Baseline MACs: {baseline_macs:.3f}G
- Target MACs: {target_macs:.3f}G
- MAC efficiency target: {mac_efficiency_target:.1f}% of baseline
- Acceptable range: {tolerance_range}
"""

    output_format = f"""
Output your ViT MAC allocation strategy as JSON:
{{
  "parameter_tuning_order": ["importance_criterion", "target_macs", "round_to"],
  "importance_criterion": "taylor",
  "baseline_macs": {baseline_macs:.3f},
  "target_macs": {target_macs:.3f},
  "macs_overshoot_tolerance_pct": {macs_overshoot_tolerance_pct:.1f},
  "macs_undershoot_tolerance_pct": {macs_undershoot_tolerance_pct:.1f},
  "round_to": {effective_round_to},
  "global_pruning": true,
  "rationale": "MAC safety verified: MLP allocation ≤ limit, QKV allocation ≤ limit. Master Agent round_to: {effective_round_to}. Show MAC calculations for {target_macs:.3f}G target.",
  "architecture_type": "vit",
  "master_suggestions_used": true,
  "isomorphic_group_ratios": {{
    "mlp_multiplier": YOUR_CALCULATED_MLP_RATIO,
    "qkv_multiplier": YOUR_CALCULATED_QKV_RATIO,
    "proj_multiplier": 0.0,
    "head_multiplier": 0.0
  }},
  "expected_mac_breakdown": {{
    "mlp_mac_g": YOUR_MLP_PERCENTAGE * {target_macs:.3f} / 100,
    "qkv_mac_g": YOUR_QKV_PERCENTAGE * {target_macs:.3f} / 100,
    "proj_mac_g": {target_macs * 0.08:.3f},
    "head_mac_g": {target_macs * 0.02:.3f},
    "total_mac_g": {target_macs:.3f}
  }},
  "mac_safety_verification": {{
    "mlp_mac_within_limits": true,
    "qkv_mac_within_limits": true,
    "total_allocation_valid": true,
    "expected_achieved_macs": {target_macs:.3f},
    "mac_efficiency_target": {mac_efficiency_target:.1f}
  }}
}}

CRITICAL MAC-BASED REQUIREMENTS:
- Output ONLY the JSON object above  
- No comments with # or //
- No markdown code blocks
- Use Master Agent's round_to suggestion: {effective_round_to}
- Calculate actual MAC allocation percentages that sum to achieve {target_macs:.3f}G
- Ensuremlp_multiplier + qkv_multiplier + proj_multiplier + head_multiplier ≤ 100%
- Verify expected MAC breakdown sums to target: {target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%
- Focus on MAC budget achievement, not parameter reduction percentages
"""

    return {
        'architecture_specific_guidance': guidance,
        'safety_constraints': safety_limits,
        'parameter_definitions': param_defs,
        'architecture_output_format': output_format
    }
    
def format_vit_history_analysis(history, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, dataset):
    """Enhanced MAC-based history analysis specifically for ViT models"""
    
    if not history:
        return f"No previous attempts - use conservative MAC allocation baseline approach for {target_macs:.3f}G target."
    
    analysis = [f"LEARNING FROM {len(history)} PREVIOUS ViT MAC ATTEMPTS:"]
    analysis.append("=" * 60)
    analysis.append(f"MAC TARGET: {target_macs:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) from {baseline_macs:.3f}G baseline")
    analysis.append(f"MAC EFFICIENCY TARGET: {(target_macs / baseline_macs * 100):.1f}% of baseline operations")
    analysis.append("")
    
    # Find the best performing attempt (MAC-aware)
    best_attempt = None
    best_score = -1  # Combined score considering both MAC target achievement and accuracy
    
    for entry in history:
        if dataset.lower() == 'imagenet':
            accuracy = entry.get('fine_tuned_top1_accuracy', entry.get('zero_shot_top1_accuracy', 0))
        else:
            accuracy = entry.get('fine_tuned_accuracy', entry.get('zero_shot_accuracy', 0))
        
        # Get MAC metrics
        achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
        if achieved_macs is None:
            # Fallback: estimate from achieved ratio if MAC data not available
            achieved_ratio = entry.get('achieved_ratio', 0)
            achieved_macs = baseline_macs * (1.0 - achieved_ratio) if achieved_ratio else None
        
        if achieved_macs is not None:
            # Calculate MAC target achievement score
            mac_error_pct = abs((achieved_macs - target_macs) / target_macs * 100)
            mac_score = max(0, 100 - mac_error_pct)  # Higher score for closer to target
            
            # Combined score: MAC achievement (60%) + accuracy (40%)
            combined_score = 0.6 * mac_score + 0.4 * accuracy
            
            if combined_score > best_score:
                best_score = combined_score
                best_attempt = entry
    
    if best_attempt:
        best_strategy = best_attempt.get('strategy_used', {})
        best_achieved_macs = best_attempt.get('achieved_macs', best_attempt.get('final_macs_g'))
        best_achieved_ratio = best_attempt.get('achieved_ratio', 0)
        
        # Get MAC allocation from strategy
        isomorphic_group_ratios = best_strategy.get('isomorphic_group_ratios', {})
        mlp_multiplier = isomorphic_group_ratios.get('mlp_multiplier', 'N/A')
        qkv_multiplier = isomorphic_group_ratios.get('qkv_multiplier', 'N/A')
        
        # Fallback to legacy multipliers if MAC allocation not available
        if mlp_multiplier == 'N/A':
            mlp_multiplier = best_strategy.get('mlp_multiplier', 'N/A')
            qkv_multiplier = best_strategy.get('qkv_multiplier', 'N/A')
            legacy_info = f"(Legacy: MLP mult={mlp_multiplier}, QKV mult={qkv_multiplier})"
        else:
            legacy_info = ""
        
        mac_error = (best_achieved_macs - target_macs) if best_achieved_macs else 0
        mac_error_pct = (mac_error / target_macs * 100) if target_macs else 0

        overshoot = mac_error if mac_error > 0 else 0
        undershoot = abs(mac_error) if mac_error < 0 else 0
        within_tolerance = (overshoot <= target_macs * (macs_overshoot_tolerance_pct / 100)) and (undershoot <= target_macs * (macs_undershoot_tolerance_pct / 100))
        
        analysis.append(f"🏆 BEST MAC-AWARE RESULT SO FAR:")
        analysis.append(f"   - Accuracy: {accuracy:.1f}%")
        analysis.append(f"   - Achieved MACs: {best_achieved_macs:.3f}G (target: {target_macs:.3f}G)")
        analysis.append(f"   - MAC error: {mac_error:+.3f}G ({mac_error_pct:+.1f}%)")
        analysis.append(f"   - Within tolerance: {'✅ YES' if within_tolerance else '❌ NO'}")
        analysis.append(f"   - MLP MAC allocation: {mlp_multiplier}% {legacy_info}")
        analysis.append(f"   - QKV MAC allocation: {qkv_multiplier}%")
        analysis.append(f"   - Importance: {best_strategy.get('importance_criterion', 'N/A')}")
        analysis.append(f"   - MAC efficiency: {(best_achieved_macs / baseline_macs * 100):.1f}% of baseline")
    
    # Identify MAC-based failures
    failures = []
    mac_failures = []
    
    for entry in history:
        if dataset.lower() == 'imagenet':
            accuracy = entry.get('fine_tuned_top1_accuracy', entry.get('zero_shot_top1_accuracy', 0))
            accuracy_threshold = 10.0  # Catastrophic for ImageNet
        else:
            accuracy = entry.get('fine_tuned_accuracy', entry.get('zero_shot_accuracy', 0))
            accuracy_threshold = 30.0  # Catastrophic for CIFAR-10
        
        # Check for accuracy failures
        if accuracy < accuracy_threshold:
            failures.append(entry)
        
        # Check for MAC target failures
        achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
        if achieved_macs is not None:
            mac_error_pct = abs((achieved_macs - target_macs) / target_macs * 100)
            # Use the larger tolerance as the threshold for "significantly outside tolerance"
            max_tolerance = max(macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct)
            if mac_error_pct > max_tolerance * 3:  # Significantly outside tolerance
                mac_failures.append(entry)
    
    if failures:
        analysis.append("")
        analysis.append("❌ CATASTROPHIC ACCURACY FAILURES TO AVOID:")
        for i, failure in enumerate(failures):
            strategy = failure.get('strategy_used', {})
            isomorphic_group_ratios = strategy.get('isomorphic_group_ratios', {})
            
            if dataset.lower() == 'imagenet':
                acc = failure.get('fine_tuned_top1_accuracy', failure.get('zero_shot_top1_accuracy', 0))
            else:
                acc = failure.get('fine_tuned_accuracy', failure.get('zero_shot_accuracy', 0))
            
            mlp_info = isomorphic_group_ratios.get('mlp_multiplier', strategy.get('mlp_multiplier', 'N/A'))
            qkv_info = isomorphic_group_ratios.get('qkv_multiplier', strategy.get('qkv_multiplier', 'N/A'))
            
            analysis.append(f"   Failure {i+1}: MLP={mlp_info}, QKV={qkv_info} → {acc:.1f}% accuracy")
    
    if mac_failures:
        analysis.append("")
        analysis.append("📉 MAC TARGET FAILURES TO AVOID:")
        for i, failure in enumerate(mac_failures):
            strategy = failure.get('strategy_used', {})
            isomorphic_group_ratios = strategy.get('isomorphic_group_ratios', {})
            achieved_macs = failure.get('achieved_macs', failure.get('final_macs_g', 0))
            mac_error = achieved_macs - target_macs
            mac_error_pct = (mac_error / target_macs * 100) if target_macs else 0
            
            mlp_info = isomorphic_group_ratios.get('mlp_multiplier', strategy.get('mlp_multiplier', 'N/A'))
            qkv_info = isomorphic_group_ratios.get('qkv_multiplier', strategy.get('qkv_multiplier', 'N/A'))
            
            analysis.append(f"   MAC Failure {i+1}: MLP={mlp_info}, QKV={qkv_info} → "
                          f"{achieved_macs:.3f}G ({mac_error_pct:+.1f}% error)")
    
    # MAC-aware pattern analysis
    analysis.append("")
    analysis.append("📊 MAC ALLOCATION PATTERN ANALYSIS:")
    
    # Check if same MAC allocation was repeated
    configs = {}
    for i, entry in enumerate(history):
        strategy = entry.get('strategy_used', {})
        isomorphic_group_ratios = strategy.get('isomorphic_group_ratios', {})
        
        # Create config key from MAC allocation or legacy multipliers
        if isomorphic_group_ratios:
            mlp_val = isomorphic_group_ratios.get('mlp_multiplier', 0)
            qkv_val = isomorphic_group_ratios.get('qkv_multiplier', 0)
            config_key = f"mlp_{mlp_val:.1f}%_qkv_{qkv_val:.1f}%"
        else:
            mlp_val = strategy.get('mlp_multiplier', 0)
            qkv_val = strategy.get('qkv_multiplier', 0)
            config_key = f"mlp_mult_{mlp_val:.2f}_qkv_mult_{qkv_val:.2f}"
        
        if config_key in configs:
            configs[config_key].append(i)
        else:
            configs[config_key] = [i]
    
    repeated_configs = {k: v for k, v in configs.items() if len(v) > 1}
    if repeated_configs:
        analysis.append("   ⚠️ REPEATED MAC ALLOCATIONS DETECTED:")
        for config, attempts in repeated_configs.items():
            analysis.append(f"      Config {config} tried {len(attempts)} times - avoid repeating!")
    
    # MAC convergence analysis
    if len(history) >= 3:
        recent_attempts = history[-3:]
        mac_errors = []
        for attempt in recent_attempts:
            achieved_macs = attempt.get('achieved_macs', attempt.get('final_macs_g'))
            if achieved_macs is not None:
                error = abs(achieved_macs - target_macs)
                mac_errors.append(error)
        
        if len(mac_errors) >= 2:
            if mac_errors[-1] < mac_errors[0]:
                analysis.append("   📈 MAC CONVERGENCE: Getting closer to target - continue optimization")
            else:
                analysis.append("   📉 MAC DIVERGENCE: Moving away from target - try different approach")
    
    # Suggest MAC-focused next steps
    analysis.append("")
    if len(history) == 1:
        analysis.append("🎯 NEXT MAC OPTIMIZATION STEP:")
        analysis.append(f"   - Analyze baseline MAC results vs {target_macs:.3f}G target")
        analysis.append("   - If MAC overshoot, reduce MAC allocation percentages")
        analysis.append("   - If MAC undershoot, increase MAC allocation percentages")
        analysis.append("   - Maintain total allocation ≤ 100%")
    else:
        analysis.append("🎯 MAC ALLOCATION OPTIMIZATION RECOMMENDATIONS:")
        analysis.append("   - Build upon the best MAC-performing configuration")
        analysis.append("   - Avoid MAC allocation patterns that led to target failures")
        analysis.append(f"   - Focus on achieving {target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance")
        analysis.append("   - Make incremental MAC allocation changes, not dramatic jumps")
        if repeated_configs:
            analysis.append("   - CRITICAL: Use different MAC allocations than previous attempts!")
        analysis.append(f"   - Target MAC efficiency: {(target_macs / baseline_macs * 100):.1f}% of baseline")
    
    return "\n".join(analysis)


def format_analysis_prompt(state):
    """Format MAC-based analysis prompt with architecture detection and revision awareness"""
    
    dataset = state.get('dataset', 'cifar10')
    num_classes = state.get('num_classes', 10)
    input_size = state.get('input_size', 224)
    
    # MAC-based parameters (primary)
    baseline_macs = state.get('baseline_macs', 10.0)
    target_macs = state.get('target_macs', 5.0)
    macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
    macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)
    
    # Legacy parameter (for backward compatibility)
    target_ratio = state.get('target_pruning_ratio', 1.0 - (target_macs / baseline_macs))
    
    # Calculate MAC efficiency metrics
    mac_efficiency_target = (target_macs / baseline_macs) * 100
    mac_reduction_needed = ((baseline_macs - target_macs) / baseline_macs) * 100
    
    # Extract Master Agent MAC suggestions
    master_results = state.get('master_results', {})
    recommended_strategy = master_results.get('recommended_strategy', {})
    master_suggested_macs = recommended_strategy.get('target_macs')
    master_suggested_round_to = recommended_strategy.get('round_to')
    master_suggested_importance = recommended_strategy.get('importance_criterion')
    master_suggested_isomorphic_group_ratios = recommended_strategy.get('isomorphic_group_ratios', {})
    
    # Format Master Agent MAC suggestions for the prompt
    master_suggestions = []
    if master_suggested_macs is not None:
        master_suggestions.append(f"- Target MACs: {master_suggested_macs:.3f}G")
    if master_suggested_round_to is not None:
        master_suggestions.append(f"- Round-To: {master_suggested_round_to} (for MAC measurement stability)")
    if master_suggested_importance is not None:
        master_suggestions.append(f"- Importance Criterion: {master_suggested_importance} (for MAC efficiency)")
    if master_suggested_isomorphic_group_ratios:
        master_suggestions.append("- MAC Allocation Suggestions:")
        for key, value in master_suggested_isomorphic_group_ratios.items():
            master_suggestions.append(f"  * {key}: {value}")
    
    master_suggestions_text = "\n".join(master_suggestions) if master_suggestions else "No specific MAC suggestions provided"
    
    # Handle missing model_name gracefully
    model_name = state.get('model_name')
    if not model_name:
        query = state.get('query', '')
        import re
        match = re.search(r'Prune\s+(\w+(?:_\w+)*)\s+model', query)
        if match:
            model_name = match.group(1)
            print(f"[🔧] Extracted model name from query: {model_name}")
        else:
            model_name = 'unknown_model'
            print(f"[⚠️] Using fallback model name: {model_name}")
    
    current_revision = state.get('revision_number', 0)
    
    # Detect architecture type
    cnn_types = ['resnet', 'efficientnet', 'mobilenet', 'densenet', 'resnext', 'convnext']
    vit_types = ['vit', 'deit', 'swin', 'beit']
    
    if any(cnn_type in model_name.lower() for cnn_type in cnn_types):
        architecture_type = "cnn"
        arch_content = get_cnn_analysis_content(
            dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, model_name
        )
    elif any(vit_type in model_name.lower() for vit_type in vit_types):
        architecture_type = "vit"  
        arch_content = get_vit_analysis_content(
            dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, model_name, 
            master_suggested_round_to
        )
    else:
        architecture_type = "vit"
        arch_content = get_vit_analysis_content(
            dataset, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, model_name, 
            master_suggested_round_to
        )
        print(f"[⚠️] Unknown architecture {model_name}, defaulting to ViT approach")
    
    # Add MAC-aware revision context
    if current_revision == 0:
        revision_context = f"""
🚀 REVISION 0 - MAC BASELINE ATTEMPT:
This is the first MAC budget attempt. Target: {target_macs:.3f}G from {baseline_macs:.3f}G baseline.
Use Master Agent MAC suggestions if provided, otherwise use safe baseline MAC allocation.
Priority: Establish a working MAC baseline with reasonable accuracy.
Approach: Conservative MAC allocation but not overly restrictive.
MAC Focus: Achieve {target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% with maximum accuracy.
"""
    elif current_revision == 1:
        revision_context = f"""
🧠 REVISION 1 - LEARNING FROM MAC BASELINE:
Analyze the MAC baseline results and Master Agent suggestions for targeted MAC improvements.
You have one MAC data point to learn from - use it to guide MAC optimization.
Priority: Improve upon baseline MAC performance while following Master Agent guidance.
Approach: Make informed MAC allocation adjustments based on baseline results.
MAC Focus: Optimize toward {target_macs:.3f}G target based on baseline MAC measurements.
"""
    else:
        revision_context = f"""
🎯 REVISION {current_revision} - MAC OPTIMIZATION PHASE:
You have {len(state.get('history', []))} previous MAC attempts to learn from plus Master Agent guidance.
Look for MAC patterns, avoid failed MAC configurations, and optimize based on MAC trends.
Priority: Find the optimal MAC allocation within safety limits using all available MAC data.
Approach: Data-driven MAC optimization with pattern recognition plus Master Agent intelligence.
MAC Focus: Achieve {target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% with maximum accuracy.
"""
    
    # Enhanced MAC-aware history analysis
    history = state.get('history', [])
    if architecture_type == "vit" and history:
        previous_strategies = format_vit_history_analysis(
            history, target_macs, baseline_macs, macs_overshoot_tolerance_pct, macs_undershoot_tolerance_pct, dataset
        )
    elif history:
        strategies = []
        for i, entry in enumerate(history):
            strategy = entry.get('strategy_used', {})
            
            # Get MAC metrics (preferred) or fallback to ratios
            achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g'))
            achieved_ratio = entry.get('achieved_ratio', 0)
            
            if achieved_macs is not None:
                mac_error = achieved_macs - target_macs
                mac_error_pct = (mac_error / target_macs * 100) if target_macs else 0
                mac_info = f"{achieved_macs:.3f}G ({mac_error_pct:+.1f}% error)"
            else:
                # Fallback to ratio information
                estimated_macs = baseline_macs * (1.0 - achieved_ratio) if achieved_ratio else 0
                mac_info = f"~{estimated_macs:.3f}G (estimated), {achieved_ratio*100:.1f}% reduction"
            
            if dataset.lower() == 'imagenet':
                accuracy = entry.get('fine_tuned_top1_accuracy', entry.get('zero_shot_top1_accuracy', 0))
                acc_type = "Top-1"
            else:
                accuracy = entry.get('fine_tuned_accuracy', entry.get('zero_shot_accuracy', 0))
                acc_type = "Acc"
            
            # Include MAC allocation info if available
            isomorphic_group_ratios = strategy.get('isomorphic_group_ratios', {})
            if isomorphic_group_ratios:
                alloc_info = f", MLP:{isomorphic_group_ratios.get('mlp_multiplier', 'N/A')}% QKV:{isomorphic_group_ratios.get('qkv_multiplier', 'N/A')}%"
            else:
                # Legacy multiplier info
                mlp_mult = strategy.get('mlp_multiplier', 'N/A')
                qkv_mult = strategy.get('qkv_multiplier', 'N/A')
                channel_ratio = strategy.get('channel_pruning_ratio', 'N/A')
                if mlp_mult != 'N/A':
                    alloc_info = f", MLP mult:{mlp_mult} QKV mult:{qkv_mult}"
                elif channel_ratio != 'N/A':
                    alloc_info = f", Channel ratio:{channel_ratio}"
                else:
                    alloc_info = ""
            
            strategies.append(f"Attempt {i+1}: {mac_info}, {accuracy:.1f}% {acc_type}{alloc_info}")
        
        previous_strategies = "\n".join(strategies)
    else:
        previous_strategies = f"No previous MAC attempts - this is the first iteration targeting {target_macs:.3f}G."
    
    # Create the complete MAC-based prompt
    prompt_text = ANALYSIS_PROMPT.format(
        model_name=model_name,
        architecture_type=architecture_type.upper(),
        dataset=dataset,
        num_classes=num_classes,
        input_size=input_size,
        target_macs=target_macs,
        baseline_macs=baseline_macs,
        macs_overshoot_tolerance_pct=macs_overshoot_tolerance_pct,
        macs_undershoot_tolerance_pct=macs_undershoot_tolerance_pct,
        revision_context=revision_context,
        architecture_specific_guidance=arch_content['architecture_specific_guidance'],
        master_directives=state.get('master_results', {}).get('directives', ''),
        master_suggestions=master_suggestions_text,
        profile_info=state.get('profile_results', {}).get('analysis', ''),
        requirements=state.get('query', ''),
        target_ratio=target_ratio,  # Keep for legacy compatibility
        safety_constraints=arch_content['safety_constraints'],
        parameter_definitions=arch_content['parameter_definitions'],
        previous_strategies=previous_strategies,
        architecture_output_format=arch_content['architecture_output_format']
    )
    
    return prompt_text


PRUNING_PROMPT = """You are implementing MAC-budget model pruning based on the Analysis Agent's recommendations.

Please be concise. Limit your response to 300 words or fewer.

DATASET CONTEXT:
- Dataset: {dataset} ({num_classes} classes)
- Input size: {input_size}x{input_size}
- Model complexity: {model_complexity}

MAC BUDGET CONTEXT:
- Baseline MACs: {baseline_macs:.3f}G
- Target MACs: {target_macs:.3f}G (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance)
- MAC reduction needed: {mac_reduction_needed:.1f}%
- MAC efficiency target: {mac_efficiency_target:.1f}% of baseline
- Acceptable MAC range: {target_macs * (1 - macs_undershoot_tolerance_pct / 100):.3f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.3f}G

Here is the specific MAC-aware isomorphic pruning code to use, which should be adapted based on the analysis results:

def mac_aware_isomorphic_pruning(model):
    # Import required packages
    import torch
    import torch.nn as nn
    import torch_pruning as tp
    import timm
    import pbench
    from pbench import mac_measurement
    pbench.forward_patch.patch_timm_forward()

    # Setup ignored layers and heads for MAC efficiency
    ignored_layers = []
    num_heads = {{}}
    for name, m in model.named_modules():
        # Dataset-aware layer preservation for MAC efficiency
        if (('fc' in name) or ('classifier' in name) or ('head' in name)) and isinstance(m, nn.Linear):
            # For ImageNet: preserve 1000-class classifier MAC; For CIFAR-10: preserve 10-class classifier MAC
            if m.out_features == {num_classes}:
                ignored_layers.append(m)

        if isinstance(m, timm.models.swin_transformer.WindowAttention):
            num_heads[m.qkv] = m.num_heads
            ignored_layers.append(m.relative_position_bias_table)

        if isinstance(m, timm.models.vision_transformer.Attention):
            num_heads[m.qkv] = m.num_heads

    # Initialize importance based on Analysis Agent's MAC-aware recommendation
    if importance_criterion == "taylor":
        imp = tp.importance.TaylorImportance()
    elif importance_criterion == "l1norm":
        imp = tp.importance.MagnitudeImportance(p=1)
    elif importance_criterion == "l2norm":
        imp = tp.importance.MagnitudeImportance(p=2)
    else:
        imp = tp.importance.TaylorImportance()  # Default for MAC efficiency

    # Initialize MAC-aware pruner with dataset-appropriate input size
    example_inputs = (torch.randn(1, 3, {input_size}, {input_size}),)
    
    # MAC-focused pruning strategy
    if architecture_type == "cnn":
        # CNN: Use channel pruning ratio to achieve MAC target
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs=example_inputs,
            global_pruning=True,
            importance=imp,
            isomorphic=True,
            pruning_ratio=channel_pruning_ratio,  # From Analysis Agent MAC calculation
            ignored_layers=ignored_layers,
            customized_pruners=pbench.extension.EXTENDED_PRUNERS,
            round_to=round_to_value,  # Use MAC-stable value from Analysis Agent
        )
    else:
        # ViT: Use MAC allocation strategy
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs=example_inputs,
            global_pruning=True,
            importance=imp,
            isomorphic=True,
            pruning_ratio=target_macs / baseline_macs,  # MAC efficiency ratio
            ignored_layers=ignored_layers,
            num_heads=num_heads,
            prune_head_dims=True,
            prune_num_heads=True,
            customized_pruners=pbench.extension.EXTENDED_PRUNERS,
            round_to=round_to_value,  # MAC measurement stability
            # MAC allocation constraints
            group_lasso_params={{
                'mlp_target_macs': mlp_target_macs,
                'qkv_target_macs': qkv_target_macs,
                'proj_target_macs': proj_target_macs,
                'head_target_macs': head_target_macs
            }}
        )

    # Execute MAC-aware pruning
    for i, g in enumerate(pruner.step(interactive=True)):
        g.prune()
    
    # Measure achieved MACs
    achieved_macs = mac_measurement.measure_model_macs(model, example_inputs)
    mac_error = achieved_macs - target_macs  # Convert to operations
    mac_error_pct = (mac_error / (target_macs)) * 100
    
    print(f"MAC Results: Achieved {{achieved_macs/1e9:.3f}}G, Target {target_macs/1e9:.3f}G, Error {{mac_error_pct:+.1f}}%")
    
    return model, achieved_macs

DATASET-SPECIFIC MAC STRATEGY:
{dataset_strategy}

THINK STEP BY STEP about how to achieve EXACTLY {target_macs:.3f}G MAC operations:
1. First, recognize that MAC targets are more predictable than parameter reduction targets
   because MAC operations scale more linearly with structural changes.
2. For CNNs: The channel_pruning_ratio should be calculated to achieve {target_macs:.3f}G MAC budget
   (Analysis Agent provided: {channel_pruning_ratio:.3f})
3. For ViTs: Use MAC allocation strategy with specific MAC budgets for each layer type:
   - MLP MAC target: {mlp_target_macs:.3f}G
   - QKV MAC target: {qkv_target_macs:.3f}G  
   - Projection MAC target: {proj_target_macs:.3f}G
   - Head MAC target: {head_target_macs:.3f}G
4. global_pruning must always be True for MAC-aware isomorphic pruning
5. Use the round_to value recommended by the Analysis Agent ({round_to_value}) for MAC measurement stability
   {round_to_explanation}
6. The ignored_layers list should preserve MAC-critical components:
   {ignored_layers_guidance}

MAC-AWARE IMPLEMENTATION STEPS:
1. Start with Analysis Agent's calculated MAC allocation strategy
2. Execute pruning and measure actual MAC operations
3. If achieved MACs are outside tolerance ({target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%):
   - For CNNs: Adjust channel_pruning_ratio in small increments (±0.02)
   - For ViTs: Adjust MAC allocation percentages (±2%)
4. The goal is to achieve {target_macs:.3f}G MAC operations within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance

IMPORTANT: MAC-based pruning is more predictable than parameter-based pruning
because MAC operations have clearer relationships to structural changes. {conservatism_note}

Analysis Results: {analysis_str}
Recommended Importance Criterion: {importance_criterion} (for MAC efficiency)
Recommended Round-To Value: {round_to_value} (for MAC measurement stability)
Architecture Type: {architecture_type}
MAC Allocation Strategy: {isomorphic_group_ratios_strategy}
"""

FINE_TUNING_PROMPT = """You are fine-tuning a MAC-budget pruned neural network model to recover accuracy.
The pruned model has achieved the target MAC budget but needs optimization to maintain performance.

Please be concise. Limit your response to 300 words or fewer.

DATASET CONTEXT:
- Dataset: {dataset} ({num_classes} classes)
- Input resolution: {input_size}x{input_size}
- Evaluation metrics: {evaluation_metrics}
- Target accuracy threshold: {accuracy_threshold}%

MODEL DETAILS:
- Model architecture: {model_name}
- Baseline MACs: {baseline_macs:.3f}G
- Achieved MACs: {achieved_macs:.3f}G (target was {target_macs:.3f}G)
- MAC efficiency: {mac_efficiency:.1f}% of baseline operations
- MAC error: {mac_error_pct:+.1f}% from target ({mac_error:+.3f}G)
- Pruning method used: MAC-aware isomorphic pruning with {importance_criterion} importance criterion
- Zero-shot accuracy (not used for decision-making): {zero_shot_accuracy:.2f}%

FINE-TUNING TRIGGER:
This model was selected for fine-tuning because its achieved MAC budget ({achieved_macs:.3f}G) was within the acceptable tolerance (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) of the target ({target_macs:.3f}G), regardless of zero-shot accuracy.
MAC-AWARE FINE-TUNING OBJECTIVES:
1. Maximize accuracy recovery while preserving the MAC efficiency achieved by pruning
2. Optimize the model to work efficiently at the target MAC budget ({target_macs:.3f}G)
3. Balance between training time and accuracy improvement at reduced MAC operations
4. Prevent overfitting while maintaining MAC efficiency
5. Achieve the highest possible {evaluation_metrics} recovery after MAC-budget pruning for {dataset}

DATASET-SPECIFIC RECOMMENDATIONS:
{dataset_recommendations}

MAC-AWARE FINE-TUNING PROCESS:
1. Initialize optimizer with appropriate learning rate ({recommended_lr}) for MAC-efficient model
2. Set up learning rate scheduler (cosine annealing recommended for MAC-pruned models)
3. Train for the recommended number of epochs ({recommended_epochs}) at {mac_efficiency:.1f}% MAC efficiency
4. Monitor {evaluation_metrics} and MAC efficiency to prevent overfitting
5. Save the best checkpoint based on validation {primary_metric} while maintaining MAC budget
6. Validate MAC operations remain within {target_macs:.3f}G +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% throughout training

TRAINING PARAMETERS (MAC-OPTIMIZED):
- Batch size: {recommended_batch_size} (adjusted for {dataset} and MAC efficiency)
- Learning rate: {recommended_lr} (tuned for MAC-pruned model)
- Weight decay: {recommended_weight_decay}
- Epochs: {recommended_epochs}
- Optimizer: {recommended_optimizer}
- MAC validation frequency: Every 5 epochs

DATA AUGMENTATION (MAC-AWARE):
{data_augmentation_strategy}

MAC EFFICIENCY MONITORING:
- Track MAC operations during training to ensure consistency
- Verify no structural changes affect MAC budget
- Monitor MAC efficiency vs accuracy trade-off

Please report the MAC-aware fine-tuning progress including:
- Training {evaluation_metrics} curve (Learning curve) with MAC efficiency overlay
- Final training and validation {evaluation_metrics} at target MAC budget
- MAC operations consistency throughout training ({target_macs:.3f}G target)
- Any signs of overfitting or underfitting at reduced MAC operations
- Comparison to pre-fine-tuning accuracy and MAC efficiency
- {dataset}-specific performance analysis at {mac_efficiency:.1f}% MAC efficiency
- Final MAC validation: Achieved {achieved_macs:.3f}G vs Target {target_macs:.3f}G
"""

EVALUATION_PROMPT = """You are evaluating a MAC-budget pruned and fine-tuned neural network model.
Your task is to conduct a comprehensive evaluation and report the MAC efficiency results.

Please be concise. Limit your response to 300 words or fewer.

DATASET CONTEXT:
- Dataset: {dataset} ({num_classes} classes)
- Input resolution: {input_size}x{input_size}
- Evaluation metrics: {evaluation_metrics}
- Success threshold: {accuracy_threshold}%

MODEL DETAILS:
- Model architecture: {model_name}
- Baseline MACs: {baseline_macs:.3f}G
- Achieved MACs: {achieved_macs:.3f}G (target was {target_macs:.3f}G)
- MAC efficiency: {mac_efficiency:.1f}% of baseline operations
- MAC reduction: {mac_reduction:.1f}% ({mac_reduction_g:.3f}G reduction)
- MAC error: {mac_error_pct:+.1f}% from target ({mac_error:+.3f}G)
- MAC tolerance met: {'✅ YES' if mac_tolerance_met else '❌ NO'} (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%)
- Fine-tuning method: {fine_tuning_method}

MAC-BASED EVALUATION PHILOSOPHY:
- Zero-shot accuracy is not used to determine model viability.
- The model was selected for evaluation because it achieved the target MAC budget within tolerance and completed fine-tuning.
- Evaluation should focus on fine-tuned performance and MAC efficiency trade-offs.
- Success is measured by both accuracy recovery and MAC budget compliance.

EVALUATION METRICS TO REPORT:
{metrics_to_report}

MAC EFFICIENCY METRICS:
- MAC operations (before and after pruning): {baseline_macs:.3f}G → {achieved_macs:.3f}G
- MAC reduction percentage: {mac_reduction:.1f}%
- MAC efficiency target achievement: {mac_efficiency:.1f}% (target: {(target_macs/baseline_macs*100):.1f}%)
- MAC operations per inference: {achieved_macs:.3f}G
- MAC budget compliance: Within +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}% tolerance
- Parameter count (before and after pruning)
- Model size reduction
- Inference time comparison at target MAC budget
- Memory usage reduction with MAC efficiency

DATASET-SPECIFIC MAC ANALYSIS:
{dataset_analysis_requirements}

MAC-BASED SUCCESS CRITERIA FOR {dataset}:
{success_criteria}

MAC-BASED FAILURE SCENARIOS FOR {dataset}:
{failure_scenarios}

MAC EFFICIENCY ANALYSIS:
- MAC target achievement: {achieved_macs:.3f}G vs {target_macs:.3f}G target
- MAC efficiency vs accuracy trade-off analysis
- MAC operations distribution across layer types (if available)
- MAC budget sustainability across different input sizes
- Hardware efficiency implications of MAC reduction

RECOMMENDATIONS:
Based on your MAC-aware evaluation, provide:
1. Whether to accept this MAC-efficient model or continue MAC optimization exploration
2. Specific MAC allocation areas for improvement if continuing
3. Potential applications for this MAC-optimized model based on its efficiency/accuracy profile
4. {dataset}-specific deployment considerations for {mac_efficiency:.1f}% MAC efficiency
5. Comparison to typical {dataset} MAC reduction benchmarks
6. MAC efficiency ranking: {'Excellent' if mac_efficiency <= 30 else 'Good' if mac_efficiency <= 50 else 'Moderate' if mac_efficiency <= 70 else 'Conservative'} efficiency ({mac_efficiency:.1f}% of baseline MAC operations)
7. MAC budget suitability for edge deployment, mobile devices, or data center optimization
"""


# Add these three methods at the END of your prompts.py file, after all the existing prompt templates

def _format_analysis_for_llm(learning_analysis: dict, target_ratio: float, dataset: str) -> str:
    """Format learning analysis results for LLM consumption (legacy compatibility)"""
    
    # For backward compatibility with ratio-based calls
    # Convert target_ratio to approximate target_macs for formatting
    target_macs = learning_analysis.get('target_macs')
    if target_macs is None:
        # Rough estimation: assume baseline of 10G MACs (adjust as needed)
        baseline_estimate = 10.0
        target_macs = baseline_estimate * (1.0 - target_ratio)
    
    return _format_mac_analysis_for_llm(learning_analysis, target_ratio, target_macs, dataset)


def _format_mac_analysis_for_llm(learning_analysis: dict, target_ratio: float, target_macs: float, macs_overshoot_tolerance_pct: float, macs_undershoot_tolerance_pct: float, dataset: str) -> str:
    """Format the MAC-based learning analysis in a clear way for the LLM"""
    
    if not learning_analysis.get('has_sufficient_data', False):
        recommendations = learning_analysis.get('recommendations', {})
        return f"""
📊 BASELINE MAC ALLOCATION GUIDANCE (No Historical Data):
- Suggested MLP MAC allocation: {recommendations.get('suggested_mlp_multiplier', 0):.1f}%
- Suggested QKV MAC allocation: {recommendations.get('suggested_qkv_multiplier', 0):.1f}%
- Expected MLP MAC: {(recommendations.get('suggested_mlp_multiplier', 0) * target_macs / 100):.2f}G
- Expected QKV MAC: {(recommendations.get('suggested_qkv_multiplier', 0) * target_macs / 100):.2f}G
- Reasoning: {', '.join(recommendations.get('reasoning', ['No reasoning available']))}
"""
    
    pattern = learning_analysis.get('pattern_analysis', {})
    failure = learning_analysis.get('failure_analysis', {})
    recommendations = learning_analysis.get('recommendations', {})
    
    # Calculate MAC efficiency metrics
    avg_efficiency = pattern.get('avg_mac_efficiency', 1.0) * 100
    target_efficiency = (target_macs / pattern.get('baseline_mac_g', target_macs)) * 100 if pattern.get('baseline_mac_g') else 100
    
    text = f"""
📊 COMPREHENSIVE MAC-BASED LEARNING ANALYSIS FROM {learning_analysis.get('total_attempts', 0)} ATTEMPTS:

🔍 MAC PATTERN ANALYSIS:
- Average achieved MAC: {pattern.get('avg_achieved_mac_g', 0):.2f}G (target: {target_macs:.2f}G)
- Average MLP MAC allocation: {pattern.get('avg_mlp_multiplier', 0):.1f}%
- Average QKV MAC allocation: {pattern.get('avg_qkv_multiplier', 0):.1f}%
- Average MAC deviation: {pattern.get('avg_mac_deviation', 0):.3f}G ({pattern.get('avg_mac_deviation_pct', 0):.1f}%)
- Average MAC efficiency: {avg_efficiency:.1f}% of baseline
- Systematic MAC undershoot: {pattern.get('systematic_undershoot', False)}
- Systematic MAC overshoot: {pattern.get('systematic_overshoot', False)}

🎯 HIGH-ACCURACY NEAR-MISS DETECTION:
- Has excellent near-miss: {pattern.get('has_excellent_near_miss', False)}
- High-accuracy undershoots found: {len(pattern.get('high_accuracy_undershoots', []))}
"""

    # Add near-miss details if found
    if pattern.get('has_excellent_near_miss', False):
        high_acc_attempts = pattern.get('high_accuracy_undershoots', [])
        if high_acc_attempts:
            best_near_miss = max(high_acc_attempts, key=lambda x: x.get('accuracy', 0))
            text += f"""
🌟 BEST NEAR-MISS FOUND:
- Accuracy: {best_near_miss.get('accuracy', 0):.2f}%
- Achieved MAC: {best_near_miss.get('achieved_mac_g', 0):.2f}G
- MAC deviation: {best_near_miss.get('mac_deviation', 0):.3f}G
- MLP allocation: {best_near_miss.get('mlp_multiplier', 0):.1f}%
- QKV allocation: {best_near_miss.get('qkv_multiplier', 0):.1f}%
- Strategy: importance={best_near_miss.get('importance', 'unknown')}, round_to={best_near_miss.get('round_to', 'unknown')}
"""

    text += f"""
❌ MAC FAILURE ANALYSIS:
- Total failures: {failure.get('total_failures', 0)}
- MAC undershoot failures: {len(failure.get('mac_undershoot_failures', []))}
- MAC overshoot failures: {len(failure.get('mac_overshoot_failures', []))}
- Average undershoot severity: {failure.get('avg_mac_undershoot_severity', 0):.3f}G
- Average overshoot severity: {failure.get('avg_mac_overshoot_severity', 0):.3f}G

💡 INTELLIGENT MAC RECOMMENDATIONS:
- Suggested MLP MAC allocation: {recommendations.get('suggested_mlp_multiplier', 0):.1f}%
- Suggested QKV MAC allocation: {recommendations.get('suggested_qkv_multiplier', 0):.1f}%
- Expected MLP MAC: {(recommendations.get('suggested_mlp_multiplier', 0) * target_macs / 100):.2f}G
- Expected QKV MAC: {(recommendations.get('suggested_qkv_multiplier', 0) * target_macs / 100):.2f}G
- Total allocated: {(recommendations.get('suggested_mlp_multiplier', 0) + recommendations.get('suggested_qkv_multiplier', 0)):.1f}%
- Confidence level: {recommendations.get('confidence', 'unknown')}
- Reasoning: {', '.join(recommendations.get('reasoning', ['No reasoning available']))}

🧮 MAC-BASED MATHEMATICAL CALCULATION:
{_format_math_calculation(recommendations.get('mathematical_calculation', {}))}

📈 MAC TREND ANALYSIS:
"""

    # Add trend analysis if available
    trend = learning_analysis.get('trend_analysis', {})
    if not trend.get('insufficient_data', True):
        text += f"""
- MLP MAC allocation trend: {trend.get('mlp_mac_trend', 'unknown')}
- QKV MAC allocation trend: {trend.get('qkv_mac_trend', 'unknown')}
- Achieved MAC trend: {trend.get('achieved_mac_trend', 'unknown')}
- MAC deviation trend: {trend.get('mac_deviation_trend', 'unknown')}
- Is converging to target: {trend.get('is_converging', False)}
- Target approach rate: {trend.get('target_approach_rate', 0):.1%}
"""
    else:
        text += "- Insufficient data for trend analysis"

    text += f"""

🎯 KEY MAC-BASED LEARNING INSIGHTS:
{chr(10).join('- ' + insight for insight in recommendations.get('learning_insights', ['No learning insights available']))}

⚠️ CRITICAL MAC-BASED INSTRUCTIONS:
The analysis above shows clear MAC allocation patterns in your historical attempts. 
- Target: {target_macs:.2f}G MAC operations
- Tolerance: +{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%
- Tolerance range: {target_macs * (1 - macs_undershoot_tolerance_pct / 100):.2f}G - {target_macs * (1 + macs_overshoot_tolerance_pct / 100):.2f}G
- Focus on MAC efficiency while maintaining accuracy
- Use this data to make intelligent MAC allocation decisions rather than guessing
- Pay special attention to high-accuracy near-misses for fine-tuning
"""
    
    return text


def _format_math_calculation(math_calc: dict, macs_overshoot_tolerance_pct: float = None, macs_undershoot_tolerance_pct: float = None) -> str:
    """Format MAC-based mathematical calculation details"""
    if not math_calc:
        return "No calculation details available"
    
    if math_calc.get('strategy_type') == 'high_accuracy_tiny_mac_adjustment':
        return f"""
Strategy: High-Accuracy Tiny MAC Adjustment
- Best accuracy found: {math_calc.get('best_accuracy_found', 0):.2f}%
- Best achieved MAC: {math_calc.get('best_achieved_mac_g', 0):.2f}G
- Tiny MAC deviation: {math_calc.get('tiny_mac_deviation', 0):.3f}G
- MAC correction factor: {math_calc.get('mac_correction_factor_applied', 1.0):.2f}
- Original MLP allocation: {math_calc.get('original_mlp_percent', 0):.1f}%
- Original QKV allocation: {math_calc.get('original_qkv_percent', 0):.1f}%
- Corrected MLP allocation: {math_calc.get('corrected_mlp_percent', 0):.1f}%
- Corrected QKV allocation: {math_calc.get('corrected_qkv_percent', 0):.1f}%
"""
    
    elif math_calc.get('baseline_approach'):
        return f"""
Strategy: Baseline MAC Allocation
- Target MAC: {math_calc.get('target_mac_g', 0):.2f}G
- MLP allocation: {math_calc.get('suggested_mlp_multiplier', 0):.1f}%
- QKV allocation: {math_calc.get('suggested_qkv_multiplier', 0):.1f}%
- Expected MLP MAC: {math_calc.get('expected_mlp_mac_g', 0):.2f}G
- Expected QKV MAC: {math_calc.get('expected_qkv_mac_g', 0):.2f}G
"""
    
    else:
        return f"""
Strategy: Systematic MAC Correction
- Average MLP before: {math_calc.get('avg_mlp_percent_before', 0):.1f}%
- Average QKV before: {math_calc.get('avg_qkv_percent_before', 0):.1f}%
- MAC correction factor: {math_calc.get('mac_correction_factor', 1.0):.2f}
- MLP after correction: {math_calc.get('base_mlp_percent_after_correction', 0):.1f}%
- QKV after correction: {math_calc.get('base_qkv_percent_after_correction', 0):.1f}%
- Safety-capped MLP: {math_calc.get('safety_capped_mlp_percent', 0):.1f}%
- Safety-capped QKV: {math_calc.get('safety_capped_qkv_percent', 0):.1f}%
"""