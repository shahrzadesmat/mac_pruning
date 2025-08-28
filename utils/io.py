from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
import os


async def save_final_best_model(state):
    """
    Select the best fine‑tuned candidate and save BOTH formats:
    1. State dict (weights only) - smaller, more portable
    2. Full model (architecture + weights) - complete but environment-dependent
    """

    output_dir = state.get('output_dir', './models')
    dataset = state.get('dataset', 'cifar10')
    model_name = state.get('model_name', 'unknown')

    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Find all fine‑tuned candidates within MAC tolerance
    # ──────────────────────────────────────────────────────────────────────────
    history = state.get('history', [])
    target_macs = state.get('target_macs', 5.0)
    baseline_macs = state.get('baseline_macs', 10.0)
    macs_overshoot_tolerance_pct = state.get('macs_overshoot_tolerance_pct', 1.0)
    macs_undershoot_tolerance_pct = state.get('macs_undershoot_tolerance_pct', 5.0)

    candidate_models = []
    for entry in history:
        achieved_macs = entry.get('achieved_macs', entry.get('final_macs_g', 0.0))
        target_mac = entry.get('target_macs', target_macs)
        
        if achieved_macs and target_mac:
            mac_error_pct = ((achieved_macs - target_mac) / target_mac) * 100
            within_tol = (-macs_undershoot_tolerance_pct <= mac_error_pct <= macs_overshoot_tolerance_pct)
            if not within_tol:
                continue

            ft_acc = (
                entry.get('fine_tuned_top1_accuracy')
                if dataset.lower() == 'imagenet'
                else entry.get('fine_tuned_accuracy')
            )
            if ft_acc is not None:
                candidate_models.append(entry)

    if not candidate_models:
        print(f"[⚠️] No models within MAC tolerance (+{macs_overshoot_tolerance_pct:.1f}%/-{macs_undershoot_tolerance_pct:.1f}%) with fine‑tuned results found")
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Pick the candidate with highest fine‑tuned accuracy
    # ──────────────────────────────────────────────────────────────────────────
    if dataset.lower() == 'imagenet':
        best_candidate = max(candidate_models,
                             key=lambda x: x.get('fine_tuned_top1_accuracy', 0))
        best_ft_acc = best_candidate.get('fine_tuned_top1_accuracy', 0)
        best_zs_acc = best_candidate.get('zero_shot_top1_accuracy', 0)
        acc_label = "Top‑1"
    else:
        best_candidate = max(candidate_models,
                             key=lambda x: x.get('fine_tuned_accuracy', 0))
        best_ft_acc = best_candidate.get('fine_tuned_accuracy', 0)
        best_zs_acc = best_candidate.get('zero_shot_accuracy', 0)
        acc_label = "accuracy"

    achieved_macs = best_candidate.get('achieved_macs', 0)
    revision = best_candidate.get('revision', 0)
    mac_efficiency = (achieved_macs / baseline_macs) * 100 if baseline_macs > 0 else 0

    print(
        f"\n[🏆] Selected revision {revision} "
        f"({achieved_macs:.3f}G MAC achieved, {mac_efficiency:.1f}% efficiency) "
        f"with fine‑tuned {acc_label}: {best_ft_acc:.2f}%"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Locate the actual pruned model object
    # ──────────────────────────────────────────────────────────────────────────
    pruned_model = None
    checkpoint_src = None
    pruned_checkpoint = best_candidate.get('pruned_model_checkpoint')

    # 3‑A: Load from the stored checkpoint
    if pruned_checkpoint and os.path.exists(pruned_checkpoint):
        try:
            ckpt = torch.load(pruned_checkpoint, map_location='cpu')
            pruned_model = ckpt.get('complete_model')
            checkpoint_src = pruned_checkpoint
            if pruned_model:
                print(f"[✅] Loaded pruned model object from {pruned_checkpoint}")
        except Exception as e:
            print(f"[⚠️] Failed to load {pruned_checkpoint}: {e}")

    # 3‑B: Fallback to model stored in workflow state
    if pruned_model is None and 'prune' in state and 'model' in state['prune']:
        pruned_model = state['prune']['model']
        checkpoint_src = "current_state_pruned"
        print("[✅] Using pruned model from current workflow state")

    if pruned_model is None:
        print("[❌] No pruned model object found – nothing saved")
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Save BOTH formats: weights-only and full model
    # ──────────────────────────────────────────────────────────────────────────
    base_filename = (
        f"final_pruned_{model_name}_{dataset}_rev{revision}"
        f"_macs{achieved_macs:.3f}G"
    )
    
    # 4-A: Save state dict (weights only) - RECOMMENDED for portability
    weights_filename = f"{base_filename}_weights.pth"
    weights_filepath = os.path.join(output_dir, weights_filename)
    
    torch.save(pruned_model.state_dict(), weights_filepath)
    print(f"[💾] Saved model weights (state_dict): {weights_filepath}")
    print("    ↳ Use with: model.load_state_dict(torch.load(filepath))")
    print("    ↳ Requires: Same architecture code + model = create_model(...)")
    
    # 4-B: Save full model (architecture + weights) - COMPLETE but less portable
    full_filename = f"{base_filename}_full.pt"
    full_filepath = os.path.join(output_dir, full_filename)
    
    torch.save(pruned_model, full_filepath)
    print(f"[💾] Saved full model (architecture + weights): {full_filepath}")
    print("    ↳ Use with: model = torch.load(filepath)")
    print("    ↳ Self-contained but environment-dependent")
    
    # 4-C: Save metadata for both formats
    metadata = {
        'model_name': model_name,
        'dataset': dataset,
        'achieved_macs': achieved_macs,
        'target_macs': target_macs,
        'baseline_macs': baseline_macs,
        'mac_efficiency_percent': mac_efficiency,
        'revision': revision,
        'fine_tuned_accuracy': best_ft_acc,
        'zero_shot_accuracy': best_zs_acc,
        'accuracy_type': acc_label,
        'pruning_method': 'isomorphic_dependency_aware',
        'saved_formats': {
            'weights_only': weights_filename,
            'full_model': full_filename
        },
        'usage_instructions': {
            'weights_only': 'model.load_state_dict(torch.load(filepath)); requires architecture code',
            'full_model': 'model = torch.load(filepath); self-contained but less portable'
        },
        'timestamp': datetime.now().isoformat(),
        'source': checkpoint_src
    }
    
    metadata_filename = f"{base_filename}_metadata.json"
    metadata_filepath = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[📋] Saved metadata: {metadata_filepath}")

    print(
        f"\n[🎉] Model saved in TWO formats!\n"
        f"    🔹 Weights only (recommended): {weights_filename}\n"
        f"    🔹 Full model (complete): {full_filename}\n"
        f"    🔹 Metadata: {metadata_filename}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Update state and return
    # ──────────────────────────────────────────────────────────────────────────
    state.update({
        'final_model_weights_path': weights_filepath,
        'final_model_full_path': full_filepath,
        'final_model_metadata_path': metadata_filepath,
        'final_model_selected': best_candidate,
        'final_model_formats': {
            'weights_only': weights_filepath,
            'full_model': full_filepath,
            'metadata': metadata_filepath
        },
        'final_model_source': checkpoint_src,
        'saved_at': datetime.now().isoformat(),
    })

    return state

