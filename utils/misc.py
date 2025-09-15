from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import torch.nn as nn
import pbench
pbench.forward_patch.patch_timm_forward()
from dotenv import load_dotenv, find_dotenv



def _to_g(val):
    """Normalize a MACs value to units of G (GMACs).
    Accepts either an absolute count (e.g., 4.5e9) or a value already in G.
    Returns None on bad input.
    """
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    # Heuristic: if it's big, assume it's raw ops and convert to G
    return v / 1e9 if v > 1e6 else v

def debug_file_locations(state):
    """Debug function to show where files will be saved"""
    output_dir = state.get('output_dir', './models')
    checkpoint_dir = state.get('checkpoint_dir', './checkpoints')
    current_dir = os.getcwd()
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # print(f"\n[🔍] FILE LOCATION DEBUG:")
    # print(f"   Current working directory: {current_dir}")
    # print(f"   Job ID: {job_id}")
    # print(f"   Output directory (final models): {output_dir}")
    # print(f"   Checkpoint directory (intermediate): {checkpoint_dir}")
    # print(f"   Output dir exists: {os.path.exists(output_dir)}")
    # print(f"   Checkpoint dir exists: {os.path.exists(checkpoint_dir)}")

pbench.forward_patch.patch_timm_forward()

# Load environment variables and setup
_ = load_dotenv(find_dotenv())
# openai.api_key = os.environ['OPENAI_API_KEY']
# nest_asyncio.apply()
