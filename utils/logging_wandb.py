import wandb

def add_wandb_args(parser):
    """Add WandB-related arguments to the argument parser"""
    parser.add_argument('--wandb_project', type=str, default='anonymous_project', 
                        help='WandB project name (anonymous)')
    parser.add_argument('--wandb_name', type=str, default=None, 
                       help='WandB run name (auto-generated if not provided)')
    parser.add_argument('--wandb_mode', type=str, default='online', 
                       choices=['online', 'offline', 'disabled'],
                       help='WandB mode: online, offline, or disabled')
    return parser

def log_to_wandb(metrics_dict, step_name=None, dataset=None):
    """Helper function to safely log metrics to WandB"""
    try:
        if step_name:
            metrics_dict = {f"{step_name}/{k}": v for k, v in metrics_dict.items()}
        
        # Add dataset prefix if provided
        if dataset:
            metrics_dict[f"dataset"] = dataset
            
        wandb.log(metrics_dict)
    except Exception as e:
        print(f"[⚠️] Failed to log to WandB: {e}")
