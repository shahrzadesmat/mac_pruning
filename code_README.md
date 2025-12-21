# Code Documentation

This directory contains the complete implementation of our multi-agent LLM-based pruning framework.

---

## Directory Structure

```
code/
├── README.md                   # This file
├── requirements.txt            # Python dependencies with exact versions
│
├── main.py                     # Primary entry point for pruning experiments
├── workflow.py                 # Multi-agent workflow orchestration
│
├── master_agent.py             # Convergence monitoring and revision control
├── analysis_agent.py           # LLM-based pruning strategy generation
├── profiling_agent.py          # MAC profiling and dependency analysis
├── pruning_agent.py            # Structured pruning execution
├── finetune_agent.py           # Post-pruning fine-tuning
├── eval_agent.py               # Model evaluation and metrics
│
├── deepseek_llm.py             # LLM API integration (Claude/OpenRouter)
│
├── data/                       # Dataset loaders and augmentation
│   ├── loaders.py             # ImageNet data loaders
│   └── dataset_content.py     # Dataset-specific prompt content
│
├── llm/                        # LLM integration
│   ├── provider.py            # LLM provider configuration
│   ├── prompts.py             # Prompt templates for pruning strategies
│   └── enhancer.py            # Dataset-aware prompt enhancement
│
├── utils/                      # Utility functions
│   ├── timing.py              # Performance profiling
│   ├── logging_wandb.py       # Experiment tracking
│   ├── json_utils.py          # JSON parsing and validation
│   ├── analysis_*.py          # Dependency analysis, isomorphic grouping
│   ├── pruning_math.py        # MAC calculations and ratio computations
│   ├── pruning_safety.py      # Safety constraint validation
│   └── io.py                  # Checkpoint saving/loading
│
├── scripts/                    # Standalone scripts
│   ├── finetuning.py          # Standalone fine-tuning script
│   └── evaluate.py            # Standalone evaluation script
│
├── ablations/                  # Ablation study configurations
└── pbench/                     # Performance benchmarking tools
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### 3. Run Pruning

```bash
python main.py \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 2.0 \
    --batch_size 256 \
    --epochs 100
```

---

## Command-Line Arguments

### Model Configuration
```
--model              Model architecture (resnet50, resnet101, convnext_base, deit_tiny)
--pretrained         Use pre-trained weights (default: True)
```

### Dataset Configuration
```
--dataset            Dataset name (imagenet)
--data_path          Path to dataset directory
--batch_size         Training batch size (default: 256)
--num_workers        Data loader workers (default: 16)
```

### Pruning Configuration
```
--target_macs        Target MAC budget in GigaMACs (e.g., 2.0)
--macs_overshoot_tolerance_pct    Overshoot tolerance (default: 1.0%)
--macs_undershoot_tolerance_pct   Undershoot tolerance (default: 5.0%)
--max_revisions      Maximum LLM revision attempts (default: 100)
--importance         Importance criterion (taylor, magnitude, random)
```

### Training Configuration
```
--epochs             Fine-tuning epochs (default: 100)
--lr                 Learning rate (default: 0.001)
--weight_decay       Weight decay (default: 0.05)
--optimizer          Optimizer (adamw, sgd)
```

### LLM Configuration
```
--llm_model          LLM model (claude-sonnet-4-20250514, gpt-4, llama-3-70b)
--llm_temperature    LLM temperature (default: 0.7)
--llm_max_tokens     Maximum tokens per completion (default: 1000)
```

### Logging
```
--wandb_project      Weights & Biases project name
--wandb_mode         Logging mode (online, offline, disabled)
--output_dir         Output directory for checkpoints
```

---

## Usage Examples

### Example 1: ResNet-50 Pruning (Paper Table 1)

```bash
python main.py \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 2.0 \
    --macs_overshoot_tolerance_pct 1.0 \
    --macs_undershoot_tolerance_pct 5.0 \
    --max_revisions 100 \
    --batch_size 256 \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 0.05 \
    --wandb_project MAC_Pruning \
    --output_dir ./outputs
```

**Expected Output:**
- Converges in 3-5 LLM revisions
- Final MACs: ~1.77G (within tolerance)
- Final accuracy: ~77.04%

### Example 2: DeiT-Tiny Pruning

```bash
python main.py \
    --model deit_tiny \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --target_macs 0.6 \
    --batch_size 256 \
    --epochs 100
```

### Example 3: Evaluate a Checkpoint

```bash
python eval_agent.py \
    --checkpoint ../checkpoints/resnet50_finetuned_clean.pth \
    --model resnet50 \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --batch_size 256
```

### Example 4: Standalone Fine-tuning

```bash
cd scripts

python finetuning.py \
    --model resnet50 \
    --checkpoint ../checkpoints/resnet50_pruned_clean.pt \
    --data_path /path/to/imagenet \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 256
```

---

## Ablation Studies

### LLM Selection Ablation (Table 6)

**GPT-4:**
```bash
python main.py --model resnet50 --target_macs 2.0 --llm_model gpt-4
```

**Llama-3-70B:**
```bash
python main.py --model resnet50 --target_macs 2.0 --llm_model llama-3-70b
```

---

## Module Descriptions

### Core Agent Classes

**`MasterAgent`** (`master_agent.py`):
- Orchestrates multi-agent workflow
- Monitors convergence to MAC target
- Tracks revision history
- Detects catastrophic failures

**`AnalysisAgent`** (`analysis_agent.py`):
- Interfaces with LLM for strategy generation
- Parses and validates LLM responses
- Learns from previous revision feedback
- Generates layer-wise pruning ratios

**`ProfilingAgent`** (`profiling_agent.py`):
- Computes model MACs and parameters
- Analyzes dependency graphs
- Identifies isomorphic layer groups
- Validates structural constraints

**`PruningAgent`** (`pruning_agent.py`):
- Executes structured pruning using torch-pruning
- Computes importance scores (Taylor, magnitude)
- Applies safety constraints for ViTs
- Rounds channels for hardware efficiency

**`FineTuningAgent`** (`finetune_agent.py`):
- Fine-tunes pruned models
- Implements data augmentation
- Uses EMA for weight averaging
- Tracks training metrics

**`EvaluationAgent`** (`eval_agent.py`):
- Evaluates model accuracy
- Computes performance metrics
- Profiles hardware latency
- Generates evaluation reports

### Utility Modules

**`utils/analysis_*.py`**:
- Dependency analysis for structured pruning
- Isomorphic group detection for ViTs
- Learning rate and pruning ratio calculations

**`utils/pruning_*.py`**:
- MAC target calculations
- Safety constraint validation
- Pruning ratio extraction from LLM responses

**`llm/prompts.py`**:
- Prompt templates for LLM interactions
- Dataset-specific content injection
- Revision history formatting

---

## Dataset Preparation

### ImageNet-1K

```bash
# Expected structure
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

---

## Experiment Tracking

All experiments log to Weights & Biases:
- Training curves (loss, accuracy)
- MAC statistics per revision
- LLM strategy JSONs
- Hardware performance metrics

**Setup:**
```bash
wandb login
export WANDB_PROJECT=MAC_Pruning
```

---

## Reproducibility Checklist

✅ Random seed: 42 (set in all scripts)  
✅ Exact package versions in `requirements.txt`  
✅ Pre-trained models from timm library  
✅ Command-line examples for all experiments  
✅ Dataset preparation instructions  
✅ Hardware specifications in Compute Reporting Form  

---

## Troubleshooting

**Issue: Out of memory during pruning**
```bash
# Reduce calibration batch size
python main.py ... --calibration_batch_size 32
```

**Issue: LLM API errors**
```bash
# Verify API key
echo $OPENROUTER_API_KEY

# Use offline mode for testing
python main.py ... --llm_model none --use_heuristic_pruning
```

**Issue: Slow data loading**
```bash
# Increase workers
python main.py ... --num_workers 32
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥24GB VRAM
- **RAM**: ≥64GB system memory
- **Storage**: ~200GB for ImageNet + checkpoints

**Tested on:**
- NVIDIA H200 (paper experiments)
- NVIDIA A100 (40GB/80GB)
- NVIDIA RTX 3090 (24GB)

---

## Additional Notes

1. **First run**: Downloads pre-trained models from timm (~200MB per model)
2. **LLM calls**: Typically 3-5 API calls per pruning run (~$0.05 cost)
3. **Training time**: ResNet-50 fine-tuning takes ~8 hours on H200
4. **Checkpoints**: Saved to `--output_dir` every 10 epochs

---

For questions, refer to:
- Compute Reporting Form (submitted separately) for parameter specifications
- CHECKPOINTS_README.md for checkpoint loading
- Root README.md for reproduction commands
