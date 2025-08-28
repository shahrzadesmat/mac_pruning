# Flat Layout Refactor

## Top-level
- utils/                # helpers (timing, wandb, json, analysis core, pruning math/safety, io)
- data/                 # dataset loaders + dataset-specific prompt content
- llm/                  # provider + prompts + enhancer
- main.py               # thin CLI entry
- workflow.py           # orchestration glue
- profiling_agent.py    # agent classes at root
- analysis_agent.py
- pruning_agent.py
- finetune_agent.py
- eval_agent.py
- master_agent.py

## Move your code like this
- utils/timing.py: TimingProfiler, time_it, time_it_async
- utils/logging_wandb.py: add_wandb_args, log_to_wandb
- utils/json_utils.py: parse_llm_json_response, clean/extract helpers, deep_merge
- utils/misc.py: _to_g, debug_file_locations

- utils/analysis_structures.py: MLPCouple, AttentionCouple, PruningState
- utils/analysis_dependency.py: DependencyAnalyzer
- utils/analysis_isomorphism.py: IsomorphicGroup, ViTIsomorphicAnalyzer
- utils/analysis_vit.py: ViTLearningAnalyzer
- utils/analysis_cnn.py: CNNLearningAnalyzer

- utils/pruning_math.py: extract_pruning_ratio, extract_mac_target, extract_mac_allocation_from_analysis, calculate_mac_targets
- utils/pruning_safety.py: PruningSafetyValidator
- utils/io.py: save_final_best_model

- data/loaders.py: get_dataset_loaders, get_cifar10_loaders_pbench, get_imagenet_folder_loaders_pbench
- data/dataset_content.py: get_dataset_specific_content

- llm/provider.py: get_llm
- llm/prompts.py: all prompt-formatters
- llm/enhancer.py: DatasetAwarePromptEnhancer

- profiling_agent.py: ProfilingAgent
- analysis_agent.py: AnalysisAgent
- pruning_agent.py: PruningAgent
- finetune_agent.py: FineTuningAgent
- eval_agent.py: EvaluationAgent
- master_agent.py: MasterAgent (orchestrates others)

- workflow.py: create_pruning_workflow, run_pruning_workflow, validate_results
- main.py: argparse, calls run_pruning_workflow()

## Run
python main.py --help
python main.py --model resnet50 --dataset cifar10 --max_revisions=1

## Import examples
from utils.timing import time_it
from utils.logging_wandb import add_wandb_args
from utils.analysis_structures import PruningState
from utils.pruning_math import calculate_mac_targets
from data.loaders import get_dataset_loaders
from llm.prompts import format_pruning_prompt
