from dataclasses import dataclass
from typing import TypedDict, Any, List, Dict
import traceback
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from llm.provider import get_llm
from data.dataset_content import get_dataset_specific_content
from data.loaders import get_imagenet_folder_loaders_pbench
from utils.timing import time_it, time_it_async
import copy
from utils import logging_wandb
import os
from datetime import datetime



class FineTuningAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def get_device(self):
        """Get the best available device for training"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _setup_dataset_loaders(self, dataset: str, data_path: str, batch_size=64, num_workers=16):
        """Setup dataset-appropriate data loaders with augmentation for training"""
        
        if dataset.lower() == 'imagenet':
            return self._setup_imagenet_data(data_path, batch_size, num_workers)
        else:
            return self._setup_cifar10_data(batch_size, num_workers)

    def _setup_cifar10_data(self, batch_size=64, num_workers=16):
        """Setup CIFAR-10 data loaders with augmentation for training and use test set for validation"""
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        # Training transform with data augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((224, 224), antialias=True)
        ])

        # Validation/test transform without augmentation
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((224, 224), antialias=True)
        ])

        # Load training dataset
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )

        # Load official test set as validation set
        val_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=val_transform
        )

        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader



    def _setup_imagenet_data(self, data_path: str, batch_size=64, num_workers=16):
        """UPDATED: Get subset from state"""
        # Get subset from current state if available
        imagenet_subset = getattr(self, '_current_state', {}).get('imagenet_subset', 1.0)
        return get_imagenet_folder_loaders_pbench(data_path, batch_size, num_workers, imagenet_subset)


    def _validate_model(self, model, val_loader, device, dataset: str):
        """Unified validation for CIFAR-10 and ImageNet."""
        model.eval()
        model.to(device)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_correct_top5 = 0

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Top-5 accuracy (only meaningful for ImageNet)
                if dataset.lower() == 'imagenet':
                    _, top5_preds = outputs.topk(5, dim=1)
                    val_correct_top5 += top5_preds.eq(targets.view(-1, 1)).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        if dataset.lower() == 'imagenet':
            val_top5_acc = 100. * val_correct_top5 / val_total
            return val_loss, val_acc, val_top5_acc
        else:
            return val_loss, val_acc, None

    def _fix_classification_head(self, model, num_classes: int, device):
        """Fix classification head for the appropriate number of classes"""
        
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            if model.head.out_features != num_classes:
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes).to(device)
                # print(f"[🔧] Fixed classification head: {in_features} -> {num_classes} classes")
                
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            if model.fc.out_features != num_classes:
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes).to(device)
                # print(f"[🔧] Fixed FC layer: {in_features} -> {num_classes} classes")
                
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            if model.classifier.out_features != num_classes:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes).to(device)
                # print(f"[🔧] Fixed classifier: {in_features} -> {num_classes} classes")

    def _get_dataset_specific_params(self, dataset: str, is_vit_model: bool):
        """Get dataset and model-specific training parameters"""
        
        if dataset.lower() == 'imagenet':
            if is_vit_model:
                return {
                    'num_epochs': 5,
                    'learning_rate': 0.0001,
                    'weight_decay': 0.01,
                    'optimizer': 'adamw',
                    'batch_size_factor': 1.0  # Smaller batches for ImageNet
                }
            else:
                return {
                    'num_epochs': 5,
                    'learning_rate': 0.01,
                    'weight_decay': 5e-4,
                    'optimizer': 'sgd',
                    'batch_size_factor': 1.0
                }
        else:  # CIFAR-10
            if is_vit_model:
                return {
                    'num_epochs': 5,
                    'learning_rate': 0.005,
                    'weight_decay': 0.05,
                    'optimizer': 'adamw',
                    'batch_size_factor': 1.0
                }
            else:
                return {
                    'num_epochs': 5,
                    'learning_rate': 0.01,
                    'weight_decay': 5e-4,
                    'optimizer': 'sgd',
                    'batch_size_factor': 1.0
                }
            


    def setup_scheduler(optimizer, scheduler_type, num_epochs, total_steps=None):
        """Setup appropriate scheduler for pruned model fine-tuning"""
        
        if scheduler_type == 'constant_with_warmup':
            # Keep LR mostly constant with slight warmup
            def lr_lambda(epoch):
                if epoch == 0:
                    return 0.1  # Start at 10% of target LR
                else:
                    return 1.0  # Then constant
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        elif scheduler_type == 'cosine_slow':
            # Much slower cosine decay
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs * 3,  # 3x slower decay
                eta_min=optimizer.param_groups[0]['lr'] * 0.1
            )
        
        elif scheduler_type == 'step':
            # Step decay every few epochs
            return torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=max(1, num_epochs // 2), 
                gamma=0.5
            )
        
        else:
            # Default: very gentle cosine
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs,
                eta_min=optimizer.param_groups[0]['lr'] * 0.01  # Don't go to zero
            )


    @time_it_async("5. Fine-tuning Agent")
    async def fine_tune(self, state: Dict) -> Dict:
        """COMPLETELY REVISED: Dataset-aware fine-tuning with pruning considerations"""

        self._current_state = state
        
        # Extract dataset information
        dataset = state.get("dataset", "cifar10")
        num_classes = state.get("num_classes", 10)
        data_path = state.get("data_path", "./data")
        
        print(f"\n[🔄] Starting REVISED {dataset.upper()} fine-tuning process...")
        # print(f"[🔄] Dataset: {dataset} ({num_classes} classes)")

        # Check for pruning success
        pruning_success = (
            state.get("pruning_results", {}).get("success", False) or
            state.get("prune", {}).get("pruning_results", {}).get("success", False)
        )
        
        if not pruning_success:
            print("[⚠️] Skipping fine-tuning: pruning failed or model is missing.")
            state["fine_tuning_results"] = {"skipped": True}
            return state

        try:
            if 'prune' not in state or 'model' not in state['prune']:
                return {
                    **state,
                    'error': 'No pruned model found in state'
                }

            # Get model and pruning details
            model = state['prune']['model']
            model_name = state.get('model_name', 'unknown_model')
            
            # FIX: Get device EARLY
            device = self.get_device()
            print(f"[💻] Fine-tuning on device: {device}")
            model = model.to(device)
            
            # FIX: Check if ViT AFTER getting model
            is_vit_model = any(isinstance(m, timm.models.vision_transformer.Attention) for m in model.modules())
            print(f"[🔍] Model type: {'ViT' if is_vit_model else 'CNN'}")
            
            # Get pruning metrics
            pruning_results = state.get('prune', {}).get('pruning_results', {})
            achieved_ratio = pruning_results.get('achieved_ratio', 0)
            
            # FIX: Get zero-shot accuracy EARLY for comparison
            history = state.get('history', [{}])
            last_entry = history[-1] if history else {}
            
            if dataset.lower() == 'imagenet':
                zero_shot = last_entry.get('zero_shot_top1_accuracy', 0)
            else:
                zero_shot = last_entry.get('zero_shot_accuracy', 0)
            
            # Get user-specified number of epochs
            params = self._get_dataset_specific_params(dataset, is_vit_model)
            num_epochs = params['num_epochs']  # Respect user setting (you set this to 1)
            
            # SIMPLE FIX: Use very conservative LR for pruned models
            if dataset.lower() == 'imagenet' and is_vit_model:
                if achieved_ratio > 0.15:  # Heavily pruned
                    base_lr = 0.00005      # Very conservative
                else:
                    base_lr = 0.0001       # Still conservative
            else:  # CIFAR-10 or CNN
                base_lr = 0.001 if achieved_ratio < 0.1 else 0.0005
            
            print(f"[🔧] Using LR {base_lr} for {achieved_ratio:.1%} pruned model")
            # print(f"[⚙️] User set epochs: {num_epochs}")
            
            # FIX: Setup data loaders EARLY
            train_loader, val_loader = self._setup_dataset_loaders(dataset, data_path, 64)
            # print(f"[📊] Setup data loaders: {len(train_loader)} train, {len(val_loader)} val")
            
            # Setup optimizer
            if is_vit_model:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=base_lr,
                    weight_decay=0.02,
                    betas=(0.9, 0.999)
                )
                print(f"[🧮] Using AdamW optimizer for ViT")
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=base_lr,
                    momentum=0.9,
                    weight_decay=1e-4
                )
                print(f"[🧮] Using SGD optimizer for CNN")
            
            # SIMPLE SCHEDULER: Just constant LR (no fancy scheduling)
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            
            # Loss with light label smoothing
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
            
            # Track best model
            best_val_acc = 0.0
            best_model_state = None
            
            print(f"[🎯] Simple fine-tuning: {num_epochs} epochs, LR {base_lr}")
            
            # Training loop
            for epoch in range(num_epochs):
                print(f"\n[📊] Epoch {epoch+1}/{num_epochs}")
                
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    # Handle data format
                    if isinstance(batch, dict):
                        inputs = batch['pixel_values'].to(device)
                        targets = batch['label'].to(device)
                    else:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Track metrics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                    
                    # Progress update
                    if batch_idx % 100 == 0:
                        current_acc = 100. * train_correct / train_total
                        print(f'[🔄] Batch {batch_idx}: Loss {loss.item():.3f}, Acc {current_acc:.2f}%')
                
                # Validation
                val_result = self._validate_model(model, val_loader, device, dataset)
                if dataset.lower() == 'imagenet':
                    val_loss, val_acc, val_top5 = val_result
                    print(f'[📊] Validation: Top-1 {val_acc:.2f}%, Top-5 {val_top5:.2f}%')
                else:
                    val_loss, val_acc, _ = val_result
                    print(f'[📊] Validation: Accuracy {val_acc:.2f}%')
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f'[✅] New best: {best_val_acc:.2f}%')
                else:
                    print(f'[📉] No improvement (best: {best_val_acc:.2f}%)')
                
                # EARLY STOPPING: Only if more than 1 epoch and accuracy degrades significantly
                if num_epochs > 1 and epoch > 0 and val_acc < (best_val_acc - 2.0):  # 2% degradation tolerance
                    print(f"[🛑] Early stopping - validation degrading")
                    break
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
                print(f"[🏆] Using best model with {best_val_acc:.2f}% accuracy")
            
            # Calculate improvement
            improvement = best_val_acc - zero_shot
            
            print(f"\n[📈] Fine-tuning Results:")
            print(f"  - Zero-shot: {zero_shot:.2f}%")
            print(f"  - Fine-tuned: {best_val_acc:.2f}%")
            print(f"  - Improvement: {improvement:.2f}%")
            print(f"  - Status: {'✅ SUCCESS' if improvement > 0 else '⚠️ NO GAIN'}")

            # Log fine-tuning results to WandB
            logging_wandb.log_to_wandb({
                "best_accuracy": best_val_acc,
                "accuracy_improvement": improvement,
                "zero_shot_baseline": zero_shot,
                "learning_rate": base_lr,
                "num_epochs": num_epochs,
                "training_approach": "conservative_lr",
            }, step_name="fine_tuning", dataset=dataset)
            
            # Save checkpoint
            checkpoint_dir = state.get('checkpoint_dir', './checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_filename = f"fine_tuned_{model_name}_{dataset}_rev{state.get('revision_number', 0)}_ratio{achieved_ratio:.3f}_acc{best_val_acc:.1f}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

            torch.save({
                'complete_model': model,
                'model_state_dict': model.state_dict(),
                'dataset': dataset,
                'achieved_ratio': achieved_ratio,
                'zero_shot_accuracy': zero_shot,
                'fine_tuned_accuracy': best_val_acc,
                'improvement': improvement,
                'revision_number': state.get('revision_number', 0),
                'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)

            # print(f"[💾] Saved fine-tuned checkpoint: {checkpoint_path}")
            
            # Return results
            return {
                **state,
                'fine_tuning_results': {
                    'model': model,
                    'best_accuracy': best_val_acc,
                    'accuracy_improvement': improvement,
                    'zero_shot_accuracy': zero_shot,
                    'method': f'Conservative LR {base_lr}, {num_epochs} epochs',
                    'simplified_approach': True,
                    'dataset': dataset,
                    'checkpoint_path': checkpoint_path
                }
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[❌] Error in REVISED {dataset} fine-tuning: {error_trace}")
            return {
                **state,
                'error': str(e)
            }
        
