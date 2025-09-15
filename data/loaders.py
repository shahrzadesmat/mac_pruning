import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import pbench


import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import pbench

# ===================================================================
# MODULE-LEVEL CACHE - This survives across all function calls
# ===================================================================
_DATASET_CACHE = {}

def _create_cache_key(dataset: str, data_path: str, batch_size: int, subset_fraction: float = 1.0) -> str:
    """Create a unique cache key for dataset loaders"""
    return f"{dataset}_{data_path}_{batch_size}_{subset_fraction}"

def get_imagenet_folder_loaders_pbench(data_path, batch_size=64, num_workers=16, subset_fraction=1.0):
    """
    CACHED version - will only load ImageNet once per unique configuration
    """
    # Create cache key
    cache_key = _create_cache_key("imagenet", data_path, batch_size, subset_fraction)
    
    # Check if we already have this exact configuration cached
    if cache_key in _DATASET_CACHE:
        # print(f"[DEBUG] Using cached ImageNet loaders: {cache_key}")
        return _DATASET_CACHE[cache_key]
    
    # print(f"[DEBUG] Creating new ImageNet loaders: {cache_key}")
    print(f"[📂] Loading ImageNet from folder structure: {data_path}")

    # ADD: Subset info message
    if subset_fraction < 1.0:
        print(f"[🔬] ImageNet TESTING MODE: Using {subset_fraction*100:.1f}% of training data")

    # Define paths
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    
    # Verify directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    print(f"[✅] Found train directory: {train_dir}")
    print(f"[✅] Found val directory: {val_dir}")
    
    # EXACT same parameters as your working prune_v5.py
    use_imagenet_mean_std = True
    interpolation = 'bicubic'
    val_resize = 256
    
    # Convert interpolation string to enum (exactly like prune_v5.py)
    interpolation = getattr(T.InterpolationMode, interpolation.upper())
    
    print('Parsing dataset...')
    
    train_dst = ImageFolder(
        os.path.join(data_path, 'train'), 
        transform=pbench.data.presets.ClassificationPresetEval(
            crop_size=224,                    # Required parameter
            resize_size=val_resize,           # This was missing in ma_img08.py!
            mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
            std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
            interpolation=interpolation
        )
    )
    val_dst = ImageFolder(
        os.path.join(data_path, 'val'), 
        transform=pbench.data.presets.ClassificationPresetEval(
            crop_size=224,                    # Required parameter
            resize_size=val_resize,           # This was missing in ma_img08.py!
            mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
            std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
            interpolation=interpolation
        )
    )

    if subset_fraction < 1.0:
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset
        
        total_train = len(train_dst)
        all_indices = list(range(total_train))
        _, subset_indices = train_test_split(
            all_indices, 
            test_size=subset_fraction,
            random_state=42
        )
        
        train_dst = Subset(train_dst, subset_indices)
        print(f"[🔬] Created subset: {len(subset_indices):,} samples ({subset_fraction*100:.1f}% of {total_train:,})")

    print(f"[📊] Train dataset: {len(train_dst)} images, {len(train_dst.classes if hasattr(train_dst, 'classes') else train_dst.dataset.classes)} classes")
    print(f"[📊] Val dataset: {len(val_dst)} images, {len(val_dst.classes)} classes")

    train_loader = torch.utils.data.DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dst, 
        batch_size=batch_size * 2, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"[✅] Created pbench ImageNet loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # CACHE THE RESULT - This is the key fix!
    _DATASET_CACHE[cache_key] = (train_loader, val_loader)
    print(f"[DEBUG] Cached ImageNet loaders with key: {cache_key}")
    
    return train_loader, val_loader

def get_cifar10_loaders_pbench(batch_size=64, num_workers=16):
    """CACHED CIFAR-10 loaders"""
    # Create cache key
    cache_key = _create_cache_key("cifar10", "./data", batch_size, 1.0)
    
    # Check cache first
    if cache_key in _DATASET_CACHE:
        # print(f"[DEBUG] Using cached CIFAR-10 loaders: {cache_key}")
        return _DATASET_CACHE[cache_key]
    
    # print(f"[DEBUG] Creating new CIFAR-10 loaders: {cache_key}")
    
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    
    # Use standard transforms for CIFAR-10 to avoid any pbench conflicts
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        T.Resize((224, 224), antialias=True)  # Resize to 224 for model compatibility
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        T.Resize((224, 224), antialias=True)
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    # Cache the result
    _DATASET_CACHE[cache_key] = (train_loader, val_loader)
    #print(f"[DEBUG] Cached CIFAR-10 loaders with key: {cache_key}")
    
    return train_loader, val_loader

def get_dataset_loaders(dataset: str, data_path: str, batch_size: int = 64, num_workers: int = 16, imagenet_subset: float = 1.0):
    """Main entry point - now with caching support"""
    if dataset.lower() == 'imagenet':
        return get_imagenet_folder_loaders_pbench(data_path, batch_size, num_workers, imagenet_subset)
    elif dataset.lower() == 'cifar10':
        return get_cifar10_loaders_pbench(batch_size, num_workers)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def clear_dataset_cache():
    """Utility function to clear the cache if needed"""
    global _DATASET_CACHE
    _DATASET_CACHE.clear()
    # print("[DEBUG] Dataset cache cleared")

def get_cache_info():
    """Utility function to see what's cached"""
    # print(f"[DEBUG] Cache contains {len(_DATASET_CACHE)} entries:")
    for key in _DATASET_CACHE.keys():
        print(f"  - {key}")
