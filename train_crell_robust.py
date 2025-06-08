#!/usr/bin/env python3
"""
Crell Dataset Robust Training - Outlier Detection & Stability
=============================================================

Apply the same robust training approach that worked for MindBigData to Crell dataset.
Crell has different characteristics:
- 64 EEG channels (vs 14 for MindBigData)
- 500Hz sampling rate (vs 128Hz)
- 2250 samples per epoch (vs 256)
- Letter stimuli (vs digits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import scipy.io as sio
from torch.utils.data import DataLoader
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from train_ntvit import (
    NTViTEEGToFMRI,
    EEGFMRIDataset,
    CrellDataLoader  # Use existing working loader
)

# Use existing CrellDataLoader from train_ntvit.py which already works

def detect_crell_outliers(samples, threshold_std=2.5):
    """Detect outlier samples in Crell dataset"""
    
    print(f"🔍 Detecting Crell Outlier Samples (threshold: {threshold_std} std)...")
    
    # Collect statistics from all samples
    all_eeg_stats = []
    
    for i, sample in enumerate(samples):
        eeg_data = sample['eeg_data']
        
        stats = {
            'index': i,
            'mean': np.mean(eeg_data),
            'std': np.std(eeg_data),
            'min': np.min(eeg_data),
            'max': np.max(eeg_data),
            'range': np.max(eeg_data) - np.min(eeg_data),
            'letter': sample['letter']
        }
        all_eeg_stats.append(stats)
    
    # Calculate global statistics
    means = [s['mean'] for s in all_eeg_stats]
    stds = [s['std'] for s in all_eeg_stats]
    ranges = [s['range'] for s in all_eeg_stats]
    
    global_mean_mean = np.mean(means)
    global_mean_std = np.std(means)
    global_std_mean = np.mean(stds)
    global_std_std = np.std(stds)
    global_range_mean = np.mean(ranges)
    global_range_std = np.std(ranges)
    
    print(f"📊 Crell Global Statistics:")
    print(f"  Mean: {global_mean_mean:.4f} ± {global_mean_std:.4f}")
    print(f"  Std: {global_std_mean:.4f} ± {global_std_std:.4f}")
    print(f"  Range: {global_range_mean:.4f} ± {global_range_std:.4f}")
    
    # Detect outliers
    outlier_indices = []
    
    for stats in all_eeg_stats:
        is_outlier = False
        reasons = []
        
        # Check mean outlier
        if abs(stats['mean'] - global_mean_mean) > threshold_std * global_mean_std:
            is_outlier = True
            reasons.append(f"mean={stats['mean']:.4f}")
        
        # Check std outlier
        if abs(stats['std'] - global_std_mean) > threshold_std * global_std_std:
            is_outlier = True
            reasons.append(f"std={stats['std']:.4f}")
        
        # Check range outlier
        if abs(stats['range'] - global_range_mean) > threshold_std * global_range_std:
            is_outlier = True
            reasons.append(f"range={stats['range']:.4f}")
        
        # Check extreme values
        if abs(stats['min']) > 1e4 or abs(stats['max']) > 1e4:
            is_outlier = True
            reasons.append(f"extreme_values=[{stats['min']:.2e}, {stats['max']:.2e}]")
        
        # Check zero variance
        if stats['std'] < 1e-6:
            is_outlier = True
            reasons.append(f"zero_variance={stats['std']:.2e}")
        
        if is_outlier:
            outlier_indices.append(stats['index'])
            print(f"  ⚠️  Outlier {stats['index']} (letter {stats['letter']}): {', '.join(reasons)}")
    
    # Filter out outliers
    filtered_samples = [samples[i] for i in range(len(samples)) if i not in outlier_indices]
    
    print(f"📊 Crell Outlier Detection Results:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Outlier samples: {len(outlier_indices)}")
    print(f"  Filtered samples: {len(filtered_samples)}")
    print(f"  Retention rate: {len(filtered_samples)/len(samples)*100:.1f}%")
    
    return filtered_samples, outlier_indices

def create_crell_robust_loaders(datasets_dir: str, batch_size: int = 4):
    """Create robust Crell data loaders with outlier filtering"""

    print(f"🧠 Creating Robust Crell Data Loaders...")

    datasets_path = Path(datasets_dir)

    # Load Crell with existing working loader
    crell_loader = CrellDataLoader(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=None  # No limit - get all available
    )
    
    samples = crell_loader.samples
    print(f"✓ Loaded {len(samples)} raw Crell samples")
    
    if len(samples) == 0:
        raise ValueError("No Crell samples loaded!")
    
    # Check letter distribution
    letters = [sample['letter'] for sample in samples]
    print(f"📊 Letter distribution:")
    for letter in ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']:
        count = letters.count(letter)
        print(f"  Letter {letter}: {count} samples")
    
    # Detect and filter outliers
    filtered_samples, outlier_indices = detect_crell_outliers(samples, threshold_std=2.5)
    
    if len(filtered_samples) < 50:
        print(f"⚠️  Too few samples after filtering ({len(filtered_samples)}), using less strict filtering")
        filtered_samples, outlier_indices = detect_crell_outliers(samples, threshold_std=3.5)
    
    # Robust normalization for Crell (64 channels, higher sampling rate)
    print(f"🔧 Applying robust normalization for Crell...")
    
    # Calculate robust statistics
    all_eeg_data = []
    for sample in filtered_samples:
        all_eeg_data.append(sample['eeg_data'].flatten())
    
    all_eeg_flat = np.concatenate(all_eeg_data)
    
    # Use median and MAD for robust normalization
    eeg_median = np.median(all_eeg_flat)
    eeg_mad = np.median(np.abs(all_eeg_flat - eeg_median))
    
    print(f"📊 Crell robust statistics: median={eeg_median:.4f}, MAD={eeg_mad:.4f}")
    
    # Apply robust normalization
    for sample in filtered_samples:
        eeg_data = sample['eeg_data']
        # Robust z-score using median and MAD
        eeg_normalized = (eeg_data - eeg_median) / (1.4826 * eeg_mad + 1e-8)
        # Conservative clipping for Crell (might have different range)
        eeg_normalized = np.clip(eeg_normalized, -4.0, 4.0)
        sample['eeg_data'] = eeg_normalized
    
    # Check letter distribution after filtering
    letters_filtered = [sample['letter'] for sample in filtered_samples]
    print(f"📊 Letter distribution after filtering:")
    for letter in ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']:
        count = letters_filtered.count(letter)
        print(f"  Letter {letter}: {count} samples")
    
    # Split samples
    train_samples = filtered_samples[:int(0.8 * len(filtered_samples))]
    val_samples = filtered_samples[int(0.8 * len(filtered_samples)):]
    
    # Create datasets
    train_dataset = EEGFMRIDataset(train_samples)
    val_dataset = EEGFMRIDataset(val_samples)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"✓ Created robust Crell data loaders:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Outliers removed: {len(outlier_indices)}")
    
    return train_loader, val_loader, outlier_indices

def main():
    """Main function for Crell robust training"""
    
    print("🧠 Crell Dataset Robust Training")
    print("=" * 50)
    print("📋 Crell Characteristics:")
    print("  • 64 EEG channels (vs 14 for MindBigData)")
    print("  • 500Hz sampling rate (vs 128Hz)")
    print("  • 2250 samples per epoch (vs 256)")
    print("  • Letter stimuli: a,d,e,f,j,n,o,s,t,v")
    print("  • Visual phases only (4.5s duration)")
    
    # Configuration
    datasets_dir = "datasets"
    output_dir = "crell_robust_outputs"
    batch_size = 2  # Smaller batch size due to larger data (64 channels, 2250 samples)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  Dataset: Crell (with outlier filtering)")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size} (smaller due to larger data)")
    print(f"  Output: {output_dir}")
    
    # Check if datasets exist
    datasets_path = Path(datasets_dir)
    required_files = ["S01.mat", "crellStimuli"]
    
    missing_files = []
    for file in required_files:
        if not (datasets_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing required files/directories:")
        for file in missing_files:
            print(f"  - {datasets_path / file}")
        return
    
    try:
        # Create robust data loaders
        train_loader, val_loader, outlier_indices = create_crell_robust_loaders(
            datasets_dir=datasets_dir,
            batch_size=batch_size
        )
        
        print(f"\n🎉 Crell robust data loading completed!")
        print(f"📊 Ready for training with {len(train_loader.dataset)} train samples")
        print(f"📊 Validation with {len(val_loader.dataset)} val samples")
        print(f"📊 Outliers removed: {len(outlier_indices)}")
        
        # Test a few batches
        print(f"\n🧪 Testing data loading...")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
            
            eeg_data = batch['eeg_data']
            target_fmri = batch['translated_fmri_target']
            
            print(f"  Batch {i+1}: EEG {eeg_data.shape}, fMRI {target_fmri.shape}")
            print(f"    EEG range: [{torch.min(eeg_data):.4f}, {torch.max(eeg_data):.4f}]")
            print(f"    fMRI range: [{torch.min(target_fmri):.4f}, {torch.max(target_fmri):.4f}]")
            print(f"    No NaN/Inf: {torch.isfinite(eeg_data).all() and torch.isfinite(target_fmri).all()}")
        
        print(f"\n✅ Crell robust data loading successful!")
        print(f"💡 Ready for robust training with 64-channel EEG data")
        
    except Exception as e:
        print(f"❌ Crell robust loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
