#!/usr/bin/env python3
"""
Simple Crell Training - Using Existing Loader + Outlier Detection
=================================================================

Use the existing working CrellDataLoader and add robust training features.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from train_ntvit import (
    NTViTEEGToFMRI, 
    EEGFMRIDataset,
    CrellDataLoader
)

def detect_crell_outliers(samples, threshold_std=2.5):
    """Detect outlier samples in Crell dataset"""
    
    print(f"🔍 Detecting Crell Outlier Samples (threshold: {threshold_std} std)...")
    
    if len(samples) == 0:
        print("❌ No samples to analyze")
        return [], []
    
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
    if len(samples) > 0:
        print(f"  Retention rate: {len(filtered_samples)/len(samples)*100:.1f}%")
    
    return filtered_samples, outlier_indices

def create_crell_data_loaders(datasets_dir: str, batch_size: int = 2):
    """Create Crell data loaders with existing loader + outlier filtering"""
    
    print(f"🧠 Creating Crell Data Loaders...")
    
    datasets_path = Path(datasets_dir)
    
    # Load Crell with existing working loader - get more samples
    print(f"📊 Loading Crell dataset...")
    crell_loader = CrellDataLoader(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=1000  # Try to get more samples
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
    filtered_samples, outlier_indices = detect_crell_outliers(samples, threshold_std=3.0)
    
    if len(filtered_samples) < 10:
        print(f"⚠️  Too few samples after filtering ({len(filtered_samples)}), using all samples")
        filtered_samples = samples
        outlier_indices = []
    
    # Robust normalization for Crell (64 channels, higher sampling rate)
    print(f"🔧 Applying robust normalization for Crell...")
    
    if len(filtered_samples) > 0:
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
            # Conservative clipping for Crell
            eeg_normalized = np.clip(eeg_normalized, -4.0, 4.0)
            sample['eeg_data'] = eeg_normalized
    
    # Check letter distribution after filtering
    letters_filtered = [sample['letter'] for sample in filtered_samples]
    print(f"📊 Letter distribution after filtering:")
    for letter in ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']:
        count = letters_filtered.count(letter)
        print(f"  Letter {letter}: {count} samples")
    
    # Split samples (80% train, 20% val)
    if len(filtered_samples) >= 10:
        train_samples = filtered_samples[:int(0.8 * len(filtered_samples))]
        val_samples = filtered_samples[int(0.8 * len(filtered_samples)):]
    else:
        # If very few samples, use all for training
        train_samples = filtered_samples
        val_samples = filtered_samples[:2] if len(filtered_samples) >= 2 else filtered_samples
    
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
        drop_last=True if len(train_samples) > batch_size else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"✓ Created Crell data loaders:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Outliers removed: {len(outlier_indices)}")
    
    return train_loader, val_loader, outlier_indices

def test_crell_training(datasets_dir: str, device: str = 'cuda'):
    """Test Crell training with a few epochs"""
    
    print(f"🧠 Testing Crell Training...")
    print("=" * 50)
    
    # GPU setup
    if device == 'cuda' and torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
        print("⚠️  Using CPU")
    
    try:
        # Create data loaders
        train_loader, val_loader, outlier_indices = create_crell_data_loaders(
            datasets_dir, batch_size=2
        )
        
        # Determine channel count from loaded data
        sample_batch = next(iter(train_loader))
        eeg_channels = sample_batch['eeg_data'].shape[1]
        print(f"\n📊 EEG channels detected: {eeg_channels}")
        
        # Create model for Crell (64 channels)
        model = NTViTEEGToFMRI(eeg_channels=eeg_channels).to(device)
        
        # Conservative optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-5,  # Very conservative
            weight_decay=1e-6
        )
        
        print(f"\n🚀 Testing training for 3 epochs...")
        
        model.train()
        for epoch in range(3):
            print(f"\nEpoch {epoch+1}/3")
            
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 5:  # Test only first 5 batches per epoch
                    break
                
                # Move to device
                eeg_data = batch['eeg_data'].to(device, non_blocking=True)
                target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
                
                # Check data
                if not torch.isfinite(eeg_data).all() or not torch.isfinite(target_fmri).all():
                    print(f"  ⚠️  Non-finite input data in batch {batch_idx}")
                    continue
                
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    outputs = model(eeg_data, target_fmri)
                    
                    # Calculate loss
                    recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                    domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                    total_loss = recon_loss + 0.001 * domain_loss  # Very low domain weight
                    
                    # Check loss
                    if not torch.isfinite(total_loss):
                        print(f"  ⚠️  Non-finite loss in batch {batch_idx}")
                        continue
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                    print(f"  ✅ Batch {batch_idx}: Loss = {total_loss.item():.6f}")
                    
                except Exception as e:
                    print(f"  ❌ Batch {batch_idx}: Error = {e}")
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"  📊 Epoch {epoch+1} avg loss: {avg_loss:.6f}")
            else:
                print(f"  ❌ No successful batches in epoch {epoch+1}")
        
        print(f"\n✅ Crell training test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Crell training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("🧠 Simple Crell Training Test")
    print("=" * 50)
    print("📋 Using existing CrellDataLoader + outlier detection")
    
    # Configuration
    datasets_dir = "datasets"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  Dataset: Crell (existing loader)")
    print(f"  Device: {device}")
    print(f"  Approach: Test training feasibility")
    
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
    
    # Test training
    success = test_crell_training(datasets_dir, device)
    
    if success:
        print(f"\n🎉 Crell training test successful!")
        print(f"💡 Ready for full Crell training implementation")
    else:
        print(f"\n❌ Crell training test failed")
        print(f"💡 Need to debug Crell data loading issues")

if __name__ == "__main__":
    main()
