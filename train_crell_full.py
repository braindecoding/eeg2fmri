#!/usr/bin/env python3
"""
Crell Dataset Full Training - 64 Channels, 500Hz, Letters
==========================================================

Full training implementation for Crell dataset based on successful test.
Uses the same robust approach that worked for MindBigData.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
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

def detect_crell_outliers(samples, threshold_std=3.0):
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
    
    # Load Crell with existing working loader - get all samples
    print(f"📊 Loading full Crell dataset...")
    crell_loader = CrellDataLoader(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=1000  # Use same limit as successful test
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
    
    if len(filtered_samples) < 50:
        print(f"⚠️  Too few samples after filtering ({len(filtered_samples)}), using less strict filtering")
        filtered_samples, outlier_indices = detect_crell_outliers(samples, threshold_std=4.0)
    
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

def train_crell_model(train_loader, val_loader, device, output_dir, epochs=30):
    """Train Crell model with robust settings"""
    
    print(f"\n🚀 Starting Crell full training for {epochs} epochs...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Determine channel count from loaded data
    sample_batch = next(iter(train_loader))
    eeg_channels = sample_batch['eeg_data'].shape[1]
    print(f"📊 EEG channels detected: {eeg_channels}")
    
    # Create model for Crell (64 channels)
    model = NTViTEEGToFMRI(eeg_channels=eeg_channels).to(device)
    
    # Conservative optimizer settings (same as successful MindBigData)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-6,  # Very conservative learning rate
        weight_decay=1e-6
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.8, 
        patience=3,
        min_lr=1e-7
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    failed_steps = 0
    total_steps = 0
    
    print(f"🛡️  Robust Crell settings:")
    print(f"  - Learning rate: 5e-6 (very conservative)")
    print(f"  - Gradient clipping: 0.1 (aggressive)")
    print(f"  - Domain loss weight: 0.001 (minimal)")
    print(f"  - EEG channels: {eeg_channels}")
    print(f"  - Batch size: {train_loader.batch_size}")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        epoch_train_losses = []
        epoch_failed_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            total_steps += 1
            
            # Move to device
            eeg_data = batch['eeg_data'].to(device, non_blocking=True)
            target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
            
            # Check data validity
            if not torch.isfinite(eeg_data).all() or not torch.isfinite(target_fmri).all():
                print(f"  ⚠️  Non-finite input data in batch {batch_idx}")
                failed_steps += 1
                epoch_failed_steps += 1
                continue
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(eeg_data, target_fmri)
                
                # Calculate loss
                recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                total_loss = recon_loss + 0.001 * domain_loss  # Very low domain weight
                
                # Check loss validity
                if not torch.isfinite(total_loss):
                    print(f"  ⚠️  Non-finite loss in batch {batch_idx}")
                    failed_steps += 1
                    epoch_failed_steps += 1
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping (aggressive)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                # Optimizer step
                optimizer.step()
                
                epoch_train_losses.append(total_loss.item())
                
                # Log progress
                if batch_idx % 50 == 0:
                    print(f"  ✅ Step {batch_idx}: Loss={total_loss.item():.6f}, Grad={grad_norm:.6f}")
                
            except Exception as e:
                print(f"  ❌ Batch {batch_idx}: Error = {e}")
                failed_steps += 1
                epoch_failed_steps += 1
        
        # Calculate epoch training loss
        if epoch_train_losses:
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
        else:
            print(f"  ❌ No successful training batches in epoch {epoch+1}")
            continue
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                eeg_data = batch['eeg_data'].to(device, non_blocking=True)
                target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
                
                if torch.isfinite(eeg_data).all() and torch.isfinite(target_fmri).all():
                    try:
                        outputs = model(eeg_data, target_fmri)
                        recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                        domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                        total_loss = recon_loss + 0.001 * domain_loss
                        
                        if torch.isfinite(total_loss):
                            epoch_val_losses.append(total_loss.item())
                    except:
                        pass
        
        # Calculate epoch validation loss
        if epoch_val_losses:
            avg_val_loss = np.mean(epoch_val_losses)
            val_losses.append(avg_val_loss)
        else:
            avg_val_loss = float('inf')
            val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, output_path / 'best_crell_model.pth')
            print(f"  ✅ Saved best model")
        
        # Log epoch results
        success_rate = (len(epoch_train_losses) / len(train_loader)) * 100 if len(train_loader) > 0 else 0
        print(f"  📊 Train Loss: {avg_train_loss:.6f}")
        print(f"  📊 Val Loss: {avg_val_loss:.6f}")
        print(f"  📊 Failed Steps: {epoch_failed_steps}/{len(train_loader)}")
        print(f"  📊 Success Rate: {success_rate:.1f}%")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_failed_steps': failed_steps,
        'total_steps': total_steps
    }, output_path / 'final_crell_model.pth')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_failed_steps': failed_steps,
        'total_steps': total_steps,
        'success_rate': ((total_steps - failed_steps) / total_steps * 100) if total_steps > 0 else 0,
        'epochs_completed': len(train_losses)
    }
    
    with open(output_path / 'crell_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Crell training complete!")
    print(f"📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Success Rate: {((total_steps - failed_steps) / total_steps * 100):.1f}%")
    print(f"  Total Failed Steps: {failed_steps}/{total_steps}")
    print(f"  Models saved to: {output_dir}")
    
    return model, history

def main():
    """Main function for Crell full training"""
    
    print("🧠 Crell Dataset Full Training")
    print("=" * 60)
    print("📋 Crell Characteristics:")
    print("  • 64 EEG channels (vs 14 for MindBigData)")
    print("  • 500Hz sampling rate (vs 128Hz)")
    print("  • 2250 samples per epoch (vs 256)")
    print("  • Letter stimuli: a,d,e,f,j,n,o,s,t,v")
    print("  • Visual phases only (4.5s duration)")
    
    # Configuration
    datasets_dir = "datasets"
    output_dir = "crell_full_outputs"
    batch_size = 2  # Optimal for 64 channels + 2250 samples
    epochs = 30  # Same as successful MindBigData training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  Dataset: Crell (full training)")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output_dir}")
    
    # GPU setup
    if device == 'cuda':
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        # Conservative memory settings for larger Crell data
        torch.cuda.set_per_process_memory_fraction(0.7)
    else:
        print("⚠️  Using CPU")
    
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
        # Create data loaders
        train_loader, val_loader, outlier_indices = create_crell_data_loaders(
            datasets_dir=datasets_dir,
            batch_size=batch_size
        )
        
        # Train model
        model, history = train_crell_model(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
            epochs=epochs
        )
        
        print(f"\n🎉 Crell full training completed successfully!")
        print(f"📊 Training Summary:")
        print(f"  • Epochs completed: {history['epochs_completed']}")
        print(f"  • Best validation loss: {history['best_val_loss']:.6f}")
        print(f"  • Overall success rate: {history['success_rate']:.1f}%")
        print(f"  • Outliers removed: {len(outlier_indices)}")
        print(f"  • Output directory: {output_dir}")
        
        print(f"\n🚀 Ready for Crell fMRI generation!")
        
    except Exception as e:
        print(f"❌ Crell full training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
