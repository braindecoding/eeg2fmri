#!/usr/bin/env python3
"""
NT-ViT Robust Training - Outlier Detection & Stability
======================================================

Based on analysis: model architecture and small-scale training are healthy.
Issue likely caused by outlier samples or accumulation effects in large dataset.

This script implements robust training with:
1. Outlier detection and filtering
2. Loss stability monitoring
3. Automatic recovery mechanisms
4. Conservative training settings
"""

import torch
import torch.nn as nn
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
    MindBigDataLoader,
    EEGFMRIDataset
)

def detect_outlier_samples(samples, threshold_std=3.0):
    """Detect and filter outlier samples that might cause infinity loss"""
    
    print(f"🔍 Detecting Outlier Samples (threshold: {threshold_std} std)...")
    
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
            'range': np.max(eeg_data) - np.min(eeg_data)
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
    
    print(f"📊 Global Statistics:")
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
            print(f"  ⚠️  Outlier {stats['index']}: {', '.join(reasons)}")
    
    # Filter out outliers
    filtered_samples = [samples[i] for i in range(len(samples)) if i not in outlier_indices]
    
    print(f"📊 Outlier Detection Results:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Outlier samples: {len(outlier_indices)}")
    print(f"  Filtered samples: {len(filtered_samples)}")
    print(f"  Retention rate: {len(filtered_samples)/len(samples)*100:.1f}%")
    
    return filtered_samples, outlier_indices

def create_robust_data_loaders(datasets_dir: str, batch_size: int = 4):
    """Create robust data loaders with outlier filtering and normalization"""
    
    print(f"🧠 Creating Robust Data Loaders...")
    
    datasets_path = Path(datasets_dir)
    
    # Load MindBigData with balanced distribution
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=1200,
        balanced_per_label=True
    )
    
    samples = mindbig_loader.samples
    print(f"✓ Loaded {len(samples)} raw samples")
    
    if len(samples) == 0:
        raise ValueError("No samples loaded!")
    
    # Detect and filter outliers
    filtered_samples, outlier_indices = detect_outlier_samples(samples, threshold_std=2.5)
    
    if len(filtered_samples) < 100:
        print(f"⚠️  Too few samples after filtering ({len(filtered_samples)}), using less strict filtering")
        filtered_samples, outlier_indices = detect_outlier_samples(samples, threshold_std=3.5)
    
    # Robust normalization
    print(f"🔧 Applying robust normalization...")
    
    # Calculate robust statistics (using median and MAD instead of mean/std)
    all_eeg_data = []
    for sample in filtered_samples:
        all_eeg_data.append(sample['eeg_data'].flatten())
    
    all_eeg_flat = np.concatenate(all_eeg_data)
    
    # Use median and MAD for robust normalization
    eeg_median = np.median(all_eeg_flat)
    eeg_mad = np.median(np.abs(all_eeg_flat - eeg_median))
    
    print(f"📊 Robust statistics: median={eeg_median:.4f}, MAD={eeg_mad:.4f}")
    
    # Apply robust normalization
    for sample in filtered_samples:
        eeg_data = sample['eeg_data']
        # Robust z-score using median and MAD
        eeg_normalized = (eeg_data - eeg_median) / (1.4826 * eeg_mad + 1e-8)
        # Conservative clipping
        eeg_normalized = np.clip(eeg_normalized, -3.0, 3.0)
        sample['eeg_data'] = eeg_normalized
    
    # Check label distribution after filtering
    labels = [sample['label'] for sample in filtered_samples]
    print(f"📊 Label distribution after filtering:")
    for digit in range(10):
        count = labels.count(digit)
        print(f"  Digit {digit}: {count} samples")
    
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
        drop_last=True  # Drop incomplete batches for stability
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"✓ Created robust data loaders:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Outliers removed: {len(outlier_indices)}")
    
    return train_loader, val_loader, outlier_indices

def robust_training_step(model, batch, optimizer, device, step_info):
    """Robust training step with stability monitoring"""
    
    # Move to device
    eeg_data = batch['eeg_data'].to(device, non_blocking=True)
    target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
    
    # Pre-check input data
    if not torch.isfinite(eeg_data).all() or not torch.isfinite(target_fmri).all():
        return None, "Non-finite input data"
    
    # Zero gradients
    optimizer.zero_grad()
    
    try:
        # Forward pass
        outputs = model(eeg_data, target_fmri)
        
        # Calculate losses
        recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
        domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
        
        # Conservative domain loss weight
        total_loss = recon_loss + 0.001 * domain_loss  # Very low domain weight
        
        # Check loss stability
        if not torch.isfinite(total_loss):
            return None, f"Non-finite loss: recon={recon_loss.item()}, domain={domain_loss.item()}"
        
        # Check for extreme loss values
        if total_loss.item() > 100.0:
            return None, f"Extreme loss value: {total_loss.item()}"
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0
        nan_grads = 0
        inf_grads = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                if torch.isnan(param_grad_norm):
                    nan_grads += 1
                elif torch.isinf(param_grad_norm):
                    inf_grads += 1
                else:
                    grad_norm += param_grad_norm.item() ** 2
        
        if nan_grads > 0 or inf_grads > 0:
            return None, f"Bad gradients: {nan_grads} NaN, {inf_grads} Inf"
        
        grad_norm = grad_norm ** 0.5
        
        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        # Optimizer step
        optimizer.step()
        
        # Return success
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'domain_loss': domain_loss.item(),
            'grad_norm': grad_norm
        }, None
        
    except Exception as e:
        return None, f"Training step error: {e}"

def train_robust_model(datasets_dir: str,
                      output_dir: str = "ntvit_robust_outputs",
                      num_epochs: int = 30,
                      batch_size: int = 4,
                      device: str = 'cuda'):
    """Robust training with outlier detection and stability monitoring"""
    
    print("🧠 NT-ViT Robust Training Pipeline")
    print("=" * 60)
    print("🛡️  Features:")
    print("  • Outlier detection and filtering")
    print("  • Robust normalization (median + MAD)")
    print("  • Loss stability monitoring")
    print("  • Automatic recovery mechanisms")
    print("  • Conservative training settings")
    
    # GPU setup
    if device == 'cuda' and torch.cuda.is_available():
        print(f"\n🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.7)  # Conservative memory usage
    else:
        device = 'cpu'
        print("⚠️  Using CPU")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load robust data
    train_loader, val_loader, outlier_indices = create_robust_data_loaders(
        datasets_dir, batch_size=batch_size
    )
    
    # Create model
    sample_batch = next(iter(train_loader))
    eeg_channels = sample_batch['eeg_data'].shape[1]
    
    model = NTViTEEGToFMRI(eeg_channels=eeg_channels).to(device)
    
    # Conservative optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-6,  # Very conservative learning rate
        weight_decay=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"\n🚀 Starting robust training for {num_epochs} epochs...")
    print(f"🛡️  Robust settings:")
    print(f"  - Learning rate: 5e-6 (very conservative)")
    print(f"  - Gradient clipping: 0.1 (aggressive)")
    print(f"  - Domain loss weight: 0.001 (minimal)")
    print(f"  - Memory fraction: 0.7 (conservative)")
    print(f"  - Outliers removed: {len(outlier_indices)}")
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    failed_steps = 0
    total_steps = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        epoch_train_losses = []
        epoch_failed_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            total_steps += 1
            
            step_result, error_msg = robust_training_step(
                model, batch, optimizer, device, 
                {'epoch': epoch, 'batch': batch_idx}
            )
            
            if step_result is None:
                epoch_failed_steps += 1
                failed_steps += 1
                print(f"  ⚠️  Step {batch_idx}: {error_msg}")
                
                # If too many failures, stop epoch
                if epoch_failed_steps > len(train_loader) * 0.5:
                    print(f"  ❌ Too many failed steps ({epoch_failed_steps}), stopping epoch")
                    break
            else:
                epoch_train_losses.append(step_result['total_loss'])
                
                if batch_idx % 50 == 0:
                    print(f"  ✅ Step {batch_idx}: Loss={step_result['total_loss']:.6f}, "
                          f"Grad={step_result['grad_norm']:.6f}")
        
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
                        pass  # Skip problematic validation batches
        
        # Calculate metrics
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        
        if np.isfinite(avg_train_loss) and np.isfinite(avg_val_loss):
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            print(f"  📊 Train Loss: {avg_train_loss:.6f}")
            print(f"  📊 Val Loss: {avg_val_loss:.6f}")
            print(f"  📊 Failed Steps: {epoch_failed_steps}/{len(train_loader)}")
            print(f"  📊 Success Rate: {(len(train_loader)-epoch_failed_steps)/len(train_loader)*100:.1f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'outlier_indices': outlier_indices,
                    'failed_steps': failed_steps,
                    'total_steps': total_steps
                }, output_path / "best_robust_model.pth")
                print(f"  ✅ Saved best model")
        else:
            print(f"  ❌ Non-finite losses, skipping epoch")
        
        # Memory cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'outlier_indices': outlier_indices,
        'failed_steps': failed_steps,
        'total_steps': total_steps,
        'success_rate': (total_steps - failed_steps) / total_steps if total_steps > 0 else 0
    }, output_path / "final_robust_model.pth")
    
    # Save training report
    report = {
        'training_completed': True,
        'best_val_loss': best_val_loss,
        'total_epochs': num_epochs,
        'outliers_removed': len(outlier_indices),
        'failed_steps': failed_steps,
        'total_steps': total_steps,
        'success_rate': (total_steps - failed_steps) / total_steps if total_steps > 0 else 0,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'robust_features': [
            'Outlier detection and filtering',
            'Robust normalization (median + MAD)',
            'Loss stability monitoring',
            'Conservative learning rate (5e-6)',
            'Aggressive gradient clipping (0.1)',
            'Minimal domain loss weight (0.001)'
        ]
    }
    
    with open(output_path / "robust_training_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Robust training complete!")
    print(f"📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Success Rate: {report['success_rate']*100:.1f}%")
    print(f"  Outliers Removed: {len(outlier_indices)}")
    print(f"  Models saved to: {output_path}")
    
    return model

def main():
    """Main function"""
    
    print("🛡️  NT-ViT Robust Training - Outlier Detection & Stability")
    print("=" * 70)
    
    # Configuration
    datasets_dir = "datasets"
    output_dir = "ntvit_robust_outputs"
    num_epochs = 30
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📋 Configuration:")
    print(f"  Dataset: MindBigData (with outlier filtering)")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output_dir}")
    
    try:
        model = train_robust_model(
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )
        
        print(f"\n🎉 Robust training completed successfully!")
        
    except Exception as e:
        print(f"❌ Robust training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
