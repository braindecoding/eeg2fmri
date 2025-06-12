#!/usr/bin/env python3
"""
NT-ViT Full GPU-Optimized Training - Maximum GPU Utilization
==========================================================

Full GPU acceleration for:
1. Outlier detection using CUDA tensors
2. Robust normalization on GPU
3. Batched statistical operations
4. Memory-efficient GPU operations
5. Parallel processing on GPU
6. Minimal CPU-GPU transfers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from train_ntvit import (
    NTViTEEGToFMRI, 
    MindBigDataLoader,
    EEGFMRIDataset
)

class GPUOptimizedDataProcessor:
    """GPU-based data processing pipeline"""
    
    def __init__(self, device='cuda'):
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available!")
        
        print(f"🚀 GPU Data Processor initialized on {torch.cuda.get_device_name()}")
        
    def gpu_outlier_detection(self, eeg_tensor, threshold_std=3.0):
        """GPU-accelerated outlier detection using CUDA tensors"""
        
        print(f"🔍 GPU Outlier Detection (threshold: {threshold_std} std)...")
        
        # Ensure tensor is on GPU
        if eeg_tensor.device.type != 'cuda':
            eeg_tensor = eeg_tensor.to(self.device)
        
        n_samples = eeg_tensor.shape[0]
        
        # GPU-accelerated statistics calculation
        print("📊 Computing GPU statistics...")
        
        # Flatten each sample for statistics
        flat_samples = eeg_tensor.view(n_samples, -1)  # [n_samples, features]
        
        # Vectorized statistics on GPU
        sample_means = torch.mean(flat_samples, dim=1)  # [n_samples]
        sample_stds = torch.std(flat_samples, dim=1)    # [n_samples]
        sample_mins = torch.min(flat_samples, dim=1)[0] # [n_samples]
        sample_maxs = torch.max(flat_samples, dim=1)[0] # [n_samples]
        sample_ranges = sample_maxs - sample_mins
        
        # Global statistics
        global_mean_mean = torch.mean(sample_means)
        global_mean_std = torch.std(sample_means)
        global_std_mean = torch.mean(sample_stds)
        global_std_std = torch.std(sample_stds)
        global_range_mean = torch.mean(sample_ranges)
        global_range_std = torch.std(sample_ranges)
        
        print(f"📊 Global Statistics (GPU):")
        print(f"  Mean: {global_mean_mean:.4f} ± {global_mean_std:.4f}")
        print(f"  Std: {global_std_mean:.4f} ± {global_std_std:.4f}")
        print(f"  Range: {global_range_mean:.4f} ± {global_range_std:.4f}")
        
        # Vectorized outlier detection on GPU
        outlier_mask = (
            (torch.abs(sample_means - global_mean_mean) > threshold_std * global_mean_std) |
            (torch.abs(sample_stds - global_std_mean) > threshold_std * global_std_std) |
            (torch.abs(sample_ranges - global_range_mean) > threshold_std * global_range_std) |
            (torch.abs(sample_mins) > 1e4) |
            (torch.abs(sample_maxs) > 1e4) |
            (sample_stds < 1e-6)
        )
        
        # Get valid indices
        valid_mask = ~outlier_mask
        outlier_indices = torch.where(outlier_mask)[0].cpu().tolist()
        
        print(f"📊 GPU Outlier Detection Results:")
        print(f"  Original samples: {n_samples}")
        print(f"  Outlier samples: {len(outlier_indices)}")
        print(f"  Valid samples: {torch.sum(valid_mask).item()}")
        print(f"  Retention rate: {torch.sum(valid_mask).item()/n_samples*100:.1f}%")
        
        return valid_mask, outlier_indices
    
    def gpu_robust_normalization(self, eeg_tensor, sample_fraction=0.1):
        """GPU-accelerated robust normalization"""
        
        print(f"🔧 GPU Robust Normalization (sampling {sample_fraction*100:.1f}%)...")
        
        # Ensure tensor is on GPU
        if eeg_tensor.device.type != 'cuda':
            eeg_tensor = eeg_tensor.to(self.device)
        
        n_samples = eeg_tensor.shape[0]
        
        # Sample indices for statistics
        sample_size = max(1, int(n_samples * sample_fraction))
        sample_indices = torch.randperm(n_samples, device=self.device)[:sample_size]
        
        # Sample data on GPU
        sampled_data = eeg_tensor[sample_indices].flatten()
        
        # GPU-accelerated robust statistics using quantiles
        eeg_median = torch.quantile(sampled_data, 0.5)
        eeg_mad = torch.quantile(torch.abs(sampled_data - eeg_median), 0.5)
        
        print(f"📊 GPU Robust statistics: median={eeg_median:.4f}, MAD={eeg_mad:.4f}")
        
        # In-place normalization on GPU
        eeg_tensor.sub_(eeg_median)
        eeg_tensor.div_(1.4826 * eeg_mad + 1e-8)
        
        # Conservative clipping on GPU
        torch.clamp_(eeg_tensor, -3.0, 3.0)
        
        return eeg_tensor
    
    def create_gpu_tensor_dataset(self, samples):
        """Convert samples to GPU tensor dataset"""
        
        print("🔄 Converting to GPU tensor dataset...")
        
        # Extract data
        eeg_data_list = []
        fmri_data_list = []
        labels_list = []
        
        for sample in samples:
            eeg_data_list.append(torch.from_numpy(sample['eeg_data']).float())
            fmri_data_list.append(torch.from_numpy(sample['translated_fmri_target']).float())
            labels_list.append(sample['label'])
        
        # Stack into tensors
        eeg_tensor = torch.stack(eeg_data_list)
        fmri_tensor = torch.stack(fmri_data_list)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        print(f"📊 Tensor shapes: EEG={eeg_tensor.shape}, fMRI={fmri_tensor.shape}, Labels={labels_tensor.shape}")
        
        return eeg_tensor, fmri_tensor, labels_tensor

def create_gpu_optimized_data_loaders(datasets_dir: str, batch_size: int = 4, device='cuda'):
    """Create fully GPU-optimized data loaders"""
    
    print(f"🧠 Creating GPU-Optimized Data Loaders...")
    
    # Initialize GPU processor
    gpu_processor = GPUOptimizedDataProcessor(device)
    
    datasets_path = Path(datasets_dir)
    
    # Load raw data
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=1000,
        balanced_per_label=True
    )
    
    samples = mindbig_loader.samples
    print(f"✓ Loaded {len(samples)} raw samples")
    
    if len(samples) == 0:
        raise ValueError("No samples loaded!")
    
    # Convert to GPU tensors
    eeg_tensor, fmri_tensor, labels_tensor = gpu_processor.create_gpu_tensor_dataset(samples)
    
    # Move to GPU
    eeg_tensor = eeg_tensor.to(device)
    fmri_tensor = fmri_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    
    # GPU outlier detection
    valid_mask, outlier_indices = gpu_processor.gpu_outlier_detection(eeg_tensor, threshold_std=2.5)
    
    # Filter tensors
    eeg_filtered = eeg_tensor[valid_mask]
    fmri_filtered = fmri_tensor[valid_mask]
    labels_filtered = labels_tensor[valid_mask]
    
    print(f"📊 After filtering: {eeg_filtered.shape[0]} samples")
    
    # GPU robust normalization
    eeg_normalized = gpu_processor.gpu_robust_normalization(eeg_filtered, sample_fraction=0.2)
    
    # Check label distribution on GPU
    unique_labels, counts = torch.unique(labels_filtered, return_counts=True)
    print(f"📊 Label distribution after GPU filtering:")
    for label, count in zip(unique_labels.cpu(), counts.cpu()):
        print(f"  Digit {label}: {count} samples")
    
    # Split data on GPU
    n_samples = eeg_normalized.shape[0]
    split_idx = int(0.8 * n_samples)
    
    # Create train/val splits
    train_eeg = eeg_normalized[:split_idx]
    train_fmri = fmri_filtered[:split_idx]
    train_labels = labels_filtered[:split_idx]
    
    val_eeg = eeg_normalized[split_idx:]
    val_fmri = fmri_filtered[split_idx:]
    val_labels = labels_filtered[split_idx:]
    
    # Create tensor datasets (data stays on GPU)
    train_dataset = TensorDataset(train_eeg, train_fmri, train_labels)
    val_dataset = TensorDataset(val_eeg, val_fmri, val_labels)
    
    # Create data loaders with GPU-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,  # Data already on GPU
        drop_last=True,
        num_workers=0,  # No multiprocessing needed
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        persistent_workers=False
    )
    
    print(f"✓ Created GPU-optimized data loaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  All data on GPU: {device}")
    
    return train_loader, val_loader, outlier_indices

def gpu_optimized_training_step(model, batch, optimizer, device, use_amp=True):
    """GPU-optimized training step with mixed precision"""
    
    # Data is already on GPU from TensorDataset
    eeg_data, target_fmri, labels = batch
    
    # Ensure on correct device
    eeg_data = eeg_data.to(device, non_blocking=True)
    target_fmri = target_fmri.to(device, non_blocking=True)
    
    # Quick GPU-based finite check
    if not (torch.isfinite(eeg_data).all() and torch.isfinite(target_fmri).all()):
        return None, "Non-finite input"
    
    optimizer.zero_grad()
    
    # Use mixed precision for faster training
    if use_amp and device != 'cpu':
        with torch.cuda.amp.autocast():
            outputs = model(eeg_data, target_fmri)
            recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
            domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
            total_loss = recon_loss + 0.001 * domain_loss
    else:
        outputs = model(eeg_data, target_fmri)
        recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
        domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
        total_loss = recon_loss + 0.001 * domain_loss
    
    # GPU-based loss validation
    if not torch.isfinite(total_loss) or total_loss.item() > 100.0:
        return None, f"Bad loss: {total_loss.item()}"
    
    # Backward pass
    if use_amp and device != 'cpu':
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'domain_loss': domain_loss.item()
    }, None

def train_gpu_optimized_model(datasets_dir: str,
                             output_dir: str = "ntvit_gpu_optimized",
                             num_epochs: int = 25,
                             batch_size: int = 8,  # Larger batch for GPU efficiency
                             device: str = 'cuda'):
    """Full GPU-optimized training pipeline"""
    
    print("🚀 NT-ViT Full GPU-Optimized Training Pipeline")
    print("=" * 70)
    print("⚡ GPU Optimizations:")
    print("  • CUDA tensor operations for outlier detection")
    print("  • GPU-based robust normalization")
    print("  • All data kept on GPU memory")
    print("  • Mixed precision training (AMP)")
    print("  • Vectorized GPU operations")
    print("  • Minimal CPU-GPU transfers")
    
    # Verify GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    device = torch.device(device)
    print(f"\n🚀 Using GPU: {torch.cuda.get_device_name()}")
    
    # Optimize GPU settings
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.cuda.empty_cache()
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create GPU-optimized data loaders
    train_loader, val_loader, outlier_indices = create_gpu_optimized_data_loaders(
        datasets_dir, batch_size=batch_size, device=device
    )
    
    # Create model
    sample_batch = next(iter(train_loader))
    eeg_channels = sample_batch[0].shape[1]  # TensorDataset format
    
    model = NTViTEEGToFMRI(eeg_channels=eeg_channels).to(device)
    
    # Enable mixed precision training
    model = torch.compile(model, mode="max-autotune")  # PyTorch 2.0+ optimization
    
    # Optimized optimizer for GPU
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # Higher LR for larger batches
        weight_decay=1e-5,
        fused=True  # Fused AdamW for better GPU performance
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    print(f"\n🚀 Starting GPU-optimized training for {num_epochs} epochs...")
    print(f"🔧 GPU Settings:")
    print(f"  - Batch size: {batch_size} (optimized for GPU)")
    print(f"  - Mixed precision: Enabled")
    print(f"  - Fused optimizer: Enabled")
    print(f"  - Compiled model: Enabled")
    print(f"  - GPU memory: 90% allocated")
    print(f"  - Data on GPU: All operations")
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            result, error = gpu_optimized_training_step(
                model, batch, optimizer, device, use_amp=True
            )
            
            if result is None:
                if batch_idx % 50 == 0:
                    print(f"  ⚠️  Step {batch_idx}: {error}")
            else:
                epoch_losses.append(result['total_loss'])
                
                if batch_idx % 30 == 0:
                    print(f"  ✅ Step {batch_idx}: Loss={result['total_loss']:.6f}")
            
            # Update scheduler
            scheduler.step()
        
        # Validation phase
        model.eval()
        val_losses_epoch = []
        
        with torch.no_grad():
            for batch in val_loader:
                eeg_data, target_fmri, labels = batch
                eeg_data = eeg_data.to(device, non_blocking=True)
                target_fmri = target_fmri.to(device, non_blocking=True)
                
                try:
                    with torch.cuda.amp.autocast():
                        outputs = model(eeg_data, target_fmri)
                        recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                    
                    if torch.isfinite(recon_loss):
                        val_losses_epoch.append(recon_loss.item())
                except:
                    pass
        
        # Calculate metrics
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_val_loss = np.mean(val_losses_epoch) if val_losses_epoch else float('inf')
        
        if np.isfinite(avg_train_loss) and np.isfinite(avg_val_loss):
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"  📊 Train Loss: {avg_train_loss:.6f}")
            print(f"  📊 Val Loss: {avg_val_loss:.6f}")
            print(f"  📊 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            print(f"  📊 Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'scaler_state_dict': scaler.state_dict()
                }, output_path / "best_gpu_model.pth")
                print(f"  ✅ Saved best GPU model")
    
    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'device': str(device),
            'mixed_precision': True,
            'fused_optimizer': True
        }
    }, output_path / "final_gpu_model.pth")
    
    print(f"\n✅ GPU-optimized training complete!")
    print(f"📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  GPU Memory Used: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
    print(f"  Models saved to: {output_path}")
    
    return model

def main():
    """Main function"""
    
    print("⚡ NT-ViT Full GPU-Optimized Training")
    print("=" * 60)
    
    # Set environment for single GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # Verify GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    # Configuration
    datasets_dir = "datasets"
    output_dir = "ntvit_gpu_optimized"
    num_epochs = 25
    batch_size = 8  # Larger batch for GPU efficiency
    device = 'cuda'
    
    print(f"📋 Configuration:")
    print(f"  Dataset: MindBigData (GPU-optimized)")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Mixed precision: Enabled")
    
    try:
        model = train_gpu_optimized_model(
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )
        
        print(f"\n🎉 Full GPU-optimized training completed!")
        
    except Exception as e:
        print(f"❌ GPU training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
