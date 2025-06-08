#!/usr/bin/env python3
"""
NT-ViT Training Script - MindBigData Only (FIXED VERSION)
=========================================================

Fixed issues:
- Lower learning rate to prevent gradient explosion
- Better gradient clipping
- Data normalization
- Loss stability checks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import math
import torchaudio.transforms as T
from pathlib import Path
import json
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Import NT-ViT components from main script
import sys
sys.path.append('.')
from train_ntvit import (
    NTViTEEGToFMRI, 
    MindBigDataLoader,
    EEGFMRIDataset
)

def create_mindbigdata_loaders_fixed(datasets_dir: str, batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create MindBigData data loaders with balanced sampling and better normalization"""
    
    datasets_path = Path(datasets_dir)
    
    print(f"🧠 Loading MindBigData with balanced distribution...")
    
    # Load MindBigData with balanced distribution (1200 samples, 120 per digit)
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=1200,
        balanced_per_label=True
    )
    
    samples = mindbig_loader.samples
    print(f"✓ Loaded {len(samples)} MindBigData samples")
    
    if len(samples) == 0:
        raise ValueError("No MindBigData samples loaded!")
    
    # Check distribution
    labels = [sample['label'] for sample in samples]
    print(f"📊 Label distribution:")
    for digit in range(10):
        count = labels.count(digit)
        print(f"  Digit {digit}: {count} samples")
    
    # Normalize EEG data to prevent numerical instability
    print(f"🔧 Normalizing EEG data...")
    all_eeg_data = []
    for sample in samples:
        eeg_data = sample['eeg_data']
        if not np.isfinite(eeg_data).all():
            print(f"⚠️  Found non-finite values in EEG data, replacing with zeros")
            eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
        all_eeg_data.append(eeg_data.flatten())
    
    # Calculate global statistics for normalization
    all_eeg_flat = np.concatenate(all_eeg_data)
    eeg_mean = np.mean(all_eeg_flat)
    eeg_std = np.std(all_eeg_flat)
    
    print(f"📊 EEG statistics: mean={eeg_mean:.4f}, std={eeg_std:.4f}")
    
    # Apply normalization to all samples
    for sample in samples:
        eeg_data = sample['eeg_data']
        # Z-score normalization
        eeg_normalized = (eeg_data - eeg_mean) / (eeg_std + 1e-8)
        # Clip extreme values
        eeg_normalized = np.clip(eeg_normalized, -5.0, 5.0)
        sample['eeg_data'] = eeg_normalized
    
    # Split samples (80% train, 20% val)
    train_samples = samples[:int(0.8 * len(samples))]
    val_samples = samples[int(0.8 * len(samples)):]
    
    # Create datasets
    train_dataset = EEGFMRIDataset(train_samples)
    val_dataset = EEGFMRIDataset(val_samples)
    
    # Create data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Created data loaders:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    
    return train_loader, val_loader

def train_mindbigdata_model_fixed(datasets_dir: str,
                                 output_dir: str = "ntvit_mindbigdata_outputs_fixed",
                                 num_epochs: int = 50,
                                 batch_size: int = 4,
                                 device: str = 'cuda'):
    """Training pipeline for MindBigData only with stability fixes"""
    
    print("🧠 NT-ViT MindBigData Training Pipeline (FIXED)")
    print("=" * 50)
    
    # GPU optimization
    if device == 'cuda' and torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    else:
        device = 'cpu'
        print("⚠️  Using CPU (GPU not available)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data with fixes
    train_loader, val_loader = create_mindbigdata_loaders_fixed(datasets_dir, batch_size=batch_size)
    
    # Determine channel count from loaded data
    sample_batch = next(iter(train_loader))
    eeg_channels = sample_batch['eeg_data'].shape[1]
    print(f"📊 EEG channels detected: {eeg_channels}")
    
    # Create model
    model = NTViTEEGToFMRI(eeg_channels=eeg_channels).to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    print(f"✓ Applied proper weight initialization")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer with MUCH lower learning rate and better settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-5,  # Much lower learning rate
        weight_decay=1e-5,  # Lower weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < 5:  # Warmup for first 5 epochs
            return (epoch + 1) / 5
        else:
            return 0.5 ** ((epoch - 5) // 10)  # Decay every 10 epochs
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"🔧 Stability fixes applied:")
    print(f"  - Lower learning rate: 1e-5")
    print(f"  - Better weight initialization")
    print(f"  - Data normalization and clipping")
    print(f"  - Gradient clipping: 0.5")
    print(f"  - Loss stability checks")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            eeg_data = batch['eeg_data'].to(device, non_blocking=True)
            target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
            
            # Check for NaN/Inf in input data
            if not torch.isfinite(eeg_data).all() or not torch.isfinite(target_fmri).all():
                print(f"  Warning: Non-finite input data in batch {batch_idx}, skipping...")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(eeg_data, target_fmri)
                    recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                    domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                    total_loss = recon_loss + 0.01 * domain_loss  # Much lower domain loss weight
                
                # Check for NaN/Inf loss
                if not torch.isfinite(total_loss):
                    print(f"  Warning: Non-finite loss in batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass with scaling
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                
                # Gradient clipping with lower threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(eeg_data, target_fmri)
                recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                total_loss = recon_loss + 0.01 * domain_loss
                
                # Check for NaN/Inf loss
                if not torch.isfinite(total_loss):
                    print(f"  Warning: Non-finite loss in batch {batch_idx}, skipping...")
                    continue
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
            
            epoch_train_losses.append(total_loss.item())
            
            # Memory cleanup
            if device == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                eeg_data = batch['eeg_data'].to(device, non_blocking=True)
                target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
                
                # Check for NaN/Inf in input data
                if not torch.isfinite(eeg_data).all() or not torch.isfinite(target_fmri).all():
                    continue
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(eeg_data, target_fmri)
                        recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                        domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                        total_loss = recon_loss + 0.01 * domain_loss
                else:
                    outputs = model(eeg_data, target_fmri)
                    recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
                    domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                    total_loss = recon_loss + 0.01 * domain_loss
                
                if torch.isfinite(total_loss):
                    epoch_val_losses.append(total_loss.item())
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        
        # Check for stability
        if not np.isfinite(avg_train_loss) or not np.isfinite(avg_val_loss):
            print(f"  ⚠️  Non-finite losses detected, stopping training")
            break
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        if device == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'eeg_channels': eeg_channels
            }, output_path / "best_model.pth")
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'eeg_channels': eeg_channels
            }, output_path / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"  ✓ Saved checkpoint")
        
        # Memory cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Final model save
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'eeg_channels': eeg_channels,
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'best_val_loss': best_val_loss
    }, output_path / "final_model.pth")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'eeg_channels': eeg_channels,
        'total_samples': len(train_loader.dataset) + len(val_loader.dataset),
        'fixes_applied': [
            'Lower learning rate (1e-5)',
            'Better weight initialization',
            'Data normalization and clipping',
            'Gradient clipping (0.5)',
            'Loss stability checks',
            'Lower domain loss weight (0.01)'
        ]
    }
    
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Final Train Loss: {train_losses[-1] if train_losses else 'N/A':.6f}")
    print(f"  Final Val Loss: {val_losses[-1] if val_losses else 'N/A':.6f}")
    print(f"  Models saved to: {output_path}")
    
    return model

def main():
    """Main function for MindBigData training (FIXED)"""
    
    print("🧠 NT-ViT MindBigData Training (FIXED VERSION)")
    print("=" * 50)
    
    # Configuration
    datasets_dir = "datasets"
    output_dir = "ntvit_mindbigdata_outputs_fixed"
    num_epochs = 30
    batch_size = 4  # Adjust based on GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📋 Configuration:")
    print(f"  Dataset: MindBigData (1174 balanced samples)")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output_dir}")
    print(f"  Learning rate: 1e-5 (FIXED)")
    
    # Check if datasets exist
    datasets_path = Path(datasets_dir)
    required_files = ["EP1.01.txt", "MindbigdataStimuli"]
    
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
        # Train model
        print(f"\n🚀 Starting MindBigData training (FIXED)...")
        model = train_mindbigdata_model_fixed(
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )
        
        print(f"\n🎉 MindBigData training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
