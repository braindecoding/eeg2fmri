#!/usr/bin/env python3
"""
NT-ViT Training Script - Crell Only
===================================

Optimized for:
- Crell: Maximum available samples
- GPU training with memory optimization
- Separate from MindBigData to avoid memory issues
- Fixed slicing issues
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
import scipy.io
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
    EEGFMRIDataset
)

class CrellDataLoaderFixed:
    """Fixed Crell dataset loader with proper slicing"""
    
    def __init__(self, filepath: str, stimuli_dir: str, max_samples: int = None):
        self.filepath = filepath
        self.stimuli_dir = Path(stimuli_dir)
        self.max_samples = max_samples if max_samples is not None else float('inf')
        
        # Letter mapping
        self.letter_mapping = {100: 'a', 103: 'd', 104: 'e', 105: 'f', 109: 'j',
                              113: 'n', 114: 'o', 118: 's', 119: 't', 121: 'v'}
        
        self.samples = []
        self.load_data()
        
    def load_data(self):
        """Load Crell samples with fixed slicing"""
        print(f"Loading Crell data from {self.filepath}...")
        
        try:
            data = scipy.io.loadmat(self.filepath)
            print(f"  Available keys: {[k for k in data.keys() if not k.startswith('__')]}")
            
            # Find paradigm keys
            paradigm_keys = [k for k in data.keys() if 'paradigm' in k]
            print(f"  Found paradigm keys: {paradigm_keys}")
            
            for paradigm_key in paradigm_keys:
                if len(self.samples) >= self.max_samples:
                    break
                    
                round_data = data[paradigm_key][0, 0]
                
                # Extract data according to Crell specification
                eeg_data = round_data['BrainVisionRDA_data'].T  # (64, timepoints) at 500Hz
                eeg_times = round_data['BrainVisionRDA_time'].flatten()
                marker_data = round_data['ParadigmMarker_data'].flatten()
                marker_times = round_data['ParadigmMarker_time'].flatten()
                
                print(f"  Processing {paradigm_key}: {eeg_data.shape[1]} timepoints, {len(marker_data)} markers")
                
                # Extract visual epochs with fixed slicing
                visual_epochs = self.extract_visual_epochs_fixed(
                    eeg_data, eeg_times, marker_data, marker_times
                )
                
                for epoch_data, letter_label, phase_info in visual_epochs:
                    # Convert letter_label back to letter for stimulus loading
                    letter_to_char = {0: 'a', 1: 'd', 2: 'e', 3: 'f', 4: 'j',
                                    5: 'n', 6: 'o', 7: 's', 8: 't', 9: 'v'}
                    letter = letter_to_char.get(letter_label, 'a')
                    
                    # Load corresponding stimulus
                    stimulus_path = self.stimuli_dir / f"{letter}.png"
                    if stimulus_path.exists():
                        stimulus_image = self.load_stimulus_image(stimulus_path)
                        
                        self.samples.append({
                            'eeg_data': epoch_data,
                            'stimulus_code': phase_info['letter_code'],
                            'stimulus_image': stimulus_image,
                            'dataset_type': 'crell',
                            'letter': letter,
                            'phase': phase_info['phase'],
                            'duration': phase_info['duration'],
                            'paradigm_round': paradigm_key,
                            'label': letter_label
                        })
                        
                        if len(self.samples) >= self.max_samples:
                            break
                
                if len(self.samples) >= self.max_samples:
                    break
                    
        except Exception as e:
            print(f"Error loading Crell data: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"Loaded {len(self.samples)} Crell samples")
    
    def extract_visual_epochs_fixed(self, eeg_data, eeg_times, marker_data, marker_times):
        """Extract visual epochs with fixed slicing issues"""
        epochs = []
        
        # Convert to numpy arrays and ensure proper dtypes
        eeg_data = np.array(eeg_data, dtype=np.float32)
        eeg_times = np.array(eeg_times, dtype=np.float64).flatten()
        marker_data = np.array(marker_data, dtype=np.int32).flatten()
        marker_times = np.array(marker_times, dtype=np.float64).flatten()
        
        # Parse markers according to Crell specification
        letter_events = []
        current_letter_code = None
        current_letter = None
        fade_in_time = None
        fade_in_complete_time = None
        fade_out_start_time = None
        fade_out_complete_time = None
        
        for marker, marker_time in zip(marker_data, marker_times):
            if marker >= 100:  # Letter code (100+ascii_index)
                current_letter_code = int(marker)
                current_letter = self.letter_mapping.get(marker, None)
            elif marker == 1:  # Target letter starts to fade in
                fade_in_time = float(marker_time)
            elif marker == 2:  # Target letter is completely faded in
                fade_in_complete_time = float(marker_time)
            elif marker == 3:  # Target letter starts to fade out
                fade_out_start_time = float(marker_time)
            elif marker == 4:  # Target letter is completely faded out
                fade_out_complete_time = float(marker_time)
                
                # We focus on VISUAL phases only
                if (current_letter is not None and fade_in_time is not None and
                    fade_in_complete_time is not None and fade_out_start_time is not None):
                    
                    letter_events.append({
                        'letter': current_letter,
                        'letter_code': current_letter_code,
                        'start_time': fade_in_time,
                        'end_time': fade_out_complete_time,
                        'total_duration': fade_out_complete_time - fade_in_time
                    })
                
                # Reset for next letter
                current_letter_code = None
                current_letter = None
                fade_in_time = None
                fade_in_complete_time = None
                fade_out_start_time = None
                fade_out_complete_time = None
        
        print(f"    Found {len(letter_events)} visual letter events")
        
        # Extract EEG epochs for visual phases with fixed slicing
        for event in letter_events[:int(self.max_samples) if self.max_samples != float('inf') else len(letter_events)]:
            start_time = float(event['start_time'])
            end_time = float(event['end_time'])
            
            # Find indices with proper type conversion
            start_idx = int(np.searchsorted(eeg_times, start_time))
            end_idx = int(np.searchsorted(eeg_times, end_time))
            
            # Ensure valid range
            start_idx = max(0, min(start_idx, eeg_data.shape[1] - 1))
            end_idx = max(start_idx + 1, min(end_idx, eeg_data.shape[1]))
            
            if end_idx > start_idx and (end_idx - start_idx) > 100:
                # Extract epoch with proper slicing
                epoch_data = eeg_data[:, start_idx:end_idx].copy()
                
                # Fixed length processing for consistent tensor shapes
                target_length = 2250  # 4.5 seconds at 500Hz
                
                if epoch_data.shape[1] >= target_length:
                    # Take first target_length samples
                    epoch_data = epoch_data[:, :target_length]
                else:
                    # Pad with last values if shorter
                    padded_epoch = np.zeros((64, target_length), dtype=np.float32)
                    current_length = epoch_data.shape[1]
                    padded_epoch[:, :current_length] = epoch_data
                    # Pad remaining with last column values
                    if current_length > 0:
                        for ch in range(64):
                            padded_epoch[ch, current_length:] = epoch_data[ch, -1]
                    epoch_data = padded_epoch
                
                # Convert letter to numeric
                letter_to_num = {'a': 0, 'd': 1, 'e': 2, 'f': 3, 'j': 4,
                               'n': 5, 'o': 6, 's': 7, 't': 8, 'v': 9}
                letter_label = letter_to_num.get(event['letter'], 0)
                
                phase_info = {
                    'phase': 'visual_presentation',
                    'duration': event['total_duration'],
                    'letter_code': event['letter_code']
                }
                
                epochs.append((epoch_data, letter_label, phase_info))
        
        return epochs
    
    def load_stimulus_image(self, image_path: Path) -> np.ndarray:
        """Load and process stimulus image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            return img_array.transpose(2, 0, 1)  # CHW format
        except Exception as e:
            print(f"Error loading stimulus {image_path}: {e}")
            return np.zeros((3, 224, 224))

def create_crell_loaders(datasets_dir: str, batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create Crell data loaders with maximum samples"""
    
    datasets_path = Path(datasets_dir)
    
    print(f"🧠 Loading Crell with maximum available samples...")
    
    # Load Crell with maximum available samples
    crell_loader = CrellDataLoaderFixed(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=None  # No limit
    )
    
    samples = crell_loader.samples
    print(f"✓ Loaded {len(samples)} Crell samples")
    
    if len(samples) == 0:
        raise ValueError("No Crell samples loaded!")
    
    # Check distribution
    labels = [sample['label'] for sample in samples]
    unique_labels = sorted(set(labels))
    print(f"📊 Label distribution:")
    letters = ['a','d','e','f','j','n','o','s','t','v']
    for label in unique_labels:
        count = labels.count(label)
        letter = letters[label] if 0 <= label < len(letters) else '?'
        print(f"  Letter {letter} (label {label}): {count} samples")
    
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

def train_crell_model(datasets_dir: str,
                     output_dir: str = "ntvit_crell_outputs",
                     num_epochs: int = 50,
                     batch_size: int = 4,
                     device: str = 'cuda'):
    """Training pipeline for Crell only"""
    
    print("🧠 NT-ViT Crell Training Pipeline")
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
    
    # Load data
    train_loader, val_loader = create_crell_loaders(datasets_dir, batch_size=batch_size)
    
    # Crell always has 64 channels
    eeg_channels = 64
    print(f"📊 EEG channels: {eeg_channels}")
    
    # Create model
    model = NTViTEEGToFMRI(eeg_channels=eeg_channels).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer with gradient scaling for mixed precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            eeg_data = batch['eeg_data'].to(device, non_blocking=True)
            target_fmri = batch['synthetic_fmri_target'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(eeg_data, target_fmri)
                    recon_loss = F.mse_loss(outputs['synthetic_fmri'], target_fmri)
                    domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                    total_loss = recon_loss + 0.1 * domain_loss
                
                # Backward pass with scaling
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(eeg_data, target_fmri)
                recon_loss = F.mse_loss(outputs['synthetic_fmri'], target_fmri)
                domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                total_loss = recon_loss + 0.1 * domain_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Check for NaN
            if torch.isnan(total_loss):
                print(f"  Warning: NaN loss detected in batch {batch_idx}, skipping...")
                continue
            
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
                target_fmri = batch['synthetic_fmri_target'].to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(eeg_data, target_fmri)
                        recon_loss = F.mse_loss(outputs['synthetic_fmri'], target_fmri)
                        domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                        total_loss = recon_loss + 0.1 * domain_loss
                else:
                    outputs = model(eeg_data, target_fmri)
                    recon_loss = F.mse_loss(outputs['synthetic_fmri'], target_fmri)
                    domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                    total_loss = recon_loss + 0.1 * domain_loss
                
                if not torch.isnan(total_loss):
                    epoch_val_losses.append(total_loss.item())
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
        avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else float('inf')
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
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
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
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
        'total_samples': len(train_loader.dataset) + len(val_loader.dataset)
    }
    
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Final Train Loss: {train_losses[-1] if train_losses else 'N/A':.4f}")
    print(f"  Final Val Loss: {val_losses[-1] if val_losses else 'N/A':.4f}")
    print(f"  Models saved to: {output_path}")
    
    return model

def main():
    """Main function for Crell training"""
    
    print("🧠 NT-ViT Crell Training")
    print("=" * 50)
    
    # Configuration
    datasets_dir = "datasets"
    output_dir = "ntvit_crell_outputs"
    num_epochs = 30
    batch_size = 4  # Adjust based on GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📋 Configuration:")
    print(f"  Dataset: Crell (maximum available samples)")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
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
        # Train model
        print(f"\n🚀 Starting Crell training...")
        model = train_crell_model(
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )
        
        print(f"\n🎉 Crell training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
