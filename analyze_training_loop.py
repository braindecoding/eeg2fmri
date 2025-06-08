#!/usr/bin/env python3
"""
Training Loop Analysis - Find Infinity Loss Root Cause
======================================================

Since model architecture is healthy, analyze training loop for issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from train_ntvit import (
    NTViTEEGToFMRI, 
    MindBigDataLoader,
    EEGFMRIDataset
)
from torch.utils.data import DataLoader

def analyze_data_loading():
    """Analyze data loading for problematic samples"""
    
    print("🔍 Analyzing Data Loading...")
    print("=" * 50)
    
    datasets_path = Path("datasets")
    
    # Load small sample for analysis
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=100,
        balanced_per_label=False
    )
    
    samples = mindbig_loader.samples
    print(f"✓ Loaded {len(samples)} samples for analysis")
    
    if len(samples) == 0:
        print("❌ No samples loaded!")
        return False
    
    # Analyze raw samples
    problematic_samples = []
    
    for i, sample in enumerate(samples):
        eeg_data = sample['eeg_data']
        
        # Check for issues
        issues = []
        
        if np.isnan(eeg_data).any():
            issues.append("NaN values")
        
        if np.isinf(eeg_data).any():
            issues.append("Inf values")
        
        if np.abs(eeg_data).max() > 1e6:
            issues.append(f"Extreme values (max: {np.abs(eeg_data).max():.2e})")
        
        if np.std(eeg_data) > 1e6:
            issues.append(f"High variance (std: {np.std(eeg_data):.2e})")
        
        if np.std(eeg_data) < 1e-10:
            issues.append(f"Zero variance (std: {np.std(eeg_data):.2e})")
        
        if issues:
            problematic_samples.append((i, issues))
            print(f"  ⚠️  Sample {i}: {', '.join(issues)}")
    
    print(f"\n📊 Data Quality Summary:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Problematic samples: {len(problematic_samples)}")
    print(f"  Success rate: {(len(samples) - len(problematic_samples)) / len(samples) * 100:.1f}%")
    
    if problematic_samples:
        print(f"\n❌ Found {len(problematic_samples)} problematic samples")
        return False
    else:
        print(f"\n✅ All samples appear healthy")
        return True

def analyze_dataset_processing():
    """Analyze EEGFMRIDataset processing"""
    
    print(f"\n🔍 Analyzing Dataset Processing...")
    print("-" * 40)
    
    datasets_path = Path("datasets")
    
    # Load samples
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=50,
        balanced_per_label=False
    )
    
    samples = mindbig_loader.samples
    
    if len(samples) == 0:
        print("❌ No samples loaded!")
        return False
    
    # Create dataset
    dataset = EEGFMRIDataset(samples)
    
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    # Test dataset processing
    problematic_indices = []
    
    for i in range(min(20, len(dataset))):  # Test first 20 samples
        try:
            batch = dataset[i]
            
            eeg_data = batch['eeg_data']
            target_fmri = batch['translated_fmri_target']
            
            # Check processed data
            issues = []
            
            if torch.isnan(eeg_data).any():
                issues.append("EEG NaN")
            
            if torch.isinf(eeg_data).any():
                issues.append("EEG Inf")
            
            if torch.isnan(target_fmri).any():
                issues.append("fMRI NaN")
            
            if torch.isinf(target_fmri).any():
                issues.append("fMRI Inf")
            
            if torch.abs(eeg_data).max() > 1e6:
                issues.append(f"EEG extreme ({torch.abs(eeg_data).max():.2e})")
            
            if torch.abs(target_fmri).max() > 1e6:
                issues.append(f"fMRI extreme ({torch.abs(target_fmri).max():.2e})")
            
            if issues:
                problematic_indices.append((i, issues))
                print(f"  ⚠️  Sample {i}: {', '.join(issues)}")
            
        except Exception as e:
            problematic_indices.append((i, [f"Processing error: {e}"]))
            print(f"  ❌ Sample {i}: Processing error: {e}")
    
    print(f"\n📊 Dataset Processing Summary:")
    print(f"  Tested samples: {min(20, len(dataset))}")
    print(f"  Problematic samples: {len(problematic_indices)}")
    
    if problematic_indices:
        print(f"❌ Found processing issues")
        return False
    else:
        print(f"✅ Dataset processing appears healthy")
        return True

def analyze_batch_loading():
    """Analyze DataLoader batch creation"""
    
    print(f"\n🔍 Analyzing Batch Loading...")
    print("-" * 40)
    
    datasets_path = Path("datasets")
    
    # Load samples
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=50,
        balanced_per_label=False
    )
    
    samples = mindbig_loader.samples
    
    if len(samples) == 0:
        print("❌ No samples loaded!")
        return False
    
    # Create dataset and dataloader
    dataset = EEGFMRIDataset(samples)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print(f"✓ Created dataloader with batch_size=4")
    
    # Test batch loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    problematic_batches = []
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Test first 5 batches
            break
        
        try:
            eeg_data = batch['eeg_data'].to(device)
            target_fmri = batch['translated_fmri_target'].to(device)
            
            # Check batch data
            issues = []
            
            if torch.isnan(eeg_data).any():
                issues.append("EEG NaN")
            
            if torch.isinf(eeg_data).any():
                issues.append("EEG Inf")
            
            if torch.isnan(target_fmri).any():
                issues.append("fMRI NaN")
            
            if torch.isinf(target_fmri).any():
                issues.append("fMRI Inf")
            
            if torch.abs(eeg_data).max() > 1e6:
                issues.append(f"EEG extreme ({torch.abs(eeg_data).max():.2e})")
            
            if torch.abs(target_fmri).max() > 1e6:
                issues.append(f"fMRI extreme ({torch.abs(target_fmri).max():.2e})")
            
            if issues:
                problematic_batches.append((batch_idx, issues))
                print(f"  ⚠️  Batch {batch_idx}: {', '.join(issues)}")
            else:
                print(f"  ✅ Batch {batch_idx}: OK (EEG: {eeg_data.shape}, fMRI: {target_fmri.shape})")
            
        except Exception as e:
            problematic_batches.append((batch_idx, [f"Loading error: {e}"]))
            print(f"  ❌ Batch {batch_idx}: Loading error: {e}")
    
    print(f"\n📊 Batch Loading Summary:")
    print(f"  Tested batches: {min(5, len(dataloader))}")
    print(f"  Problematic batches: {len(problematic_batches)}")
    
    if problematic_batches:
        print(f"❌ Found batch loading issues")
        return False
    else:
        print(f"✅ Batch loading appears healthy")
        return True

def analyze_training_step():
    """Analyze single training step for issues"""
    
    print(f"\n🔍 Analyzing Training Step...")
    print("-" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = NTViTEEGToFMRI(eeg_channels=14).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Load real data
    datasets_path = Path("datasets")
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=20,
        balanced_per_label=False
    )
    
    samples = mindbig_loader.samples
    if len(samples) == 0:
        print("❌ No samples loaded!")
        return False
    
    dataset = EEGFMRIDataset(samples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print(f"✓ Created training setup")
    
    # Test multiple training steps
    model.train()
    step_results = []
    
    for step, batch in enumerate(dataloader):
        if step >= 10:  # Test first 10 steps
            break
        
        try:
            # Move to device
            eeg_data = batch['eeg_data'].to(device)
            target_fmri = batch['translated_fmri_target'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(eeg_data, target_fmri)
            
            # Calculate loss
            recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
            domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
            total_loss = recon_loss + 0.1 * domain_loss
            
            # Check loss values
            step_result = {
                'step': step,
                'recon_loss': recon_loss.item(),
                'domain_loss': domain_loss.item(),
                'total_loss': total_loss.item(),
                'recon_finite': torch.isfinite(recon_loss).item(),
                'domain_finite': torch.isfinite(domain_loss).item(),
                'total_finite': torch.isfinite(total_loss).item()
            }
            
            step_results.append(step_result)
            
            if not step_result['total_finite']:
                print(f"  ❌ Step {step}: Non-finite loss!")
                print(f"    Recon: {step_result['recon_loss']}")
                print(f"    Domain: {step_result['domain_loss']}")
                print(f"    Total: {step_result['total_loss']}")
            else:
                print(f"  ✅ Step {step}: Loss = {step_result['total_loss']:.6f}")
                
                # Try backward pass
                try:
                    total_loss.backward()
                    
                    # Check gradients
                    grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    if not np.isfinite(grad_norm):
                        print(f"    ⚠️  Non-finite gradient norm: {grad_norm}")
                    else:
                        print(f"    Grad norm: {grad_norm:.6f}")
                    
                    optimizer.step()
                    
                except Exception as e:
                    print(f"    ❌ Backward/step failed: {e}")
            
        except Exception as e:
            print(f"  ❌ Step {step}: Training error: {e}")
            step_results.append({
                'step': step,
                'error': str(e)
            })
    
    # Analyze results
    finite_steps = [r for r in step_results if r.get('total_finite', False)]
    infinite_steps = [r for r in step_results if not r.get('total_finite', True)]
    
    print(f"\n📊 Training Step Summary:")
    print(f"  Total steps tested: {len(step_results)}")
    print(f"  Finite loss steps: {len(finite_steps)}")
    print(f"  Infinite loss steps: {len(infinite_steps)}")
    
    if finite_steps:
        losses = [r['total_loss'] for r in finite_steps]
        print(f"  Loss range: [{min(losses):.6f}, {max(losses):.6f}]")
        print(f"  Loss mean: {np.mean(losses):.6f}")
    
    if infinite_steps:
        print(f"❌ Found infinite loss steps")
        return False
    else:
        print(f"✅ All training steps produced finite losses")
        return True

def main():
    """Main analysis function"""
    
    print("🔍 Training Loop Analysis - Find Infinity Loss Root Cause")
    print("=" * 70)
    print("Model architecture is healthy, analyzing training pipeline...")
    
    # Run analyses
    data_ok = analyze_data_loading()
    dataset_ok = analyze_dataset_processing()
    batch_ok = analyze_batch_loading()
    training_ok = analyze_training_step()
    
    # Summary
    print(f"\n📋 Training Loop Analysis Summary:")
    print("=" * 40)
    print(f"✅ Data Loading: {'OK' if data_ok else 'ISSUES FOUND'}")
    print(f"✅ Dataset Processing: {'OK' if dataset_ok else 'ISSUES FOUND'}")
    print(f"✅ Batch Loading: {'OK' if batch_ok else 'ISSUES FOUND'}")
    print(f"✅ Training Steps: {'OK' if training_ok else 'ISSUES FOUND'}")
    
    if all([data_ok, dataset_ok, batch_ok, training_ok]):
        print(f"\n🎉 All training components appear healthy!")
        print(f"💡 Infinity loss might be caused by:")
        print(f"  1. Specific data samples in larger dataset")
        print(f"  2. Learning rate too high for some batches")
        print(f"  3. Accumulation of small numerical errors")
        print(f"  4. GPU memory issues causing corruption")
    else:
        print(f"\n❌ Found issues in training pipeline!")
        print(f"💡 Focus on fixing the problematic components above")

if __name__ == "__main__":
    main()
