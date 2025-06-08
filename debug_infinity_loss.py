#!/usr/bin/env python3
"""
Debug Infinity Loss Issue
=========================

Deep analysis of what's causing infinity loss in NT-ViT training
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

def analyze_data_statistics():
    """Analyze data statistics to find potential issues"""
    
    print("🔍 Analyzing Data Statistics...")
    print("=" * 50)
    
    # Load a small sample of data
    datasets_path = Path("datasets")
    
    # Load MindBigData
    print("📊 Loading MindBigData sample...")
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=50,  # Small sample for analysis
        balanced_per_label=True
    )
    
    samples = mindbig_loader.samples
    print(f"✓ Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("❌ No samples loaded")
        return
    
    # Analyze EEG data
    print("\n📈 EEG Data Analysis:")
    eeg_values = []
    for i, sample in enumerate(samples[:10]):  # First 10 samples
        eeg_data = sample['eeg_data']
        print(f"  Sample {i}:")
        print(f"    Shape: {eeg_data.shape}")
        print(f"    Min: {np.min(eeg_data):.4f}")
        print(f"    Max: {np.max(eeg_data):.4f}")
        print(f"    Mean: {np.mean(eeg_data):.4f}")
        print(f"    Std: {np.std(eeg_data):.4f}")
        print(f"    Has NaN: {np.isnan(eeg_data).any()}")
        print(f"    Has Inf: {np.isinf(eeg_data).any()}")
        
        eeg_values.extend(eeg_data.flatten())
    
    # Overall statistics
    eeg_values = np.array(eeg_values)
    print(f"\n📊 Overall EEG Statistics:")
    print(f"  Total values: {len(eeg_values)}")
    print(f"  Min: {np.min(eeg_values):.4f}")
    print(f"  Max: {np.max(eeg_values):.4f}")
    print(f"  Mean: {np.mean(eeg_values):.4f}")
    print(f"  Std: {np.std(eeg_values):.4f}")
    print(f"  Has NaN: {np.isnan(eeg_values).any()}")
    print(f"  Has Inf: {np.isinf(eeg_values).any()}")
    
    # Check for extreme values
    extreme_threshold = 1e6
    extreme_count = np.sum(np.abs(eeg_values) > extreme_threshold)
    print(f"  Values > {extreme_threshold}: {extreme_count}")
    
    # Analyze synthetic fMRI targets
    print("\n🧠 Synthetic fMRI Analysis:")
    dataset = EEGFMRIDataset(samples[:5])
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        fmri_target = sample['synthetic_fmri_target']
        
        print(f"  Sample {i}:")
        print(f"    Shape: {fmri_target.shape}")
        print(f"    Min: {torch.min(fmri_target):.4f}")
        print(f"    Max: {torch.max(fmri_target):.4f}")
        print(f"    Mean: {torch.mean(fmri_target):.4f}")
        print(f"    Std: {torch.std(fmri_target):.4f}")
        print(f"    Has NaN: {torch.isnan(fmri_target).any()}")
        print(f"    Has Inf: {torch.isinf(fmri_target).any()}")

def test_model_forward_pass():
    """Test model forward pass with simple data"""
    
    print("\n🔍 Testing Model Forward Pass...")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = NTViTEEGToFMRI(eeg_channels=14).to(device)
    
    # Test with simple synthetic data
    batch_size = 2
    eeg_channels = 14
    eeg_length = 256
    fmri_size = 3092
    
    print(f"\n🧪 Test 1: Random normal data")
    eeg_data = torch.randn(batch_size, eeg_channels, eeg_length).to(device)
    fmri_target = torch.randn(batch_size, fmri_size).to(device)
    
    print(f"  EEG input shape: {eeg_data.shape}")
    print(f"  EEG stats: min={torch.min(eeg_data):.4f}, max={torch.max(eeg_data):.4f}")
    print(f"  fMRI target shape: {fmri_target.shape}")
    print(f"  fMRI stats: min={torch.min(fmri_target):.4f}, max={torch.max(fmri_target):.4f}")
    
    try:
        with torch.no_grad():
            outputs = model(eeg_data, fmri_target)
            
        print(f"  ✅ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        
        if 'synthetic_fmri' in outputs:
            synthetic_fmri = outputs['synthetic_fmri']
            print(f"  Synthetic fMRI shape: {synthetic_fmri.shape}")
            print(f"  Synthetic fMRI stats: min={torch.min(synthetic_fmri):.4f}, max={torch.max(synthetic_fmri):.4f}")
            print(f"  Has NaN: {torch.isnan(synthetic_fmri).any()}")
            print(f"  Has Inf: {torch.isinf(synthetic_fmri).any()}")
            
            # Test loss calculation
            loss = F.mse_loss(synthetic_fmri, fmri_target)
            print(f"  MSE Loss: {loss.item():.6f}")
            print(f"  Loss is finite: {torch.isfinite(loss)}")
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🧪 Test 2: Scaled data (like real EEG)")
    # Scale to match real EEG data range
    eeg_data_scaled = torch.randn(batch_size, eeg_channels, eeg_length).to(device) * 1000 + 4000
    
    print(f"  Scaled EEG stats: min={torch.min(eeg_data_scaled):.4f}, max={torch.max(eeg_data_scaled):.4f}")
    
    try:
        with torch.no_grad():
            outputs = model(eeg_data_scaled, fmri_target)
            
        print(f"  ✅ Forward pass successful")
        
        if 'synthetic_fmri' in outputs:
            synthetic_fmri = outputs['synthetic_fmri']
            print(f"  Synthetic fMRI stats: min={torch.min(synthetic_fmri):.4f}, max={torch.max(synthetic_fmri):.4f}")
            print(f"  Has NaN: {torch.isnan(synthetic_fmri).any()}")
            print(f"  Has Inf: {torch.isinf(synthetic_fmri).any()}")
            
            loss = F.mse_loss(synthetic_fmri, fmri_target)
            print(f"  MSE Loss: {loss.item():.6f}")
            print(f"  Loss is finite: {torch.isfinite(loss)}")
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")

def test_with_real_data():
    """Test model with actual loaded data"""
    
    print("\n🔍 Testing with Real Data...")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load real data
    datasets_path = Path("datasets")
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=10,  # Very small sample
        balanced_per_label=False
    )
    
    samples = mindbig_loader.samples
    if len(samples) == 0:
        print("❌ No samples loaded")
        return
    
    dataset = EEGFMRIDataset(samples)
    
    # Test first sample
    sample = dataset[0]
    eeg_data = sample['eeg_data'].unsqueeze(0).to(device)  # Add batch dimension
    fmri_target = sample['synthetic_fmri_target'].unsqueeze(0).to(device)
    
    print(f"Real data shapes: EEG {eeg_data.shape}, fMRI {fmri_target.shape}")
    print(f"EEG stats: min={torch.min(eeg_data):.4f}, max={torch.max(eeg_data):.4f}")
    print(f"fMRI stats: min={torch.min(fmri_target):.4f}, max={torch.max(fmri_target):.4f}")
    
    # Create model
    model = NTViTEEGToFMRI(eeg_channels=eeg_data.shape[1]).to(device)
    
    try:
        with torch.no_grad():
            outputs = model(eeg_data, fmri_target)
            
        print(f"✅ Forward pass successful")
        
        if 'synthetic_fmri' in outputs:
            synthetic_fmri = outputs['synthetic_fmri']
            print(f"Synthetic fMRI stats: min={torch.min(synthetic_fmri):.4f}, max={torch.max(synthetic_fmri):.4f}")
            print(f"Has NaN: {torch.isnan(synthetic_fmri).any()}")
            print(f"Has Inf: {torch.isinf(synthetic_fmri).any()}")
            
            loss = F.mse_loss(synthetic_fmri, fmri_target)
            print(f"MSE Loss: {loss.item():.6f}")
            print(f"Loss is finite: {torch.isfinite(loss)}")
            
            # Test backward pass
            loss.backward()
            print(f"✅ Backward pass successful")
            
            # Check gradients
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            print(f"Total gradient norm: {total_grad_norm:.6f}")
        
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_reduced_data():
    """Test with reduced data size (640 samples)"""
    
    print("\n🔍 Testing with Reduced Data (640 samples)...")
    print("=" * 50)
    
    datasets_path = Path("datasets")
    
    # Load with reduced size
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=640,  # Reduced from 1200
        balanced_per_label=True
    )
    
    samples = mindbig_loader.samples
    print(f"✓ Loaded {len(samples)} samples (reduced)")
    
    if len(samples) == 0:
        print("❌ No samples loaded")
        return
    
    # Check distribution
    labels = [sample['label'] for sample in samples]
    print(f"📊 Label distribution:")
    for digit in range(10):
        count = labels.count(digit)
        print(f"  Digit {digit}: {count} samples")
    
    # Test data statistics
    eeg_values = []
    for sample in samples[:10]:
        eeg_data = sample['eeg_data']
        eeg_values.extend(eeg_data.flatten())
    
    eeg_values = np.array(eeg_values)
    print(f"\n📊 EEG Statistics (640 samples):")
    print(f"  Mean: {np.mean(eeg_values):.4f}")
    print(f"  Std: {np.std(eeg_values):.4f}")
    print(f"  Min: {np.min(eeg_values):.4f}")
    print(f"  Max: {np.max(eeg_values):.4f}")
    
    return len(samples)

def main():
    """Main debug function"""
    
    print("🔍 NT-ViT Infinity Loss Debug Analysis")
    print("=" * 60)
    
    # Test 1: Data statistics
    analyze_data_statistics()
    
    # Test 2: Model forward pass
    test_model_forward_pass()
    
    # Test 3: Real data
    test_with_real_data()
    
    # Test 4: Reduced data
    reduced_samples = test_reduced_data()
    
    print(f"\n📋 Summary:")
    print("=" * 30)
    print(f"🔍 Analysis completed")
    print(f"📊 Reduced data size: {reduced_samples} samples")
    print(f"💡 Check output above for potential issues")
    
    print(f"\n🎯 Recommendations:")
    print(f"1. If model works with synthetic data but fails with real data:")
    print(f"   → Data preprocessing issue")
    print(f"2. If model fails even with synthetic data:")
    print(f"   → Model architecture issue")
    print(f"3. If gradients are too large:")
    print(f"   → Need better gradient clipping/learning rate")

if __name__ == "__main__":
    main()
