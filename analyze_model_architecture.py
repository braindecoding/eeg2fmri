#!/usr/bin/env python3
"""
NT-ViT Model Architecture Analysis
=================================

Deep analysis of NT-ViT architecture to find issues causing infinity loss
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
    SpectrogramGenerator,
    NTViTGenerator,
    VisionTransformerEncoder,
    VisionTransformerDecoder,
    DomainMatchingModule
)

def analyze_model_components():
    """Analyze each component of NT-ViT model"""
    
    print("🔍 NT-ViT Model Architecture Analysis")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test parameters
    batch_size = 2
    eeg_channels = 14
    eeg_length = 256
    fmri_size = 3092
    
    print(f"\n📊 Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  EEG channels: {eeg_channels}")
    print(f"  EEG length: {eeg_length}")
    print(f"  fMRI size: {fmri_size}")
    
    # Create test data
    eeg_data = torch.randn(batch_size, eeg_channels, eeg_length).to(device)
    target_fmri = torch.randn(batch_size, fmri_size).to(device)
    
    print(f"\n🧪 Test Data:")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  fMRI shape: {target_fmri.shape}")
    print(f"  EEG range: [{torch.min(eeg_data):.4f}, {torch.max(eeg_data):.4f}]")
    print(f"  fMRI range: [{torch.min(target_fmri):.4f}, {torch.max(target_fmri):.4f}]")
    
    return eeg_data, target_fmri, device

def test_spectrogram_generator(eeg_data, device):
    """Test SpectrogramGenerator component"""
    
    print(f"\n🔍 Testing SpectrogramGenerator...")
    print("-" * 40)
    
    try:
        spectrogrammer = SpectrogramGenerator(eeg_channels=14).to(device)
        
        print(f"✓ SpectrogramGenerator created")
        print(f"  Parameters: {sum(p.numel() for p in spectrogrammer.parameters()):,}")
        
        # Forward pass
        with torch.no_grad():
            mel_spectrograms = spectrogrammer(eeg_data)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {eeg_data.shape}")
        print(f"  Output shape: {mel_spectrograms.shape}")
        print(f"  Output range: [{torch.min(mel_spectrograms):.4f}, {torch.max(mel_spectrograms):.4f}]")
        print(f"  Has NaN: {torch.isnan(mel_spectrograms).any()}")
        print(f"  Has Inf: {torch.isinf(mel_spectrograms).any()}")
        
        return mel_spectrograms, True
        
    except Exception as e:
        print(f"❌ SpectrogramGenerator failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_generator(mel_spectrograms, device):
    """Test NTViTGenerator component"""

    print(f"\n🔍 Testing NTViTGenerator...")
    print("-" * 40)

    try:
        generator = NTViTGenerator(
            mel_bins=128,
            fmri_voxels=3092,
            patch_size=16,
            embed_dim=256,
            num_heads=8,
            num_layers=6
        ).to(device)

        print(f"✓ NTViTGenerator created")
        print(f"  Parameters: {sum(p.numel() for p in generator.parameters()):,}")
        
        # Forward pass
        with torch.no_grad():
            outputs = generator(mel_spectrograms)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {mel_spectrograms.shape}")
        print(f"  Output keys: {list(outputs.keys())}")
        
        if 'translated_fmri' in outputs:
            translated_fmri = outputs['translated_fmri']
            print(f"  Translated fMRI shape: {translated_fmri.shape}")
            print(f"  Translated fMRI range: [{torch.min(translated_fmri):.4f}, {torch.max(translated_fmri):.4f}]")
            print(f"  Has NaN: {torch.isnan(translated_fmri).any()}")
            print(f"  Has Inf: {torch.isinf(translated_fmri).any()}")
        
        if 'latent_representation' in outputs:
            latent = outputs['latent_representation']
            print(f"  Latent shape: {latent.shape}")
            print(f"  Latent range: [{torch.min(latent):.4f}, {torch.max(latent):.4f}]")
            print(f"  Latent NaN: {torch.isnan(latent).any()}")
            print(f"  Latent Inf: {torch.isinf(latent).any()}")
        
        return outputs, True
        
    except Exception as e:
        print(f"❌ EEGToFMRIGenerator failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_domain_matcher(eeg_latent, target_fmri, device):
    """Test DomainMatchingModule component"""

    print(f"\n🔍 Testing DomainMatchingModule...")
    print("-" * 40)

    try:
        domain_matcher = DomainMatchingModule(
            fmri_voxels=3092,
            embed_dim=256
        ).to(device)

        print(f"✓ DomainMatchingModule created")
        print(f"  Parameters: {sum(p.numel() for p in domain_matcher.parameters()):,}")
        
        # Forward pass
        with torch.no_grad():
            outputs = domain_matcher(eeg_latent, target_fmri)
        
        print(f"✓ Forward pass successful")
        print(f"  EEG latent shape: {eeg_latent.shape}")
        print(f"  Target fMRI shape: {target_fmri.shape}")
        print(f"  Output keys: {list(outputs.keys())}")
        
        if 'alignment_loss' in outputs:
            alignment_loss = outputs['alignment_loss']
            print(f"  Alignment loss: {alignment_loss.item():.6f}")
            print(f"  Loss is finite: {torch.isfinite(alignment_loss)}")
        
        if 'eeg_latent' in outputs:
            eeg_out = outputs['eeg_latent']
            print(f"  EEG latent out shape: {eeg_out.shape}")
            print(f"  EEG latent range: [{torch.min(eeg_out):.4f}, {torch.max(eeg_out):.4f}]")
        
        if 'fmri_latent' in outputs:
            fmri_out = outputs['fmri_latent']
            print(f"  fMRI latent out shape: {fmri_out.shape}")
            print(f"  fMRI latent range: [{torch.min(fmri_out):.4f}, {torch.max(fmri_out):.4f}]")
        
        return outputs, True
        
    except Exception as e:
        print(f"❌ DomainMatcher failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_full_model(eeg_data, target_fmri, device):
    """Test full NT-ViT model"""
    
    print(f"\n🔍 Testing Full NT-ViT Model...")
    print("-" * 40)
    
    try:
        model = NTViTEEGToFMRI(eeg_channels=14).to(device)
        
        print(f"✓ Full model created")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(eeg_data, target_fmri)
        
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        
        if 'translated_fmri' in outputs:
            translated_fmri = outputs['translated_fmri']
            print(f"  Translated fMRI shape: {translated_fmri.shape}")
            print(f"  Translated fMRI range: [{torch.min(translated_fmri):.4f}, {torch.max(translated_fmri):.4f}]")
            print(f"  Has NaN: {torch.isnan(translated_fmri).any()}")
            print(f"  Has Inf: {torch.isinf(translated_fmri).any()}")
            
            # Test loss calculation
            loss = F.mse_loss(translated_fmri, target_fmri)
            print(f"  MSE Loss: {loss.item():.6f}")
            print(f"  Loss is finite: {torch.isfinite(loss)}")
        
        if 'domain_alignment_loss' in outputs:
            domain_loss = outputs['domain_alignment_loss']
            print(f"  Domain loss: {domain_loss.item():.6f}")
            print(f"  Domain loss finite: {torch.isfinite(domain_loss)}")
        
        return model, outputs, True
        
    except Exception as e:
        print(f"❌ Full model failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def test_gradient_flow(model, eeg_data, target_fmri, device):
    """Test gradient flow through the model"""
    
    print(f"\n🔍 Testing Gradient Flow...")
    print("-" * 40)
    
    try:
        model.train()  # Set to training mode
        
        # Forward pass
        outputs = model(eeg_data, target_fmri)
        
        # Calculate loss
        recon_loss = F.mse_loss(outputs['translated_fmri'], target_fmri)
        domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
        total_loss = recon_loss + 0.1 * domain_loss
        
        print(f"✓ Loss calculation successful")
        print(f"  Reconstruction loss: {recon_loss.item():.6f}")
        print(f"  Domain loss: {domain_loss.item():.6f}")
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  All losses finite: {torch.isfinite(total_loss)}")
        
        # Backward pass
        total_loss.backward()
        
        print(f"✓ Backward pass successful")
        
        # Check gradients
        grad_stats = []
        nan_grads = 0
        inf_grads = 0
        zero_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats.append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    nan_grads += 1
                    print(f"  ⚠️  NaN gradient in {name}")
                
                if torch.isinf(param.grad).any():
                    inf_grads += 1
                    print(f"  ⚠️  Inf gradient in {name}")
                
                if grad_norm == 0:
                    zero_grads += 1
            else:
                print(f"  ⚠️  No gradient for {name}")
        
        if grad_stats:
            print(f"  Gradient statistics:")
            print(f"    Min norm: {min(grad_stats):.6f}")
            print(f"    Max norm: {max(grad_stats):.6f}")
            print(f"    Mean norm: {np.mean(grad_stats):.6f}")
            print(f"    Std norm: {np.std(grad_stats):.6f}")
        
        print(f"  Gradient issues:")
        print(f"    NaN gradients: {nan_grads}")
        print(f"    Inf gradients: {inf_grads}")
        print(f"    Zero gradients: {zero_grads}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_layer_outputs(model, eeg_data, target_fmri, device):
    """Analyze outputs of each layer to find problematic layers"""
    
    print(f"\n🔍 Analyzing Layer Outputs...")
    print("-" * 40)
    
    try:
        model.eval()
        
        # Hook to capture layer outputs
        layer_outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = {
                        'shape': output.shape,
                        'min': torch.min(output).item(),
                        'max': torch.max(output).item(),
                        'mean': torch.mean(output).item(),
                        'std': torch.std(output).item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item()
                    }
                elif isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor):
                            layer_outputs[f"{name}_{key}"] = {
                                'shape': value.shape,
                                'min': torch.min(value).item(),
                                'max': torch.max(value).item(),
                                'mean': torch.mean(value).item(),
                                'std': torch.std(value).item(),
                                'has_nan': torch.isnan(value).any().item(),
                                'has_inf': torch.isinf(value).any().item()
                            }
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(eeg_data, target_fmri)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze results
        print(f"✓ Captured {len(layer_outputs)} layer outputs")
        
        problematic_layers = []
        
        for layer_name, stats in layer_outputs.items():
            if stats['has_nan'] or stats['has_inf']:
                problematic_layers.append(layer_name)
                print(f"  ❌ {layer_name}: NaN={stats['has_nan']}, Inf={stats['has_inf']}")
            elif abs(stats['max']) > 1e6 or abs(stats['min']) < -1e6:
                problematic_layers.append(layer_name)
                print(f"  ⚠️  {layer_name}: Extreme values [{stats['min']:.2e}, {stats['max']:.2e}]")
            elif stats['std'] > 1e3:
                print(f"  ⚠️  {layer_name}: High variance (std={stats['std']:.2e})")
        
        if not problematic_layers:
            print(f"  ✅ No obviously problematic layers found")
        else:
            print(f"  ❌ Problematic layers: {problematic_layers}")
        
        return layer_outputs, problematic_layers
        
    except Exception as e:
        print(f"❌ Layer analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main analysis function"""
    
    # Initialize
    eeg_data, target_fmri, device = analyze_model_components()
    
    # Test individual components
    mel_spectrograms, spec_ok = test_spectrogram_generator(eeg_data, device)
    
    if spec_ok and mel_spectrograms is not None:
        generator_outputs, gen_ok = test_generator(mel_spectrograms, device)
        
        if gen_ok and generator_outputs is not None:
            eeg_latent = generator_outputs.get('latent_representation')
            if eeg_latent is not None:
                domain_outputs, domain_ok = test_domain_matcher(eeg_latent, target_fmri, device)
    
    # Test full model
    model, full_outputs, full_ok = test_full_model(eeg_data, target_fmri, device)
    
    if full_ok and model is not None:
        # Test gradient flow
        grad_ok = test_gradient_flow(model, eeg_data, target_fmri, device)
        
        # Analyze layer outputs
        layer_outputs, problematic_layers = analyze_layer_outputs(model, eeg_data, target_fmri, device)
    
    # Summary
    print(f"\n📋 Analysis Summary:")
    print("=" * 30)
    print(f"✅ SpectrogramGenerator: {'OK' if spec_ok else 'FAILED'}")
    print(f"✅ EEGToFMRIGenerator: {'OK' if 'gen_ok' in locals() and gen_ok else 'FAILED'}")
    print(f"✅ DomainMatcher: {'OK' if 'domain_ok' in locals() and domain_ok else 'FAILED'}")
    print(f"✅ Full Model: {'OK' if full_ok else 'FAILED'}")
    print(f"✅ Gradient Flow: {'OK' if 'grad_ok' in locals() and grad_ok else 'FAILED'}")
    
    if 'problematic_layers' in locals() and problematic_layers:
        print(f"\n❌ Problematic Layers Found:")
        for layer in problematic_layers:
            print(f"  - {layer}")
    
    print(f"\n💡 Recommendations:")
    if not full_ok:
        print(f"  1. Fix model architecture issues")
    elif 'grad_ok' in locals() and not grad_ok:
        print(f"  1. Fix gradient flow issues")
    elif 'problematic_layers' in locals() and problematic_layers:
        print(f"  1. Investigate problematic layers")
        print(f"  2. Add layer normalization or different initialization")
    else:
        print(f"  1. Model architecture appears healthy")
        print(f"  2. Issue might be in training loop or data")

if __name__ == "__main__":
    main()
