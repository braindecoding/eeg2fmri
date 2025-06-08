#!/usr/bin/env python3
"""
Fixed Crell Translated fMRI Generation - Using Real Stimulus Images
====================================================================

Generate translated fMRI using REAL stimulus images from datasets/crellStimuli/
"""

import torch
import numpy as np
from pathlib import Path
import json
import scipy.io as sio
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from train_ntvit import (
    NTViTEEGToFMRI, 
    CrellDataLoader
)

def load_real_crell_stimulus_images(stimuli_dir: str, samples: list):
    """Load REAL stimulus images from datasets/crellStimuli/"""
    
    print(f"🖼️  Loading REAL Crell stimulus images from {stimuli_dir}...")
    
    stimuli_path = Path(stimuli_dir)
    stimulus_images = []
    
    for sample in samples:
        letter = sample['letter']
        
        # Load actual letter image
        image_path = stimuli_path / f"{letter}.png"
        
        if image_path.exists():
            try:
                # Load and process real image
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                
                # Resize to 28x28 (MNIST standard)
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.uint8)
                
                # Flatten to 784 (28*28)
                stimulus_images.append(img_array.flatten())
                
            except Exception as e:
                print(f"⚠️  Error loading {image_path}: {e}")
                # Fallback to zeros if image can't be loaded
                stimulus_images.append(np.zeros(784, dtype=np.uint8))
        else:
            print(f"⚠️  Image not found: {image_path}")
            # Fallback to zeros if image doesn't exist
            stimulus_images.append(np.zeros(784, dtype=np.uint8))
    
    stimulus_array = np.array(stimulus_images, dtype=np.uint8)
    print(f"✅ Loaded REAL Crell stimulus images: {stimulus_array.shape}")
    print(f"  Image range: [{np.min(stimulus_array)}, {np.max(stimulus_array)}]")
    
    return stimulus_array

def load_trained_crell_model(model_path: str, device: str):
    """Load trained Crell model"""
    
    print(f"🔄 Loading trained Crell model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with 64 channels for Crell
    model = NTViTEEGToFMRI(eeg_channels=64).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Crell model loaded successfully")
    print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model, checkpoint

def generate_crell_translated_fmri_real_stimuli(model_path: str, datasets_dir: str, output_dir: str, device: str = 'cuda'):
    """Generate translated fMRI for Crell using REAL stimulus images"""
    
    print(f"🧠 Generating Crell Translated fMRI with REAL Stimulus Images")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load trained model
    model, checkpoint = load_trained_crell_model(model_path, device)
    
    # Load Crell dataset
    print(f"📊 Loading Crell dataset...")
    datasets_path = Path(datasets_dir)
    
    crell_loader = CrellDataLoader(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=1000  # Use same limit as training
    )
    
    samples = crell_loader.samples
    print(f"✅ Loaded {len(samples)} Crell samples")
    
    # Check letter distribution
    letters = [sample['letter'] for sample in samples]
    print(f"📊 Letter distribution:")
    for letter in ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']:
        count = letters.count(letter)
        print(f"  Letter {letter}: {count} samples")
    
    # Generate translated fMRI
    print(f"\n🚀 Generating translated fMRI...")
    
    all_translated_fmri = []
    all_eeg_data = []
    all_labels = []
    
    # Letter to label mapping
    letter_to_label = {letter: idx for idx, letter in enumerate(['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v'])}
    
    model.eval()
    with torch.no_grad():
        # Process in batches
        batch_size = 8
        total_batches = (len(samples) + batch_size - 1) // batch_size
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            
            # Prepare batch data
            batch_eeg = []
            batch_labels = []
            
            for sample in batch_samples:
                # EEG data with normalization (same as training)
                eeg_data = torch.tensor(sample['eeg_data'], dtype=torch.float32)
                
                # Normalize EEG
                eeg_mean = eeg_data.mean()
                eeg_std = eeg_data.std()
                eeg_std = torch.clamp(eeg_std, min=1e-6)
                eeg_data = (eeg_data - eeg_mean) / eeg_std
                eeg_data = torch.clamp(eeg_data, -3.0, 3.0)
                
                batch_eeg.append(eeg_data)
                
                # Label from letter
                letter = sample['letter']
                label = letter_to_label.get(letter, 0)
                batch_labels.append(label)
            
            # Stack into tensors
            eeg_batch = torch.stack(batch_eeg).to(device, non_blocking=True)
            labels_batch = torch.tensor(batch_labels, dtype=torch.long)
            
            # Create dummy target fMRI (same as training)
            target_fmri = torch.randn(len(batch_samples), 3092) * 0.01
            target_fmri = torch.clamp(target_fmri, -1.0, 1.0).to(device, non_blocking=True)
            
            # Generate translated fMRI
            outputs = model(eeg_batch, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            # Store results
            all_translated_fmri.append(translated_fmri.cpu().numpy())
            all_eeg_data.append(eeg_batch.cpu().numpy())
            all_labels.append(labels_batch.numpy())
            
            # Progress update
            batch_idx = i // batch_size
            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"  Processed batch {batch_idx + 1}/{total_batches} ({progress:.1f}%)")
    
    # Concatenate all results
    translated_fmri_array = np.concatenate(all_translated_fmri, axis=0)
    eeg_data_array = np.concatenate(all_eeg_data, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    print(f"✅ Generated translated fMRI for {len(translated_fmri_array)} samples")
    print(f"  Translated fMRI shape: {translated_fmri_array.shape}")
    print(f"  EEG data shape: {eeg_data_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    
    # Load REAL stimulus images
    stimulus_images = load_real_crell_stimulus_images(
        str(datasets_path / "crellStimuli"), 
        samples
    )
    
    # Save in CortexFlow format
    print(f"\n💾 Saving in CortexFlow format with REAL stimulus images...")
    
    # Prepare data for CortexFlow (.mat format)
    cortexflow_data = {
        'fmri': translated_fmri_array.astype(np.float64),  # (N, 3092)
        'stim': stimulus_images.astype(np.uint8),          # (N, 784) - REAL images
        'labels': labels_array.reshape(-1, 1).astype(np.uint8)  # (N, 1)
    }
    
    # Save as .mat file
    mat_file_path = output_path / 'crell_translated_fmri_real_stimuli.mat'
    sio.savemat(str(mat_file_path), cortexflow_data)
    print(f"✅ Saved CortexFlow format: {mat_file_path}")
    
    # Save individual arrays
    np.save(output_path / 'crell_translated_fmri_real.npy', translated_fmri_array)
    np.save(output_path / 'crell_eeg_data_real.npy', eeg_data_array)
    np.save(output_path / 'crell_real_stimulus_images.npy', stimulus_images)
    np.save(output_path / 'crell_labels_real.npy', labels_array)
    print(f"✅ Saved individual arrays to {output_dir}")
    
    # Save metadata
    metadata = {
        'dataset': 'Crell',
        'model_path': model_path,
        'total_samples': len(translated_fmri_array),
        'eeg_channels': 64,
        'eeg_timepoints': 2250,
        'fmri_voxels': 3092,
        'stimulus_size': 784,
        'num_classes': 10,  # 10 letters
        'letters': ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v'],
        'stimulus_source': 'REAL images from datasets/crellStimuli/',
        'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
        'training_epochs': checkpoint.get('epoch', 'N/A'),
        'letter_distribution': {letter: letters.count(letter) for letter in ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']},
        'data_shapes': {
            'fmri': list(translated_fmri_array.shape),
            'eeg': list(eeg_data_array.shape),
            'stimulus': list(stimulus_images.shape),
            'labels': list(labels_array.shape)
        },
        'cortexflow_format': {
            'fmri': '(N, 3092) float64',
            'stim': '(N, 784) uint8 - REAL letter images', 
            'labels': '(N, 1) uint8'
        }
    }
    
    with open(output_path / 'crell_real_stimuli_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {output_path / 'crell_real_stimuli_metadata.json'}")
    
    # Summary
    print(f"\n📊 Crell Generation Summary (REAL Stimuli):")
    print(f"  Total samples: {len(translated_fmri_array)}")
    print(f"  fMRI data: {translated_fmri_array.shape} (mean: {np.mean(translated_fmri_array):.6f})")
    print(f"  Stimulus data: {stimulus_images.shape} (REAL images)")
    print(f"  Stimulus range: [{np.min(stimulus_images)}, {np.max(stimulus_images)}]")
    print(f"  Labels: {labels_array.shape}")
    print(f"  Output directory: {output_dir}")
    
    print(f"\n🎯 Ready for CortexFlow with REAL stimulus images!")
    print(f"  Use file: {mat_file_path}")
    print(f"  Format: fmri=(N,3092), stim=(N,784), labels=(N,1)")
    
    return translated_fmri_array, stimulus_images, labels_array

def main():
    """Main function"""
    
    print("🧠 Crell Translated fMRI Generation - REAL Stimulus Images")
    print("=" * 70)
    print("📋 Using REAL letter images from datasets/crellStimuli/")
    
    # Configuration
    model_path = "crell_full_outputs/best_crell_model.pth"
    datasets_dir = "datasets"
    output_dir = "crell_translated_fmri_outputs_real_stimuli"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: Crell (full dataset)")
    print(f"  Stimuli: REAL images from datasets/crellStimuli/")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print(f"💡 Please run Crell training first")
        return
    
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
        # Generate translated fMRI with REAL stimulus images
        translated_fmri, stimulus_images, labels = generate_crell_translated_fmri_real_stimuli(
            model_path=model_path,
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            device=device
        )
        
        print(f"\n🎉 Crell translated fMRI generation with REAL stimuli completed!")
        print(f"📊 Generated {len(translated_fmri)} samples")
        print(f"🖼️  Using REAL letter images from datasets/")
        print(f"🚀 Ready for CortexFlow integration!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
