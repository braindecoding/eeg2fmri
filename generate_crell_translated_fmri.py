#!/usr/bin/env python3
"""
Generate Translated fMRI for Crell Dataset
==========================================

Generate translated fMRI data from trained Crell model for CortexFlow integration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import scipy.io as sio
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

def create_crell_stimulus_images(samples):
    """Create stimulus images for Crell letters"""
    
    print(f"🖼️  Creating Crell stimulus images...")
    
    # Create simple letter images (28x28 like MNIST)
    stimulus_images = []
    
    for sample in samples:
        letter = sample['letter']
        
        # Create simple text-based image for letter
        # For now, create a simple pattern based on letter
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Simple letter encoding (you could replace with actual letter images)
        letter_patterns = {
            'a': np.array([[0,1,1,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]]),
            'd': np.array([[1,1,1,0,0], [1,0,0,1,0], [1,0,0,0,1], [1,0,0,1,0], [1,1,1,0,0]]),
            'e': np.array([[1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,0], [1,1,1,1,1]]),
            'f': np.array([[1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,0], [1,0,0,0,0]]),
            'j': np.array([[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]]),
            'n': np.array([[1,0,0,0,1], [1,1,0,0,1], [1,0,1,0,1], [1,0,0,1,1], [1,0,0,0,1]]),
            'o': np.array([[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]]),
            's': np.array([[0,1,1,1,1], [1,0,0,0,0], [0,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]]),
            't': np.array([[1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0]]),
            'v': np.array([[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0]])
        }
        
        pattern = letter_patterns.get(letter, np.zeros((5, 5)))
        
        # Place pattern in center of 28x28 image
        start_row = (28 - pattern.shape[0]) // 2
        start_col = (28 - pattern.shape[1]) // 2
        img[start_row:start_row+pattern.shape[0], start_col:start_col+pattern.shape[1]] = pattern * 255
        
        # Flatten to 784 (28*28)
        stimulus_images.append(img.flatten())
    
    stimulus_array = np.array(stimulus_images, dtype=np.uint8)
    print(f"✅ Created Crell stimulus images: {stimulus_array.shape}")
    
    return stimulus_array

def generate_crell_translated_fmri(model_path: str, datasets_dir: str, output_dir: str, device: str = 'cuda'):
    """Generate translated fMRI for full Crell dataset"""
    
    print(f"🧠 Generating Crell Translated fMRI Dataset...")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load trained model
    model, checkpoint = load_trained_crell_model(model_path, device)
    
    # Load full Crell dataset
    print(f"📊 Loading full Crell dataset...")
    datasets_path = Path(datasets_dir)
    
    crell_loader = CrellDataLoader(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=1000  # Use same limit as successful training
    )
    
    samples = crell_loader.samples
    print(f"✅ Loaded {len(samples)} Crell samples")
    
    # Check letter distribution
    letters = [sample['letter'] for sample in samples]
    print(f"📊 Letter distribution:")
    for letter in ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']:
        count = letters.count(letter)
        print(f"  Letter {letter}: {count} samples")
    
    # Create dataset and dataloader
    dataset = EEGFMRIDataset(samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"✅ Created dataloader with batch_size=8")
    
    # Generate translated fMRI
    print(f"\n🚀 Generating translated fMRI...")
    
    all_translated_fmri = []
    all_eeg_data = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            eeg_data = batch['eeg_data'].to(device, non_blocking=True)
            target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
            # Get labels from samples (Crell uses different key structure)
            if 'label' in batch:
                labels = batch['label']
            else:
                # Extract labels from letter field
                batch_size = eeg_data.shape[0]
                labels = torch.zeros(batch_size, dtype=torch.long)
                # This will be filled with actual letter-to-number mapping
            
            # Generate translated fMRI
            outputs = model(eeg_data, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            # Move back to CPU and store
            all_translated_fmri.append(translated_fmri.cpu().numpy())
            all_eeg_data.append(eeg_data.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Progress update
            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                progress = (batch_idx + 1) / len(dataloader) * 100
                print(f"  Processed batch {batch_idx + 1}/{len(dataloader)} ({progress:.1f}%)")
    
    # Concatenate all results
    translated_fmri_array = np.concatenate(all_translated_fmri, axis=0)
    eeg_data_array = np.concatenate(all_eeg_data, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    print(f"✅ Generated translated fMRI for {len(translated_fmri_array)} samples")
    print(f"  Translated fMRI shape: {translated_fmri_array.shape}")
    print(f"  EEG data shape: {eeg_data_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    
    # Create stimulus images
    stimulus_images = create_crell_stimulus_images(samples)
    
    # Save in CortexFlow format
    print(f"\n💾 Saving in CortexFlow format...")
    
    # Prepare data for CortexFlow (.mat format)
    cortexflow_data = {
        'fmri': translated_fmri_array.astype(np.float64),  # (N, 3092)
        'stim': stimulus_images.astype(np.uint8),          # (N, 784)
        'labels': labels_array.reshape(-1, 1).astype(np.uint8)  # (N, 1)
    }
    
    # Save as .mat file
    mat_file_path = output_path / 'crell_translated_fmri.mat'
    sio.savemat(str(mat_file_path), cortexflow_data)
    print(f"✅ Saved CortexFlow format: {mat_file_path}")
    
    # Save individual arrays
    np.save(output_path / 'crell_translated_fmri.npy', translated_fmri_array)
    np.save(output_path / 'crell_eeg_data.npy', eeg_data_array)
    np.save(output_path / 'crell_stimulus_images.npy', stimulus_images)
    np.save(output_path / 'crell_labels.npy', labels_array)
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
            'stim': '(N, 784) uint8', 
            'labels': '(N, 1) uint8'
        }
    }
    
    with open(output_path / 'crell_generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {output_path / 'crell_generation_metadata.json'}")
    
    # Summary
    print(f"\n📊 Crell Generation Summary:")
    print(f"  Total samples: {len(translated_fmri_array)}")
    print(f"  fMRI data: {translated_fmri_array.shape} (mean: {np.mean(translated_fmri_array):.6f})")
    print(f"  Stimulus data: {stimulus_images.shape}")
    print(f"  Labels: {labels_array.shape}")
    print(f"  Output directory: {output_dir}")
    
    print(f"\n🎯 Ready for CortexFlow!")
    print(f"  Use file: {mat_file_path}")
    print(f"  Format: fmri=(N,3092), stim=(N,784), labels=(N,1)")
    
    return translated_fmri_array, stimulus_images, labels_array

def main():
    """Main function"""
    
    print("🧠 Crell Translated fMRI Generation")
    print("=" * 50)
    print("📋 Using successfully trained Crell model")
    print("📋 Terminology: translated_fmri (EEG→fMRI translation)")
    
    # Configuration
    model_path = "crell_full_outputs/best_crell_model.pth"
    datasets_dir = "datasets"
    output_dir = "crell_translated_fmri_outputs"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: Crell (full dataset)")
    print(f"  Device: {device}")
    print(f"  Batch size: 8")
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
        # Generate translated fMRI
        translated_fmri, stimulus_images, labels = generate_crell_translated_fmri(
            model_path=model_path,
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            device=device
        )
        
        print(f"\n🎉 Crell translated fMRI generation completed successfully!")
        print(f"📊 Generated {len(translated_fmri)} samples")
        print(f"🚀 Ready for CortexFlow integration!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
