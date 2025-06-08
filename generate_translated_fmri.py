#!/usr/bin/env python3
"""
Generate Translated fMRI from Trained NT-ViT Model
==================================================

Use the successfully trained robust NT-ViT model to generate translated fMRI
from EEG data for the full dataset.
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
    MindBigDataLoader,
    EEGFMRIDataset
)

def load_trained_model(model_path: str, device: str = 'cuda'):
    """Load the trained robust NT-ViT model"""
    
    print(f"🔄 Loading trained model from {model_path}...")
    
    # Load checkpoint (fix for PyTorch 2.6 weights_only default change)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with same architecture
    model = NTViTEEGToFMRI(eeg_channels=14).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model, checkpoint

def generate_translated_fmri_dataset(model, 
                                   datasets_dir: str,
                                   output_dir: str = "translated_fmri_outputs",
                                   batch_size: int = 8,
                                   device: str = 'cuda'):
    """Generate translated fMRI for the full dataset"""
    
    print(f"🧠 Generating Translated fMRI Dataset...")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    datasets_path = Path(datasets_dir)
    
    # Load full MindBigData dataset (without outlier filtering for generation)
    print(f"📊 Loading full MindBigData dataset...")
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=1200,  # Use full available dataset
        balanced_per_label=True
    )
    
    samples = mindbig_loader.samples
    print(f"✅ Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        raise ValueError("No samples loaded!")
    
    # Check label distribution
    labels = [sample['label'] for sample in samples]
    print(f"📊 Label distribution:")
    for digit in range(10):
        count = labels.count(digit)
        print(f"  Digit {digit}: {count} samples")
    
    # Create dataset and dataloader
    dataset = EEGFMRIDataset(samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep order for consistent indexing
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"✅ Created dataloader with batch_size={batch_size}")
    
    # Generate translated fMRI
    print(f"\n🚀 Generating translated fMRI...")
    
    all_translated_fmri = []
    all_eeg_data = []
    all_labels = []
    all_stimulus_codes = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            eeg_data = batch['eeg_data'].to(device, non_blocking=True)
            target_fmri = batch['translated_fmri_target'].to(device, non_blocking=True)
            
            # Generate translated fMRI
            outputs = model(eeg_data, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            # Move to CPU and store
            all_translated_fmri.append(translated_fmri.cpu().numpy())
            all_eeg_data.append(eeg_data.cpu().numpy())
            
            # Store metadata
            for i in range(len(batch['stimulus_code'])):
                all_labels.append(samples[batch_idx * batch_size + i]['label'])
                all_stimulus_codes.append(batch['stimulus_code'][i])
            
            if batch_idx % 20 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(dataloader)} "
                      f"({(batch_idx+1)/len(dataloader)*100:.1f}%)")
    
    # Concatenate all results
    translated_fmri_array = np.concatenate(all_translated_fmri, axis=0)
    eeg_data_array = np.concatenate(all_eeg_data, axis=0)
    labels_array = np.array(all_labels)
    
    print(f"✅ Generated translated fMRI for {len(translated_fmri_array)} samples")
    print(f"  Translated fMRI shape: {translated_fmri_array.shape}")
    print(f"  EEG data shape: {eeg_data_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    
    # Save in CortexFlow format
    print(f"\n💾 Saving in CortexFlow format...")
    
    # Prepare data for CortexFlow
    # fmri: (N, 3092) float64
    # stim: (N, 784) uint8 (28x28 images)
    # labels: (N, 1) uint8
    
    cortexflow_data = {
        'fmri': translated_fmri_array.astype(np.float64),
        'labels': labels_array.reshape(-1, 1).astype(np.uint8)
    }
    
    # Create stimulus images (28x28 digit images)
    print(f"🖼️  Creating stimulus images...")
    stimulus_images = []
    
    for label in labels_array:
        # Create simple digit representation (28x28)
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Simple digit patterns (basic representations)
        if label == 0:  # Circle
            center = (14, 14)
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if 8 <= dist <= 12:
                        img[i, j] = 255
        elif label == 1:  # Vertical line
            img[4:24, 12:16] = 255
        elif label == 2:  # Horizontal lines
            img[8:12, 4:24] = 255
            img[16:20, 4:24] = 255
        elif label == 3:  # Curves
            img[6:10, 8:20] = 255
            img[12:16, 8:20] = 255
            img[18:22, 8:20] = 255
        elif label == 4:  # L shape
            img[6:22, 6:10] = 255
            img[12:16, 6:22] = 255
        elif label == 5:  # S shape
            img[6:10, 8:20] = 255
            img[10:14, 8:12] = 255
            img[14:18, 16:20] = 255
            img[18:22, 8:20] = 255
        elif label == 6:  # Partial circle
            center = (14, 14)
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if 8 <= dist <= 12 and i >= 14:
                        img[i, j] = 255
            img[14:18, 8:20] = 255
        elif label == 7:  # Diagonal
            for i in range(6, 22):
                j = int(6 + (i - 6) * 16 / 16)
                if 0 <= j < 28:
                    img[i, j:j+2] = 255
        elif label == 8:  # Double circle
            center = (14, 14)
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if 6 <= dist <= 8 or 10 <= dist <= 12:
                        img[i, j] = 255
        elif label == 9:  # Partial circle top
            center = (14, 14)
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if 8 <= dist <= 12 and i <= 14:
                        img[i, j] = 255
            img[10:18, 16:20] = 255
        
        stimulus_images.append(img.flatten())
    
    stimulus_array = np.array(stimulus_images, dtype=np.uint8)
    cortexflow_data['stim'] = stimulus_array
    
    print(f"✅ Created stimulus images: {stimulus_array.shape}")
    
    # Save as .mat file for CortexFlow
    mat_filename = output_path / "mindbigdata_translated_fmri.mat"
    sio.savemat(str(mat_filename), cortexflow_data)
    print(f"✅ Saved CortexFlow format: {mat_filename}")
    
    # Save individual components as numpy arrays
    np.save(output_path / "translated_fmri.npy", translated_fmri_array)
    np.save(output_path / "eeg_data.npy", eeg_data_array)
    np.save(output_path / "labels.npy", labels_array)
    np.save(output_path / "stimulus_images.npy", stimulus_array)
    
    print(f"✅ Saved individual arrays to {output_path}")
    
    # Save metadata
    metadata = {
        'total_samples': len(translated_fmri_array),
        'fmri_shape': translated_fmri_array.shape,
        'eeg_shape': eeg_data_array.shape,
        'stimulus_shape': stimulus_array.shape,
        'labels_shape': labels_array.shape,
        'label_distribution': {str(i): int(np.sum(labels_array == i)) for i in range(10)},
        'data_statistics': {
            'fmri_mean': float(np.mean(translated_fmri_array)),
            'fmri_std': float(np.std(translated_fmri_array)),
            'fmri_min': float(np.min(translated_fmri_array)),
            'fmri_max': float(np.max(translated_fmri_array))
        },
        'generation_info': {
            'model_type': 'NT-ViT Robust',
            'terminology': 'translated_fmri (EEG→fMRI translation)',
            'outliers_filtered': 'No (full dataset used for generation)',
            'batch_size': batch_size
        }
    }
    
    with open(output_path / "generation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata: {output_path / 'generation_metadata.json'}")
    
    # Summary
    print(f"\n📊 Generation Summary:")
    print(f"  Total samples: {len(translated_fmri_array)}")
    print(f"  fMRI data: {translated_fmri_array.shape} (mean: {np.mean(translated_fmri_array):.6f})")
    print(f"  Stimulus data: {stimulus_array.shape}")
    print(f"  Labels: {labels_array.shape}")
    print(f"  Output directory: {output_path}")
    
    print(f"\n🎯 Ready for CortexFlow!")
    print(f"  Use file: {mat_filename}")
    print(f"  Format: fmri=(N,3092), stim=(N,784), labels=(N,1)")
    
    return translated_fmri_array, stimulus_array, labels_array

def main():
    """Main function"""
    
    print("🧠 NT-ViT Translated fMRI Generation")
    print("=" * 50)
    print("📋 Using successfully trained robust model")
    print("📋 Terminology: translated_fmri (EEG→fMRI translation)")
    
    # Configuration
    model_path = "ntvit_robust_outputs/best_robust_model.pth"
    datasets_dir = "datasets"
    output_dir = "translated_fmri_outputs"
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: MindBigData (full dataset)")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output_dir}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load trained model
        model, checkpoint = load_trained_model(model_path, device)
        
        # Generate translated fMRI
        translated_fmri, stimulus_data, labels = generate_translated_fmri_dataset(
            model=model,
            datasets_dir=datasets_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            device=device
        )
        
        print(f"\n🎉 Translated fMRI generation completed successfully!")
        print(f"📊 Generated {len(translated_fmri)} samples")
        print(f"🚀 Ready for CortexFlow integration!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
