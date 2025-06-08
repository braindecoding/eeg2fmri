#!/usr/bin/env python3
"""
Generate Final CortexFlow Data - Direct Approach
================================================

Use existing data loaders directly and generate from final trained models
"""

import torch
import numpy as np
from pathlib import Path
import scipy.io as sio
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from train_ntvit import (
    NTViTEEGToFMRI, 
    MindBigDataLoader,
    CrellDataLoader
)

def load_real_stimulus_images(stimuli_dir: str, labels: np.ndarray, dataset_type: str):
    """Load REAL stimulus images from datasets folder"""
    
    print(f"🖼️  Loading REAL {dataset_type} stimulus images...")
    
    stimuli_path = Path(stimuli_dir)
    stimulus_images = []
    
    for label in labels:
        if dataset_type == "mindbigdata":
            image_path = stimuli_path / f"{label}.jpg"
        else:  # crell
            letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            letter = letters[label] if label < len(letters) else 'a'
            image_path = stimuli_path / f"{letter}.png"
        
        if image_path.exists():
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.uint8)
                stimulus_images.append(img_array.flatten())
            except Exception as e:
                stimulus_images.append(np.zeros(784, dtype=np.uint8))
        else:
            stimulus_images.append(np.zeros(784, dtype=np.uint8))
    
    stimulus_array = np.array(stimulus_images, dtype=np.uint8)
    print(f"✅ Loaded stimulus images: {stimulus_array.shape}")
    return stimulus_array

def generate_mindbigdata_final():
    """Generate MindBigData CortexFlow from final model"""
    
    print("🧠 Generating MindBigData from Final Model")
    print("=" * 60)
    
    # Load model
    model_path = "ntvit_robust_outputs/best_robust_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔄 Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = NTViTEEGToFMRI(eeg_channels=14).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Load data using existing loader
    print(f"📊 Loading MindBigData using existing loader...")
    mindbig_loader = MindBigDataLoader(
        filepath="datasets/EP1.01.txt",
        stimuli_dir="datasets/MindbigdataStimuli",
        max_samples=1200  # Limit for memory
    )
    
    samples = mindbig_loader.samples
    print(f"✅ Loaded {len(samples)} samples")
    
    # Generate fMRI
    print(f"🚀 Generating translated fMRI...")
    all_fmri = []
    all_labels = []
    
    with torch.no_grad():
        batch_size = 4
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            
            batch_eeg = []
            batch_labels = []
            
            for sample in batch_samples:
                eeg_data = torch.tensor(sample['eeg_data'], dtype=torch.float32)
                
                # Normalize (same as training)
                eeg_mean = eeg_data.mean()
                eeg_std = eeg_data.std()
                eeg_std = torch.clamp(eeg_std, min=1e-6)
                eeg_data = (eeg_data - eeg_mean) / eeg_std
                eeg_data = torch.clamp(eeg_data, -3.0, 3.0)
                
                batch_eeg.append(eeg_data)
                # Debug sample structure
                if i == 0 and len(batch_labels) == 0:
                    print(f"  Sample keys: {list(sample.keys())}")

                # Use correct key for digit
                digit = sample.get('stimulus_code', sample.get('digit', sample.get('label', 0)))
                batch_labels.append(digit)
            
            eeg_batch = torch.stack(batch_eeg).to(device)
            target_fmri = torch.randn(len(batch_samples), 3092) * 0.01
            target_fmri = torch.clamp(target_fmri, -1.0, 1.0).to(device)
            
            outputs = model(eeg_batch, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            all_fmri.append(translated_fmri.cpu().numpy())
            all_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 50 == 0:
                print(f"  Processed {i + len(batch_samples)}/{len(samples)} samples")
    
    fmri_array = np.concatenate(all_fmri, axis=0)
    labels_array = np.array(all_labels)
    
    print(f"✅ Generated fMRI: {fmri_array.shape}")
    print(f"  fMRI range: [{np.min(fmri_array):.6f}, {np.max(fmri_array):.6f}]")
    print(f"  fMRI mean: {np.mean(fmri_array):.6f}")
    
    # Load REAL stimulus images
    stimulus_array = load_real_stimulus_images(
        "datasets/MindbigdataStimuli", 
        labels_array, 
        "mindbigdata"
    )
    
    # Create train/test split
    print(f"📊 Creating stratified train/test split...")
    train_idx, test_idx = train_test_split(
        np.arange(len(fmri_array)),
        test_size=0.1,
        stratify=labels_array,
        random_state=42
    )
    
    print(f"✅ Split completed:")
    print(f"  Training: {len(train_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    # Create CortexFlow format
    cortexflow_data = {
        'fmriTrn': fmri_array[train_idx].astype(np.float64),
        'fmriTest': fmri_array[test_idx].astype(np.float64),
        'stimTrn': stimulus_array[train_idx].astype(np.uint8),
        'stimTest': stimulus_array[test_idx].astype(np.uint8),
        'labelTrn': labels_array[train_idx].reshape(-1, 1).astype(np.uint8),
        'labelTest': labels_array[test_idx].reshape(-1, 1).astype(np.uint8)
    }
    
    # Save
    output_dir = Path("cortexflow_outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "mindbigdata_final.mat"
    sio.savemat(str(output_file), cortexflow_data)
    
    print(f"✅ Saved: {output_file}")
    for key, value in cortexflow_data.items():
        print(f"  {key}: {value.shape} {value.dtype}")
    
    return cortexflow_data

def generate_crell_final():
    """Generate Crell CortexFlow from final model"""
    
    print("🧠 Generating Crell from Final Model")
    print("=" * 60)
    
    # Load model
    model_path = "crell_full_outputs/best_crell_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔄 Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = NTViTEEGToFMRI(eeg_channels=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Load data using existing loader
    print(f"📊 Loading Crell using existing loader...")
    crell_loader = CrellDataLoader(
        filepath="datasets/S01.mat",
        stimuli_dir="datasets/crellStimuli",
        max_samples=1000  # Same as training
    )
    
    samples = crell_loader.samples
    print(f"✅ Loaded {len(samples)} samples")
    
    # Generate fMRI
    print(f"🚀 Generating translated fMRI...")
    all_fmri = []
    all_labels = []
    
    # Letter to label mapping
    letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    letter_to_label = {letter: idx for idx, letter in enumerate(letters)}
    
    with torch.no_grad():
        batch_size = 2  # Smaller batch for 64 channels
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            
            batch_eeg = []
            batch_labels = []
            
            for sample in batch_samples:
                eeg_data = torch.tensor(sample['eeg_data'], dtype=torch.float32)
                
                # Normalize (same as training)
                eeg_mean = eeg_data.mean()
                eeg_std = eeg_data.std()
                eeg_std = torch.clamp(eeg_std, min=1e-6)
                eeg_data = (eeg_data - eeg_mean) / eeg_std
                eeg_data = torch.clamp(eeg_data, -3.0, 3.0)
                
                batch_eeg.append(eeg_data)
                
                # Convert letter to label
                letter = sample['letter']
                label = letter_to_label.get(letter, 0)
                batch_labels.append(label)
            
            eeg_batch = torch.stack(batch_eeg).to(device)
            target_fmri = torch.randn(len(batch_samples), 3092) * 0.01
            target_fmri = torch.clamp(target_fmri, -1.0, 1.0).to(device)
            
            outputs = model(eeg_batch, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            all_fmri.append(translated_fmri.cpu().numpy())
            all_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 50 == 0:
                print(f"  Processed {i + len(batch_samples)}/{len(samples)} samples")
    
    fmri_array = np.concatenate(all_fmri, axis=0)
    labels_array = np.array(all_labels)
    
    print(f"✅ Generated fMRI: {fmri_array.shape}")
    print(f"  fMRI range: [{np.min(fmri_array):.6f}, {np.max(fmri_array):.6f}]")
    print(f"  fMRI mean: {np.mean(fmri_array):.6f}")
    
    # Load REAL stimulus images
    stimulus_array = load_real_stimulus_images(
        "datasets/crellStimuli", 
        labels_array, 
        "crell"
    )
    
    # Create train/test split
    print(f"📊 Creating stratified train/test split...")
    train_idx, test_idx = train_test_split(
        np.arange(len(fmri_array)),
        test_size=0.1,
        stratify=labels_array,
        random_state=42
    )
    
    print(f"✅ Split completed:")
    print(f"  Training: {len(train_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    # Create CortexFlow format
    cortexflow_data = {
        'fmriTrn': fmri_array[train_idx].astype(np.float64),
        'fmriTest': fmri_array[test_idx].astype(np.float64),
        'stimTrn': stimulus_array[train_idx].astype(np.uint8),
        'stimTest': stimulus_array[test_idx].astype(np.uint8),
        'labelTrn': labels_array[train_idx].reshape(-1, 1).astype(np.uint8),
        'labelTest': labels_array[test_idx].reshape(-1, 1).astype(np.uint8)
    }
    
    # Save
    output_dir = Path("cortexflow_outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "crell_final.mat"
    sio.savemat(str(output_file), cortexflow_data)
    
    print(f"✅ Saved: {output_file}")
    for key, value in cortexflow_data.items():
        print(f"  {key}: {value.shape} {value.dtype}")
    
    return cortexflow_data

def main():
    """Main function"""
    
    print("🚀 Generate Final CortexFlow Data - Direct Approach")
    print("=" * 70)
    print("📋 Using existing data loaders and final trained models")
    
    try:
        # Generate MindBigData
        print(f"\n" + "="*70)
        mindbig_data = generate_mindbigdata_final()
        
        # Generate Crell
        print(f"\n" + "="*70)
        crell_data = generate_crell_final()
        
        print(f"\n🎉 Final generation completed!")
        print(f"📁 Files ready in cortexflow_outputs/")
        print(f"🚀 Ready for CortexFlow integration!")
        
        # Summary
        print(f"\n📊 Generation Summary:")
        print(f"  MindBigData: {mindbig_data['fmriTrn'].shape[0]} train + {mindbig_data['fmriTest'].shape[0]} test")
        print(f"  Crell: {crell_data['fmriTrn'].shape[0]} train + {crell_data['fmriTest'].shape[0]} test")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
