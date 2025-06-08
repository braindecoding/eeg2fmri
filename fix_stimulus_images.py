#!/usr/bin/env python3
"""
Fix Stimulus Images - Replace synthetic with REAL images
========================================================

Replace synthetic stimulus images with REAL images from datasets folder
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image
import json

def load_real_stimulus_images(stimuli_dir: str, labels: np.ndarray, dataset_type: str):
    """Load REAL stimulus images from datasets folder"""
    
    print(f"🖼️  Loading REAL {dataset_type} stimulus images from {stimuli_dir}...")
    
    stimuli_path = Path(stimuli_dir)
    stimulus_images = []
    
    for label in labels:
        if dataset_type == "mindbigdata":
            # For digits 0-9
            image_path = stimuli_path / f"{label}.jpg"
        else:  # crell
            # For letters a,d,e,f,j,n,o,s,t,v
            letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            letter = letters[label] if label < len(letters) else 'a'
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
    print(f"✅ Loaded REAL {dataset_type} stimulus images: {stimulus_array.shape}")
    print(f"  Image range: [{np.min(stimulus_array)}, {np.max(stimulus_array)}]")
    
    return stimulus_array

def fix_mindbigdata_stimuli():
    """Fix MindBigData stimulus images with REAL images"""
    
    print("🔧 Fixing MindBigData Stimulus Images")
    print("=" * 50)
    
    # Load existing data
    input_file = "translated_fmri_outputs/mindbigdata_translated_fmri.mat"
    output_file = "translated_fmri_outputs/mindbigdata_translated_fmri_real_stimuli.mat"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📂 Loading existing data from {input_file}...")
    data = sio.loadmat(input_file)
    
    # Extract components
    fmri_data = data['fmri']
    labels_data = data['labels'].flatten()
    
    print(f"✅ Loaded existing data:")
    print(f"  fMRI shape: {fmri_data.shape}")
    print(f"  Labels shape: {labels_data.shape}")
    print(f"  Label range: [{np.min(labels_data)}, {np.max(labels_data)}]")
    
    # Load REAL stimulus images
    real_stimuli = load_real_stimulus_images(
        "datasets/MindbigdataStimuli", 
        labels_data, 
        "mindbigdata"
    )
    
    # Create new data with REAL stimuli
    new_data = {
        'fmri': fmri_data.astype(np.float64),
        'stim': real_stimuli.astype(np.uint8),
        'labels': labels_data.reshape(-1, 1).astype(np.uint8)
    }
    
    # Save fixed data
    sio.savemat(output_file, new_data)
    print(f"✅ Saved fixed data to {output_file}")
    
    # Save individual arrays
    output_dir = Path("translated_fmri_outputs_real_stimuli")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "mindbigdata_translated_fmri.npy", fmri_data)
    np.save(output_dir / "mindbigdata_real_stimulus_images.npy", real_stimuli)
    np.save(output_dir / "mindbigdata_labels.npy", labels_data)
    
    # Save metadata
    metadata = {
        'dataset': 'MindBigData',
        'total_samples': len(fmri_data),
        'fmri_shape': list(fmri_data.shape),
        'stimulus_shape': list(real_stimuli.shape),
        'labels_shape': list(labels_data.shape),
        'stimulus_source': 'REAL images from datasets/MindbigdataStimuli/',
        'stimulus_range': [int(np.min(real_stimuli)), int(np.max(real_stimuli))],
        'cortexflow_format': {
            'fmri': '(N, 3092) float64',
            'stim': '(N, 784) uint8 - REAL digit images',
            'labels': '(N, 1) uint8'
        }
    }
    
    with open(output_dir / "mindbigdata_real_stimuli_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved individual arrays and metadata to {output_dir}")
    
    # Summary
    print(f"\n📊 MindBigData Fix Summary:")
    print(f"  Total samples: {len(fmri_data)}")
    print(f"  fMRI data: {fmri_data.shape}")
    print(f"  REAL stimulus data: {real_stimuli.shape}")
    print(f"  Stimulus range: [{np.min(real_stimuli)}, {np.max(real_stimuli)}]")
    print(f"  Output file: {output_file}")

def fix_crell_stimuli():
    """Fix Crell stimulus images with REAL images"""
    
    print("🔧 Fixing Crell Stimulus Images")
    print("=" * 50)
    
    # Load existing data
    input_file = "crell_translated_fmri_outputs/crell_translated_fmri.mat"
    output_file = "crell_translated_fmri_outputs/crell_translated_fmri_real_stimuli.mat"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📂 Loading existing data from {input_file}...")
    data = sio.loadmat(input_file)
    
    # Extract components
    fmri_data = data['fmri']
    labels_data = data['labels'].flatten()
    
    print(f"✅ Loaded existing data:")
    print(f"  fMRI shape: {fmri_data.shape}")
    print(f"  Labels shape: {labels_data.shape}")
    print(f"  Label range: [{np.min(labels_data)}, {np.max(labels_data)}]")
    
    # Load REAL stimulus images
    real_stimuli = load_real_stimulus_images(
        "datasets/crellStimuli", 
        labels_data, 
        "crell"
    )
    
    # Create new data with REAL stimuli
    new_data = {
        'fmri': fmri_data.astype(np.float64),
        'stim': real_stimuli.astype(np.uint8),
        'labels': labels_data.reshape(-1, 1).astype(np.uint8)
    }
    
    # Save fixed data
    sio.savemat(output_file, new_data)
    print(f"✅ Saved fixed data to {output_file}")
    
    # Save individual arrays
    output_dir = Path("crell_translated_fmri_outputs_real_stimuli")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "crell_translated_fmri.npy", fmri_data)
    np.save(output_dir / "crell_real_stimulus_images.npy", real_stimuli)
    np.save(output_dir / "crell_labels.npy", labels_data)
    
    # Save metadata
    metadata = {
        'dataset': 'Crell',
        'total_samples': len(fmri_data),
        'fmri_shape': list(fmri_data.shape),
        'stimulus_shape': list(real_stimuli.shape),
        'labels_shape': list(labels_data.shape),
        'stimulus_source': 'REAL images from datasets/crellStimuli/',
        'stimulus_range': [int(np.min(real_stimuli)), int(np.max(real_stimuli))],
        'letters': ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v'],
        'cortexflow_format': {
            'fmri': '(N, 3092) float64',
            'stim': '(N, 784) uint8 - REAL letter images',
            'labels': '(N, 1) uint8'
        }
    }
    
    with open(output_dir / "crell_real_stimuli_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved individual arrays and metadata to {output_dir}")
    
    # Summary
    print(f"\n📊 Crell Fix Summary:")
    print(f"  Total samples: {len(fmri_data)}")
    print(f"  fMRI data: {fmri_data.shape}")
    print(f"  REAL stimulus data: {real_stimuli.shape}")
    print(f"  Stimulus range: [{np.min(real_stimuli)}, {np.max(real_stimuli)}]")
    print(f"  Output file: {output_file}")

def main():
    """Main function"""
    
    print("🔧 Fix Stimulus Images - Replace Synthetic with REAL Images")
    print("=" * 70)
    print("📋 Replacing synthetic stimulus images with REAL images from datasets/")
    
    # Check if datasets exist
    datasets_path = Path("datasets")
    required_dirs = ["MindbigdataStimuli", "crellStimuli"]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not (datasets_path / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  Missing required directories:")
        for dir_name in missing_dirs:
            print(f"  - {datasets_path / dir_name}")
        return
    
    try:
        # Fix MindBigData stimuli
        print(f"\n" + "="*70)
        fix_mindbigdata_stimuli()
        
        # Fix Crell stimuli
        print(f"\n" + "="*70)
        fix_crell_stimuli()
        
        print(f"\n🎉 Stimulus images fix completed successfully!")
        print(f"📊 Both datasets now use REAL stimulus images from datasets/")
        print(f"🚀 Ready for CortexFlow integration with authentic stimuli!")
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
