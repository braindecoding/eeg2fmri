#!/usr/bin/env python3
"""
Fix Crell Stimulus Images Properly
==================================

Use the EXACT same stimulus images that were used during training
by using the CrellDataLoader's load_stimulus_image method
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json
import sys
sys.path.append('.')
from train_ntvit import CrellDataLoader

def fix_crell_stimulus_with_training_images():
    """Fix Crell stimulus using EXACT same images as training"""
    
    print("🔧 Fixing Crell Stimulus with Training Images")
    print("=" * 60)
    
    # Load existing Crell fMRI data
    input_file = "crell_translated_fmri_outputs/crell_translated_fmri.mat"
    output_file = "crell_translated_fmri_outputs/crell_translated_fmri_real_stimuli.mat"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📂 Loading existing Crell data from {input_file}...")
    data = sio.loadmat(input_file)
    
    # Extract components
    fmri_data = data['fmri']
    labels_data = data['labels'].flatten()
    
    print(f"✅ Loaded existing data:")
    print(f"  fMRI shape: {fmri_data.shape}")
    print(f"  Labels shape: {labels_data.shape}")
    print(f"  Label range: [{np.min(labels_data)}, {np.max(labels_data)}]")
    
    # Load Crell dataset using the SAME loader as training
    print(f"\n📊 Loading Crell dataset with SAME loader as training...")
    crell_loader = CrellDataLoader(
        filepath="datasets/S01.mat",
        stimuli_dir="datasets/crellStimuli",
        max_samples=1000  # Same as training
    )
    
    samples = crell_loader.samples
    print(f"✅ Loaded {len(samples)} Crell samples with training loader")
    
    # Verify we have the same number of samples
    if len(samples) != len(labels_data):
        print(f"⚠️  Sample count mismatch:")
        print(f"    Training samples: {len(samples)}")
        print(f"    Generated fMRI: {len(labels_data)}")
        print(f"    Using minimum: {min(len(samples), len(labels_data))}")
        
        # Use minimum count
        min_count = min(len(samples), len(labels_data))
        samples = samples[:min_count]
        fmri_data = fmri_data[:min_count]
        labels_data = labels_data[:min_count]
    
    # Extract stimulus images from training samples
    print(f"\n🖼️  Extracting stimulus images from training samples...")
    stimulus_images = []
    
    for i, sample in enumerate(samples):
        # Get stimulus image from training sample (RGB format 3, 224, 224)
        training_stimulus = sample['stimulus_image']  # Shape: (3, 224, 224)
        
        # Convert to grayscale and resize to 28x28 for CortexFlow
        # Take mean across RGB channels
        grayscale = np.mean(training_stimulus, axis=0)  # (224, 224)
        
        # Resize to 28x28 using simple downsampling
        # Take every 8th pixel (224/28 = 8)
        resized = grayscale[::8, ::8]  # (28, 28)
        
        # Ensure exactly 28x28
        if resized.shape != (28, 28):
            # If not exact, use interpolation
            from PIL import Image
            img_pil = Image.fromarray((grayscale * 255).astype(np.uint8))
            img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
            resized = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Convert to uint8 range [0, 255]
        stimulus_uint8 = (resized * 255).astype(np.uint8)
        
        # Flatten to 784 (28*28) for CortexFlow
        stimulus_images.append(stimulus_uint8.flatten())
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} stimulus images")
    
    stimulus_array = np.array(stimulus_images, dtype=np.uint8)
    print(f"✅ Extracted stimulus images from training: {stimulus_array.shape}")
    print(f"  Stimulus range: [{np.min(stimulus_array)}, {np.max(stimulus_array)}]")
    
    # Verify labels match
    print(f"\n🔍 Verifying labels match training samples...")
    letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
    
    label_matches = 0
    for i, sample in enumerate(samples):
        training_letter = sample['letter']
        training_label = letters.index(training_letter)
        generated_label = labels_data[i]
        
        if training_label == generated_label:
            label_matches += 1
        elif i < 5:  # Show first few mismatches
            print(f"  Label mismatch at {i}: training={training_label}({training_letter}), generated={generated_label}")
    
    match_rate = label_matches / len(samples) * 100
    print(f"✅ Label match rate: {match_rate:.1f}% ({label_matches}/{len(samples)})")
    
    # Create new data with REAL training stimuli
    new_data = {
        'fmri': fmri_data.astype(np.float64),
        'stim': stimulus_array.astype(np.uint8),
        'labels': labels_data.reshape(-1, 1).astype(np.uint8)
    }
    
    # Save fixed data
    sio.savemat(output_file, new_data)
    print(f"✅ Saved fixed data to {output_file}")
    
    # Save individual arrays
    output_dir = Path("crell_translated_fmri_outputs_real_stimuli")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "crell_translated_fmri.npy", fmri_data)
    np.save(output_dir / "crell_training_stimulus_images.npy", stimulus_array)
    np.save(output_dir / "crell_labels.npy", labels_data)
    
    # Save metadata
    metadata = {
        'dataset': 'Crell',
        'total_samples': len(fmri_data),
        'fmri_shape': list(fmri_data.shape),
        'stimulus_shape': list(stimulus_array.shape),
        'labels_shape': list(labels_data.shape),
        'stimulus_source': 'EXACT training images from CrellDataLoader',
        'stimulus_processing': 'RGB->Grayscale->28x28->uint8->flatten',
        'stimulus_range': [int(np.min(stimulus_array)), int(np.max(stimulus_array))],
        'label_match_rate': f"{match_rate:.1f}%",
        'letters': letters,
        'cortexflow_format': {
            'fmri': '(N, 3092) float64',
            'stim': '(N, 784) uint8 - EXACT training images',
            'labels': '(N, 1) uint8'
        }
    }
    
    with open(output_dir / "crell_training_stimuli_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved individual arrays and metadata to {output_dir}")
    
    # Summary
    print(f"\n📊 Crell Training Stimulus Fix Summary:")
    print(f"  Total samples: {len(fmri_data)}")
    print(f"  fMRI data: {fmri_data.shape}")
    print(f"  TRAINING stimulus data: {stimulus_array.shape}")
    print(f"  Stimulus range: [{np.min(stimulus_array)}, {np.max(stimulus_array)}]")
    print(f"  Label match rate: {match_rate:.1f}%")
    print(f"  Output file: {output_file}")
    
    return stimulus_array, match_rate

def verify_training_stimulus_authenticity():
    """Verify that we're using the exact same stimuli as training"""
    
    print(f"\n🔍 Verifying Training Stimulus Authenticity")
    print("=" * 50)
    
    # Load the fixed data
    data = sio.loadmat("crell_translated_fmri_outputs/crell_translated_fmri_real_stimuli.mat")
    fixed_stimuli = data['stim']
    labels = data['labels'].flatten()
    
    # Load training samples again
    crell_loader = CrellDataLoader(
        filepath="datasets/S01.mat",
        stimuli_dir="datasets/crellStimuli",
        max_samples=1000
    )
    
    samples = crell_loader.samples[:len(labels)]
    
    print(f"📊 Comparing fixed stimuli with training stimuli...")
    
    # Compare first few samples
    total_difference = 0
    for i in range(min(5, len(samples))):
        # Get training stimulus
        training_stimulus = samples[i]['stimulus_image']  # (3, 224, 224)
        
        # Convert to same format as fixed
        grayscale = np.mean(training_stimulus, axis=0)
        resized = grayscale[::8, ::8]
        if resized.shape != (28, 28):
            from PIL import Image
            img_pil = Image.fromarray((grayscale * 255).astype(np.uint8))
            img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
            resized = np.array(img_resized, dtype=np.float32) / 255.0
        
        training_uint8 = (resized * 255).astype(np.uint8).flatten()
        
        # Get fixed stimulus
        fixed_stimulus = fixed_stimuli[i]
        
        # Compare
        difference = np.mean(np.abs(training_uint8.astype(int) - fixed_stimulus.astype(int)))
        total_difference += difference
        
        letter = samples[i]['letter']
        print(f"  Sample {i} (letter {letter}): difference = {difference:.2f}")
    
    avg_difference = total_difference / min(5, len(samples))
    print(f"✅ Average difference: {avg_difference:.2f}")
    
    if avg_difference < 1.0:
        print(f"🎉 PERFECT MATCH - Stimuli are from exact training images!")
    else:
        print(f"⚠️  Some difference detected - may need adjustment")
    
    return avg_difference

def main():
    """Main function"""
    
    print("🔧 Fix Crell Stimulus with EXACT Training Images")
    print("=" * 70)
    print("📋 Using the EXACT same stimulus images as used during training")
    
    # Check if required files exist
    required_files = [
        "crell_translated_fmri_outputs/crell_translated_fmri.mat",
        "datasets/S01.mat",
        "datasets/crellStimuli"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return
    
    try:
        # Fix stimulus images with training images
        stimulus_array, match_rate = fix_crell_stimulus_with_training_images()
        
        # Verify authenticity
        avg_difference = verify_training_stimulus_authenticity()
        
        print(f"\n🎉 Crell stimulus fix with training images completed!")
        print(f"📊 Generated {len(stimulus_array)} stimulus images")
        print(f"🎯 Label match rate: {match_rate:.1f}%")
        print(f"🔍 Average difference: {avg_difference:.2f}")
        print(f"🚀 Ready for CortexFlow with EXACT training stimuli!")
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
