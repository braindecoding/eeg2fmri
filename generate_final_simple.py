#!/usr/bin/env python3
"""
Generate Final CortexFlow Data - Simple Version
===============================================

Generate translated fMRI from final trained models without loading stimulus during data loading
"""

import torch
import numpy as np
from pathlib import Path
import json
import scipy.io as sio
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from train_ntvit import NTViTEEGToFMRI

def load_mindbigdata_eeg_only():
    """Load only EEG data from MindBigData without stimulus images"""

    print("📊 Loading MindBigData EEG data only...")

    filepath = "datasets/EP1.01.txt"
    samples = []

    # Use the same parsing logic as MindBigDataLoader
    signals_by_event = {}

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= 5000:  # Limit for memory
                    break

                if line_num % 1000 == 0:
                    print(f"  Processing line {line_num}...")

                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        event_id = parts[0]
                        signal_id = parts[1]
                        device = parts[2]
                        channel = parts[3]
                        digit_code = int(parts[4])
                        signal_size = int(parts[5])
                        data_str = parts[6]

                        # Focus on EPOC device (14 channels)
                        if device == "EP" and 0 <= digit_code <= 9:
                            try:
                                data_values = [float(x) for x in data_str.split(',')]

                                if len(data_values) == signal_size:
                                    if event_id not in signals_by_event:
                                        signals_by_event[event_id] = {}

                                    signals_by_event[event_id][channel] = {
                                        'code': digit_code,
                                        'data': np.array(data_values),
                                        'size': signal_size
                                    }

                                    if line_num < 10:  # Debug first few lines
                                        print(f"    Found EP data: event={event_id}, channel={channel}, digit={digit_code}, size={signal_size}")
                            except ValueError as e:
                                if line_num < 10:
                                    print(f"    ValueError: {e}")
                                continue
                        elif line_num < 10:  # Debug first few lines
                            print(f"    Skipped: device={device}, digit={digit_code}")

                except Exception as e:
                    continue

        # Process events to create samples
        epoc_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

        print(f"  Found {len(signals_by_event)} events")

        event_count = 0
        for event_id, channels_data in signals_by_event.items():
            if event_count < 10:  # Debug first 10 events
                print(f"  Event {event_id}: {len(channels_data)} channels")
            event_count += 1

            if len(channels_data) == 14:  # All EPOC channels present
                print(f"  Processing event {event_id} with {len(channels_data)} channels")
                channel_signals = []
                digit_code = None

                for channel in epoc_channels:
                    if channel in channels_data:
                        sample = channels_data[channel]
                        channel_signals.append(sample['data'])
                        if digit_code is None:
                            digit_code = sample['code']
                    else:
                        break

                if len(channel_signals) == 14:
                    # Fixed length processing - standardize to 256 samples
                    target_length = 256
                    processed_signals = []

                    for signal in channel_signals:
                        if len(signal) >= target_length:
                            processed_signals.append(signal[:target_length])
                        else:
                            # Pad with last value
                            padded = np.full(target_length, signal[-1] if len(signal) > 0 else 0.0)
                            padded[:len(signal)] = signal
                            processed_signals.append(padded)

                    eeg_array = np.array(processed_signals)  # (14, 256)

                    samples.append({
                        'digit': digit_code,
                        'eeg_data': eeg_array
                    })

                    if len(samples) % 100 == 0:
                        print(f"  Loaded {len(samples)} samples...")

                    if len(samples) >= 1200:  # Limit samples
                        break

    except Exception as e:
        print(f"Error loading MindBigData: {e}")
        return []

    print(f"✅ Loaded {len(samples)} MindBigData EEG samples")
    return samples

def load_crell_eeg_only():
    """Load only EEG data from Crell without stimulus images"""
    
    print("📊 Loading Crell EEG data only...")
    
    filepath = "datasets/S01.mat"
    
    try:
        mat_data = sio.loadmat(filepath)
        
        # Extract EEG data and events
        eeg_data = mat_data['EEG']['data'][0, 0]  # Shape: (64, timepoints)
        events = mat_data['EEG']['event'][0, 0]
        
        # Extract letter events (codes 100+)
        samples = []
        letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
        
        for i, event in enumerate(events):
            if i >= 1000:  # Limit for memory
                break
                
            try:
                event_type = event[0][0, 0] if len(event[0]) > 0 else None
                latency = event[1][0, 0] if len(event[1]) > 0 else None
                
                if event_type and event_type >= 100:
                    # Letter event
                    letter_code = event_type - 100
                    if 97 <= letter_code <= 122:  # ASCII a-z
                        letter = chr(letter_code)
                        if letter in letters:
                            # Extract EEG segment (2250 timepoints = 4.5 seconds at 500Hz)
                            start_idx = max(0, latency - 1125)
                            end_idx = start_idx + 2250
                            
                            if end_idx <= eeg_data.shape[1]:
                                eeg_segment = eeg_data[:, start_idx:end_idx]  # (64, 2250)
                                
                                label = letters.index(letter)
                                samples.append({
                                    'letter': letter,
                                    'label': label,
                                    'eeg_data': eeg_segment
                                })
                                
                                if len(samples) % 50 == 0:
                                    print(f"  Loaded {len(samples)} samples...")
                                    
            except Exception as e:
                continue
        
        print(f"✅ Loaded {len(samples)} Crell EEG samples")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading Crell data: {e}")
        return []

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
    
    # Load EEG data only
    samples = load_mindbigdata_eeg_only()
    
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
                
                # Normalize
                eeg_mean = eeg_data.mean()
                eeg_std = eeg_data.std()
                eeg_std = torch.clamp(eeg_std, min=1e-6)
                eeg_data = (eeg_data - eeg_mean) / eeg_std
                eeg_data = torch.clamp(eeg_data, -3.0, 3.0)
                
                batch_eeg.append(eeg_data)
                batch_labels.append(sample['digit'])
            
            eeg_batch = torch.stack(batch_eeg).to(device)
            target_fmri = torch.randn(len(batch_samples), 3092) * 0.01
            target_fmri = torch.clamp(target_fmri, -1.0, 1.0).to(device)
            
            outputs = model(eeg_batch, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            all_fmri.append(translated_fmri.cpu().numpy())
            all_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 20 == 0:
                print(f"  Processed {i + len(batch_samples)}/{len(samples)} samples")
    
    fmri_array = np.concatenate(all_fmri, axis=0)
    labels_array = np.array(all_labels)
    
    print(f"✅ Generated fMRI: {fmri_array.shape}")
    
    # Load REAL stimulus images
    stimulus_array = load_real_stimulus_images(
        "datasets/MindbigdataStimuli", 
        labels_array, 
        "mindbigdata"
    )
    
    # Create train/test split
    print(f"📊 Creating train/test split...")
    train_idx, test_idx = train_test_split(
        np.arange(len(fmri_array)),
        test_size=0.1,
        stratify=labels_array,
        random_state=42
    )
    
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
        print(f"  {key}: {value.shape}")
    
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
    
    # Load EEG data only
    samples = load_crell_eeg_only()
    
    # Generate fMRI (similar to MindBigData)
    print(f"🚀 Generating translated fMRI...")
    all_fmri = []
    all_labels = []
    
    with torch.no_grad():
        batch_size = 2  # Smaller batch for 64 channels
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            
            batch_eeg = []
            batch_labels = []
            
            for sample in batch_samples:
                eeg_data = torch.tensor(sample['eeg_data'], dtype=torch.float32)
                
                # Normalize
                eeg_mean = eeg_data.mean()
                eeg_std = eeg_data.std()
                eeg_std = torch.clamp(eeg_std, min=1e-6)
                eeg_data = (eeg_data - eeg_mean) / eeg_std
                eeg_data = torch.clamp(eeg_data, -3.0, 3.0)
                
                batch_eeg.append(eeg_data)
                batch_labels.append(sample['label'])
            
            eeg_batch = torch.stack(batch_eeg).to(device)
            target_fmri = torch.randn(len(batch_samples), 3092) * 0.01
            target_fmri = torch.clamp(target_fmri, -1.0, 1.0).to(device)
            
            outputs = model(eeg_batch, target_fmri)
            translated_fmri = outputs['translated_fmri']
            
            all_fmri.append(translated_fmri.cpu().numpy())
            all_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 20 == 0:
                print(f"  Processed {i + len(batch_samples)}/{len(samples)} samples")
    
    fmri_array = np.concatenate(all_fmri, axis=0)
    labels_array = np.array(all_labels)
    
    print(f"✅ Generated fMRI: {fmri_array.shape}")
    
    # Load REAL stimulus images
    stimulus_array = load_real_stimulus_images(
        "datasets/crellStimuli", 
        labels_array, 
        "crell"
    )
    
    # Create train/test split
    print(f"📊 Creating train/test split...")
    train_idx, test_idx = train_test_split(
        np.arange(len(fmri_array)),
        test_size=0.1,
        stratify=labels_array,
        random_state=42
    )
    
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
        print(f"  {key}: {value.shape}")
    
    return cortexflow_data

def main():
    """Main function"""
    
    print("🚀 Generate Final CortexFlow Data - Simple Version")
    print("=" * 70)
    
    try:
        # Generate MindBigData
        print(f"\n" + "="*70)
        mindbig_data = generate_mindbigdata_final()
        
        # Generate Crell
        print(f"\n" + "="*70)
        crell_data = generate_crell_final()
        
        print(f"\n🎉 Final generation completed!")
        print(f"📁 Files ready in cortexflow_outputs/")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
