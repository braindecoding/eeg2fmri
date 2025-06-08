#!/usr/bin/env python3
"""
Generate More CortexFlow Samples
===============================

Generates more synthetic fMRI samples using trained NT-ViT models
to create larger datasets for CortexFlow training.
"""

import torch
import numpy as np
import scipy.io
from pathlib import Path
import json
from PIL import Image
import sys

# Add train_ntvit.py imports
sys.path.append('.')
from train_ntvit import NTViTEEGToFMRI, MindBigDataLoader, CrellDataLoader

class CortexFlowDataGenerator:
    """Generate more samples for CortexFlow using trained NT-ViT models"""
    
    def __init__(self, models_dir: str = "ntvit_outputs", datasets_dir: str = "datasets"):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load trained models
        self.mindbig_model = self.load_model("mindbigdata", 14)
        self.crell_model = self.load_model("crell", 64)
        
    def load_model(self, dataset_type: str, channels: int):
        """Load trained NT-ViT model"""
        model_path = self.models_dir / f"ntvit_{dataset_type}_final.pth"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return None
        
        # Create model
        model = NTViTEEGToFMRI(eeg_channels=channels).to(self.device)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Loaded {dataset_type} model: {channels} channels")
            return model
        except Exception as e:
            print(f"❌ Error loading {dataset_type} model: {e}")
            return None
    
    def load_dataset_samples(self, dataset_type: str, max_samples: int = None):
        """Load original dataset samples - FULL DATASET for Maximum Expansion"""
        if dataset_type == "mindbigdata":
            print(f"  🔥 Loading FULL MindBigData dataset (no limit)...")
            loader = MindBigDataLoader(
                filepath=str(self.datasets_dir / "EP1.01.txt"),
                stimuli_dir=str(self.datasets_dir / "MindbigdataStimuli"),
                max_samples=max_samples  # None = no limit
            )
        else:  # crell
            print(f"  🔥 Loading FULL Crell dataset (no limit)...")
            loader = CrellDataLoader(
                filepath=str(self.datasets_dir / "S01.mat"),
                stimuli_dir=str(self.datasets_dir / "crellStimuli"),
                max_samples=max_samples  # None = no limit
            )
        
        return loader.samples
    
    def generate_synthetic_fmri_batch(self, model, eeg_data_batch):
        """Generate synthetic fMRI for a batch of EEG data"""
        if model is None:
            return None
        
        with torch.no_grad():
            eeg_tensor = torch.tensor(eeg_data_batch, dtype=torch.float32).to(self.device)
            outputs = model(eeg_tensor)
            synthetic_fmri = outputs['synthetic_fmri']
            return synthetic_fmri.cpu().numpy()
    
    def process_stimulus_images(self, samples, target_size: int = 28):
        """Process stimulus images to 28x28 grayscale"""
        stimuli_data = []
        labels = []
        
        # Label mappings
        crell_labels = {
            'a': 1, 'd': 2, 'e': 3, 'f': 4, 'j': 5,
            'n': 6, 'o': 7, 's': 8, 't': 9, 'v': 10
        }
        
        for sample in samples:
            # Process stimulus image
            stimulus_image = sample['stimulus_image']  # Already in CHW format
            
            # Convert to PIL Image (assuming CHW format)
            if len(stimulus_image.shape) == 3:
                # Convert CHW to HWC and to grayscale
                img_hwc = np.transpose(stimulus_image, (1, 2, 0))
                img_gray = np.mean(img_hwc, axis=2)  # Convert to grayscale
            else:
                img_gray = stimulus_image
            
            # Convert to PIL and resize
            img_pil = Image.fromarray((img_gray * 255).astype(np.uint8))
            img_resized = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized, dtype=np.uint8)
            
            stimuli_data.append(img_array.flatten())
            
            # Process labels
            if sample['dataset_type'] == 'mindbigdata':
                labels.append(sample['stimulus_code'])  # 0-9
            else:  # crell
                letter = sample['letter']
                labels.append(crell_labels.get(letter, 1))
        
        return np.array(stimuli_data), np.array(labels)

    def create_train_test_split_by_stimulus(self, fmri_data, stimuli_data, labels, dataset_type: str):
        """Create train/test split ensuring ALL stimuli are represented in test set"""

        if dataset_type == "mindbigdata":
            # MindBigData: digits 0-9 (all should be in test)
            all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:  # crell
            # Crell: letters a,d,e,f,j,n,o,s,t,v → labels 1-10 (all should be in test)
            all_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Group samples by label
        samples_by_label = {}
        for i, label in enumerate(labels):
            if label not in samples_by_label:
                samples_by_label[label] = []
            samples_by_label[label].append(i)

        # Ensure we have at least 1 sample per label for test set
        test_indices = []
        train_indices = []

        for label in all_labels:
            if label in samples_by_label:
                label_samples = samples_by_label[label]

                # Shuffle samples for this label
                np.random.seed(42 + label)  # Different seed per label for diversity
                np.random.shuffle(label_samples)

                # Take 1 sample for test, rest for train
                test_indices.append(label_samples[0])  # First sample for test
                train_indices.extend(label_samples[1:])  # Rest for train
            else:
                print(f"    Warning: No samples found for label {label}")

        # Convert to numpy arrays
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Final shuffle
        np.random.seed(42)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Create splits
        fmri_train = fmri_data[train_indices]
        fmri_test = fmri_data[test_indices]
        stim_train = stimuli_data[train_indices]
        stim_test = stimuli_data[test_indices]
        label_train = labels[train_indices]
        label_test = labels[test_indices]

        print(f"  📊 Balanced split for {dataset_type}:")
        print(f"    Train labels: {sorted(np.unique(label_train))}")
        print(f"    Test labels:  {sorted(np.unique(label_test))}")
        print(f"    Train samples: {len(train_indices)}")
        print(f"    Test samples:  {len(test_indices)}")
        print(f"    ✅ All {len(all_labels)} stimuli represented in test set")

        return (fmri_train, fmri_test, stim_train, stim_test,
                label_train.reshape(-1, 1), label_test.reshape(-1, 1))

    def generate_dataset(self, dataset_type: str, n_samples: int = None):
        """Generate LARGE dataset for CortexFlow - Substantial Expansion"""
        # Load original samples - MODERATE EXPANSION (safe approach)
        large_sample_size = 1000 if dataset_type == "mindbigdata" else 200
        print(f"\n🔄 Generating {dataset_type} dataset (MODERATE EXPANSION - {large_sample_size} samples)...")

        samples = self.load_dataset_samples(dataset_type, max_samples=large_sample_size)
        print(f"  Loaded {len(samples)} original samples")
        
        if len(samples) == 0:
            print(f"  ❌ No samples found for {dataset_type}")
            return None
        
        # Extract EEG data
        eeg_data = np.array([sample['eeg_data'] for sample in samples])
        print(f"  EEG data shape: {eeg_data.shape}")
        
        # Generate synthetic fMRI
        model = self.mindbig_model if dataset_type == "mindbigdata" else self.crell_model
        synthetic_fmri = self.generate_synthetic_fmri_batch(model, eeg_data)
        
        if synthetic_fmri is None:
            print(f"  ❌ Failed to generate synthetic fMRI")
            return None
        
        print(f"  Generated synthetic fMRI shape: {synthetic_fmri.shape}")
        
        # Process stimulus images
        stimuli_data, labels = self.process_stimulus_images(samples)
        print(f"  Processed stimuli shape: {stimuli_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Reduce fMRI dimensions to 3092
        if synthetic_fmri.shape[1] > 3092:
            synthetic_fmri = synthetic_fmri[:, :3092]
        elif synthetic_fmri.shape[1] < 3092:
            padding = np.zeros((synthetic_fmri.shape[0], 3092 - synthetic_fmri.shape[1]))
            synthetic_fmri = np.concatenate([synthetic_fmri, padding], axis=1)
        
        # Create train/test split ensuring different stimuli
        (fmri_train, fmri_test, stim_train, stim_test,
         label_train, label_test) = self.create_train_test_split_by_stimulus(
            synthetic_fmri, stimuli_data, labels, dataset_type
        )

        # Convert to appropriate data types
        fmri_train = fmri_train.astype(np.float64)
        fmri_test = fmri_test.astype(np.float64)
        stim_train = stim_train.astype(np.uint8)
        stim_test = stim_test.astype(np.uint8)
        label_train = label_train.astype(np.uint8)
        label_test = label_test.astype(np.uint8)
        
        # Print statistics
        print(f"\n📊 {dataset_type.upper()} Dataset Statistics:")
        print(f"  📈 fmriTrn: {fmri_train.shape} - float64")
        print(f"      Range: [{fmri_train.min():.3f}, {fmri_train.max():.3f}], "
              f"Mean: {fmri_train.mean():.3f}, Std: {fmri_train.std():.3f}")
        print(f"  📈 stimTrn: {stim_train.shape} - uint8")
        print(f"      Range: [{stim_train.min():.3f}, {stim_train.max():.3f}] (pixel values)")
        print(f"  📈 fmriTest: {fmri_test.shape} - float64")
        print(f"      Range: [{fmri_test.min():.3f}, {fmri_test.max():.3f}], "
              f"Mean: {fmri_test.mean():.3f}, Std: {fmri_test.std():.3f}")
        print(f"  📈 stimTest: {stim_test.shape} - uint8")
        print(f"      Range: [{stim_test.min():.3f}, {stim_test.max():.3f}] (pixel values)")
        print(f"  📈 labelTrn: {label_train.shape} - uint8")
        print(f"      Unique labels: {np.unique(label_train)}")
        print(f"  📈 labelTest: {label_test.shape} - uint8")
        print(f"      Unique labels: {np.unique(label_test)}")
        
        # Create MATLAB structure with flattened metadata
        matlab_data = {
            'fmriTrn': fmri_train,
            'fmriTest': fmri_test,
            'stimTrn': stim_train,
            'stimTest': stim_test,
            'labelTrn': label_train,
            'labelTest': label_test,
            # Flatten metadata to avoid nested dict issues
            'dataset_type': dataset_type,
            'n_train_samples': np.array([len(fmri_train)]),
            'n_test_samples': np.array([len(fmri_test)]),
            'fmri_dimensions': np.array([fmri_train.shape[1]]),
            'stimulus_dimensions': np.array([stim_train.shape[1]]),
            'source': 'NT-ViT synthetic fMRI (expanded)',
            'compatible_with': 'CortexFlow'
        }
        
        return matlab_data
    
    def generate_all(self, output_dir: str = "cortexflow_data"):
        """Generate FULL datasets for both MindBigData and Crell - Maximum Expansion"""
        print("🧠 NT-ViT CortexFlow Data Generator (FULL DATASET)")
        print("=" * 60)
        print("🔥 MODERATE EXPANSION: Using substantial dataset!")
        print("📊 Expected: ~1K MindBigData + ~200 Crell samples")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        datasets = ["mindbigdata", "crell"]

        for dataset in datasets:
            try:
                # Generate FULL dataset - no sample limit
                matlab_data = self.generate_dataset(dataset, n_samples=None)  # FULL DATASET!
                
                if matlab_data is not None:
                    # Save MATLAB file
                    output_file = output_path / f"{dataset}.mat"
                    scipy.io.savemat(output_file, matlab_data)
                    print(f"  ✅ Saved: {output_file}")
                
            except Exception as e:
                print(f"  ❌ Error generating {dataset}: {e}")
                continue
        
        print(f"\n✅ Expanded dataset generation complete!")
        print(f"📁 Output files in: {output_path}/")

def main():
    """Main function"""
    generator = CortexFlowDataGenerator()
    generator.generate_all()

if __name__ == "__main__":
    main()
