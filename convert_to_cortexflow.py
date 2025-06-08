#!/usr/bin/env python3
"""
NT-ViT to CortexFlow Format Converter
====================================

Converts NT-ViT outputs to CortexFlow-compatible MATLAB format:
- fmriTrn/fmriTest: Synthetic fMRI data
- stimTrn/stimTest: Stimulus images (flattened)
- labelTrn/labelTest: Labels (digit/letter codes)

Output format matches CortexFlow requirements:
- mindbigdata.m (digits 0-9)
- crell.m (letters a,d,e,f,j,n,o,s,t,v)
"""

import numpy as np
import scipy.io
from pathlib import Path
import json
from PIL import Image
import torch
from collections import defaultdict
import argparse

class NTViTToCortexFlowConverter:
    """Convert NT-ViT outputs to CortexFlow format"""
    
    def __init__(self, ntvit_outputs_dir: str, datasets_dir: str, output_dir: str = "cortexflow_data"):
        self.ntvit_outputs_dir = Path(ntvit_outputs_dir)
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset mappings
        self.mindbig_labels = {i: i for i in range(10)}  # 0-9 digits
        self.crell_labels = {
            'a': 1, 'd': 2, 'e': 3, 'f': 4, 'j': 5,
            'n': 6, 'o': 7, 's': 8, 't': 9, 'v': 10
        }
        
    def load_synthetic_fmri_data(self, dataset_type: str):
        """Load all synthetic fMRI files for a dataset"""
        fmri_files = list(self.ntvit_outputs_dir.glob(f"{dataset_type}_synthetic_fmri_*.npy"))
        fmri_files.sort()
        
        fmri_data = []
        metadata = []
        
        for fmri_file in fmri_files:
            # Load fMRI data
            fmri = np.load(fmri_file)
            fmri_data.append(fmri)
            
            # Load metadata
            json_file = fmri_file.with_suffix('.json')
            if json_file.exists():
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    metadata.append(meta)
        
        return np.array(fmri_data), metadata
    
    def load_stimulus_images(self, dataset_type: str, target_size: int = 28):
        """Load and process stimulus images"""
        if dataset_type == "mindbigdata":
            stimuli_dir = self.datasets_dir / "MindbigdataStimuli"
            image_files = [stimuli_dir / f"{i}.jpg" for i in range(10)]
            labels = list(range(10))
        else:  # crell
            stimuli_dir = self.datasets_dir / "crellStimuli"
            letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            image_files = [stimuli_dir / f"{letter}.png" for letter in letters]
            labels = [self.crell_labels[letter] for letter in letters]
        
        stimuli_data = []
        valid_labels = []
        
        for img_file, label in zip(image_files, labels):
            if img_file.exists():
                try:
                    # Load and process image
                    img = Image.open(img_file).convert('L')  # Convert to grayscale
                    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.uint8)
                    
                    # Flatten image (28x28 -> 784 for MNIST-like format)
                    img_flat = img_array.flatten()
                    stimuli_data.append(img_flat)
                    valid_labels.append(label)
                    
                except Exception as e:
                    print(f"Warning: Could not load {img_file}: {e}")
                    continue
        
        return np.array(stimuli_data), np.array(valid_labels)
    
    def create_train_test_split(self, fmri_data, stimuli_data, labels, train_ratio: float = 0.8):
        """Create train/test split"""
        n_samples = len(fmri_data)
        n_train = int(n_samples * train_ratio)
        
        # Random shuffle with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Create train/test splits
        fmri_train = fmri_data[train_indices]
        fmri_test = fmri_data[test_indices]
        
        stim_train = stimuli_data[train_indices] if len(stimuli_data) == n_samples else stimuli_data
        stim_test = stimuli_data[test_indices] if len(stimuli_data) == n_samples else stimuli_data
        
        label_train = labels[train_indices] if len(labels) == n_samples else labels
        label_test = labels[test_indices] if len(labels) == n_samples else labels
        
        return (fmri_train, fmri_test, stim_train, stim_test, 
                label_train.reshape(-1, 1), label_test.reshape(-1, 1))
    
    def reduce_fmri_dimensions(self, fmri_data, target_dim: int = 3092):
        """Reduce fMRI dimensions to match CortexFlow format"""
        if fmri_data.shape[1] > target_dim:
            # Use PCA-like reduction or simple truncation
            # For simplicity, we'll use the first target_dim dimensions
            return fmri_data[:, :target_dim]
        elif fmri_data.shape[1] < target_dim:
            # Pad with zeros if needed
            padding = np.zeros((fmri_data.shape[0], target_dim - fmri_data.shape[1]))
            return np.concatenate([fmri_data, padding], axis=1)
        else:
            return fmri_data
    
    def print_dataset_stats(self, name, data, data_type="float64"):
        """Print dataset statistics in CortexFlow format"""
        print(f"  📈 {name}: {data.shape} - {data_type}")
        if data_type == "float64":
            print(f"      Range: [{data.min():.3f}, {data.max():.3f}], "
                  f"Mean: {data.mean():.3f}, Std: {data.std():.3f}")
        else:
            print(f"      Range: [{data.min():.3f}, {data.max():.3f}] "
                  f"(normalized pixel values)")
    
    def convert_dataset(self, dataset_type: str):
        """Convert a single dataset to CortexFlow format"""
        print(f"\n🔄 Converting {dataset_type} dataset...")
        
        # Load synthetic fMRI data
        fmri_data, metadata = self.load_synthetic_fmri_data(dataset_type)
        print(f"  Loaded {len(fmri_data)} synthetic fMRI samples")
        
        # Load stimulus images
        stimuli_data, labels = self.load_stimulus_images(dataset_type)
        print(f"  Loaded {len(stimuli_data)} stimulus images")
        
        # Ensure we have matching samples
        min_samples = min(len(fmri_data), len(stimuli_data))
        if len(fmri_data) != len(stimuli_data):
            print(f"  Warning: Mismatched samples. Using first {min_samples} samples.")
            fmri_data = fmri_data[:min_samples]
            stimuli_data = stimuli_data[:min_samples]
            labels = labels[:min_samples]
        
        # Reduce fMRI dimensions to match CortexFlow format
        fmri_data = self.reduce_fmri_dimensions(fmri_data, target_dim=3092)
        
        # Create train/test split
        (fmri_train, fmri_test, stim_train, stim_test, 
         label_train, label_test) = self.create_train_test_split(
            fmri_data, stimuli_data, labels
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
        self.print_dataset_stats("fmriTrn", fmri_train, "float64")
        self.print_dataset_stats("stimTrn", stim_train, "uint8")
        self.print_dataset_stats("fmriTest", fmri_test, "float64")
        self.print_dataset_stats("stimTest", stim_test, "uint8")
        self.print_dataset_stats("labelTrn", label_train, "uint8")
        self.print_dataset_stats("labelTest", label_test, "uint8")
        
        print(f"  📈 labelTrn: Unique labels: {np.unique(label_train)}")
        print(f"  📈 labelTest: Unique labels: {np.unique(label_test)}")
        
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
            'source': 'NT-ViT synthetic fMRI',
            'compatible_with': 'CortexFlow'
        }
        
        # Save as MATLAB file
        output_file = self.output_dir / f"{dataset_type}.mat"
        scipy.io.savemat(output_file, matlab_data)
        print(f"  ✅ Saved: {output_file}")
        
        return matlab_data
    
    def convert_all(self):
        """Convert all datasets"""
        print("🧠 NT-ViT to CortexFlow Converter")
        print("=" * 50)
        
        datasets = ["mindbigdata", "crell"]
        results = {}
        
        for dataset in datasets:
            try:
                results[dataset] = self.convert_dataset(dataset)
            except Exception as e:
                print(f"❌ Error converting {dataset}: {e}")
                continue
        
        print(f"\n✅ Conversion complete!")
        print(f"📁 Output files in: {self.output_dir}/")
        for dataset in results.keys():
            print(f"  • {dataset}.mat")
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert NT-ViT outputs to CortexFlow format")
    parser.add_argument("--ntvit_outputs", default="ntvit_outputs", 
                       help="Directory containing NT-ViT outputs")
    parser.add_argument("--datasets", default="datasets", 
                       help="Directory containing original datasets")
    parser.add_argument("--output", default="cortexflow_data", 
                       help="Output directory for CortexFlow files")
    
    args = parser.parse_args()
    
    # Create converter and run
    converter = NTViTToCortexFlowConverter(
        ntvit_outputs_dir=args.ntvit_outputs,
        datasets_dir=args.datasets,
        output_dir=args.output
    )
    
    converter.convert_all()

if __name__ == "__main__":
    main()
