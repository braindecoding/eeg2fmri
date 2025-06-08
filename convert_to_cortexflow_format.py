#!/usr/bin/env python3
"""
Convert to CortexFlow Format
============================

Convert our translated fMRI data to the correct CortexFlow format:
- fmriTrn: (N_train, 3092) training fMRI data
- fmriTest: (N_test, 3092) test fMRI data  
- stimTrn: (N_train, 784) training stimulus images
- stimTest: (N_test, 784) test stimulus images
- labelTrn: (N_train, 1) training labels
- labelTest: (N_test, 1) test labels
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def convert_to_cortexflow_format(input_file: str, output_file: str, dataset_name: str, test_size: float = 0.1):
    """Convert our format to CortexFlow format with train/test split"""
    
    print(f"🔄 Converting {dataset_name} to CortexFlow format")
    print("=" * 60)
    
    # Load our data
    print(f"📂 Loading data from {input_file}...")
    data = sio.loadmat(input_file)
    
    fmri_data = data['fmri']  # (N, 3092)
    stim_data = data['stim']  # (N, 784)
    label_data = data['labels']  # (N, 1)
    
    print(f"✅ Loaded data:")
    print(f"  fMRI: {fmri_data.shape}")
    print(f"  Stimulus: {stim_data.shape}")
    print(f"  Labels: {label_data.shape}")
    
    # Ensure labels are (N, 1) format
    if label_data.ndim == 1:
        label_data = label_data.reshape(-1, 1)
    
    # Create stratified train/test split
    print(f"\n📊 Creating stratified train/test split ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
    
    # Flatten labels for stratification
    labels_flat = label_data.flatten()
    
    # Get indices for train/test split
    train_indices, test_indices = train_test_split(
        np.arange(len(fmri_data)),
        test_size=test_size,
        stratify=labels_flat,
        random_state=42  # For reproducibility
    )
    
    # Split data
    fmri_train = fmri_data[train_indices]
    fmri_test = fmri_data[test_indices]
    
    stim_train = stim_data[train_indices]
    stim_test = stim_data[test_indices]
    
    label_train = label_data[train_indices]
    label_test = label_data[test_indices]
    
    print(f"✅ Split completed:")
    print(f"  Training set: {len(train_indices)} samples")
    print(f"  Test set: {len(test_indices)} samples")
    
    # Verify class distribution
    print(f"\n📈 Class distribution verification:")
    unique_labels = np.unique(labels_flat)
    for label in unique_labels:
        train_count = np.sum(label_train.flatten() == label)
        test_count = np.sum(label_test.flatten() == label)
        total_count = np.sum(labels_flat == label)
        print(f"  Class {label}: Train={train_count}, Test={test_count}, Total={total_count}")
    
    # Create CortexFlow format
    cortexflow_data = {
        'fmriTrn': fmri_train.astype(np.float64),    # (N_train, 3092)
        'fmriTest': fmri_test.astype(np.float64),    # (N_test, 3092)
        'stimTrn': stim_train.astype(np.uint8),      # (N_train, 784)
        'stimTest': stim_test.astype(np.uint8),      # (N_test, 784)
        'labelTrn': label_train.astype(np.uint8),    # (N_train, 1)
        'labelTest': label_test.astype(np.uint8)     # (N_test, 1)
    }
    
    # Save in CortexFlow format
    print(f"\n💾 Saving CortexFlow format to {output_file}...")
    sio.savemat(output_file, cortexflow_data)
    
    # Verify saved data
    print(f"✅ Saved CortexFlow format:")
    for key, value in cortexflow_data.items():
        print(f"  {key}: {value.shape} {value.dtype}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'original_file': input_file,
        'cortexflow_file': output_file,
        'total_samples': len(fmri_data),
        'train_samples': len(train_indices),
        'test_samples': len(test_indices),
        'test_ratio': test_size,
        'num_classes': len(unique_labels),
        'class_distribution': {
            'train': {int(label): int(np.sum(label_train.flatten() == label)) for label in unique_labels},
            'test': {int(label): int(np.sum(label_test.flatten() == label)) for label in unique_labels}
        },
        'data_shapes': {
            'fmriTrn': list(fmri_train.shape),
            'fmriTest': list(fmri_test.shape),
            'stimTrn': list(stim_train.shape),
            'stimTest': list(stim_test.shape),
            'labelTrn': list(label_train.shape),
            'labelTest': list(label_test.shape)
        },
        'data_types': {
            'fmriTrn': 'float64',
            'fmriTest': 'float64',
            'stimTrn': 'uint8',
            'stimTest': 'uint8',
            'labelTrn': 'uint8',
            'labelTest': 'uint8'
        },
        'cortexflow_format': 'Compatible with CortexFlow train/test structure'
    }
    
    metadata_file = output_file.replace('.mat', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata: {metadata_file}")
    
    return cortexflow_data, metadata

def convert_both_datasets():
    """Convert both MindBigData and Crell to CortexFlow format"""
    
    print("🔄 Converting Both Datasets to CortexFlow Format")
    print("=" * 70)
    
    # Dataset configurations
    datasets = [
        {
            'name': 'MindBigData',
            'input_file': 'translated_fmri_outputs/mindbigdata_translated_fmri.mat',
            'output_file': 'cortexflow_outputs/mindbigdata_cortexflow.mat',
            'test_size': 0.1  # 10% test set
        },
        {
            'name': 'Crell',
            'input_file': 'crell_translated_fmri_outputs/crell_translated_fmri.mat',
            'output_file': 'cortexflow_outputs/crell_cortexflow.mat',
            'test_size': 0.1  # 10% test set
        }
    ]
    
    # Create output directory
    output_dir = Path('cortexflow_outputs')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for dataset_config in datasets:
        name = dataset_config['name']
        input_file = dataset_config['input_file']
        output_file = dataset_config['output_file']
        test_size = dataset_config['test_size']
        
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"⚠️  Input file not found: {input_file}")
            continue
        
        try:
            # Convert dataset
            cortexflow_data, metadata = convert_to_cortexflow_format(
                input_file=input_file,
                output_file=output_file,
                dataset_name=name,
                test_size=test_size
            )
            
            results[name] = {
                'success': True,
                'output_file': output_file,
                'metadata': metadata
            }
            
            print(f"✅ {name} conversion completed successfully!")
            
        except Exception as e:
            print(f"❌ {name} conversion failed: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
        
        print(f"\n" + "="*70)
    
    # Summary
    print(f"\n📊 Conversion Summary:")
    successful = 0
    for name, result in results.items():
        if result['success']:
            successful += 1
            metadata = result['metadata']
            print(f"✅ {name}:")
            print(f"  Output: {result['output_file']}")
            print(f"  Train samples: {metadata['train_samples']}")
            print(f"  Test samples: {metadata['test_samples']}")
            print(f"  Classes: {metadata['num_classes']}")
        else:
            print(f"❌ {name}: {result['error']}")
    
    print(f"\n🎯 Conversion Results: {successful}/{len(datasets)} datasets converted successfully")
    
    if successful > 0:
        print(f"\n🚀 CortexFlow-ready files saved in cortexflow_outputs/")
        print(f"📋 Use these files directly with CortexFlow framework")
    
    return results

def verify_cortexflow_format(file_path: str):
    """Verify that the file matches CortexFlow format requirements"""
    
    print(f"🔍 Verifying CortexFlow format: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        data = sio.loadmat(file_path)
        
        # Expected keys
        expected_keys = ['fmriTrn', 'fmriTest', 'stimTrn', 'stimTest', 'labelTrn', 'labelTest']
        
        print(f"📋 File keys: {list(data.keys())}")
        
        # Check all required keys exist
        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        
        # Check data shapes and types
        print(f"📊 Data verification:")
        for key in expected_keys:
            array = data[key]
            print(f"  {key}: {array.shape} {array.dtype}")
            
            # Verify shapes
            if key.startswith('fmri') and array.shape[1] != 3092:
                print(f"❌ {key} should have 3092 columns (fMRI voxels)")
                return False
            
            if key.startswith('stim') and array.shape[1] != 784:
                print(f"❌ {key} should have 784 columns (28x28 stimulus)")
                return False
            
            if key.startswith('label') and array.shape[1] != 1:
                print(f"❌ {key} should have 1 column (labels)")
                return False
        
        # Check train/test consistency
        train_samples = data['fmriTrn'].shape[0]
        test_samples = data['fmriTest'].shape[0]
        
        if (data['stimTrn'].shape[0] != train_samples or 
            data['labelTrn'].shape[0] != train_samples):
            print(f"❌ Training set size mismatch")
            return False
        
        if (data['stimTest'].shape[0] != test_samples or 
            data['labelTest'].shape[0] != test_samples):
            print(f"❌ Test set size mismatch")
            return False
        
        print(f"✅ CortexFlow format verification passed!")
        print(f"  Training samples: {train_samples}")
        print(f"  Test samples: {test_samples}")
        print(f"  Total samples: {train_samples + test_samples}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🔄 CortexFlow Format Converter")
    print("=" * 50)
    print("📋 Converting translated fMRI data to CortexFlow train/test format")
    
    # Convert both datasets
    results = convert_both_datasets()
    
    # Verify converted files
    print(f"\n🔍 Verifying converted files...")
    for name, result in results.items():
        if result['success']:
            print(f"\n{name} verification:")
            verify_cortexflow_format(result['output_file'])
    
    print(f"\n🎉 CortexFlow format conversion completed!")
    print(f"📁 Output files ready in cortexflow_outputs/")

if __name__ == "__main__":
    main()
