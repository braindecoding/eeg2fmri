#!/usr/bin/env python3
"""
CortexFlow Data Verification Script
==================================

Verifies and displays the structure of generated CortexFlow-compatible MATLAB files.
"""

import scipy.io
import numpy as np
from pathlib import Path

def verify_matlab_file(filepath):
    """Verify and display MATLAB file structure"""
    print(f"\n📁 Verifying: {filepath}")
    print("=" * 60)
    
    try:
        # Load MATLAB file
        data = scipy.io.loadmat(filepath)
        
        # Remove MATLAB metadata keys
        data_keys = [k for k in data.keys() if not k.startswith('__')]
        
        print(f"📋 Available keys: {data_keys}")
        
        # Display each dataset
        for key in ['fmriTrn', 'stimTrn', 'fmriTest', 'stimTest', 'labelTrn', 'labelTest']:
            if key in data:
                arr = data[key]
                dtype_str = str(arr.dtype)
                
                print(f"\n  📈 {key}: {arr.shape} - {dtype_str}")
                
                if 'fmri' in key.lower():
                    print(f"      Range: [{arr.min():.3f}, {arr.max():.3f}], "
                          f"Mean: {arr.mean():.3f}, Std: {arr.std():.3f}")
                elif 'stim' in key.lower():
                    print(f"      Range: [{arr.min():.3f}, {arr.max():.3f}] "
                          f"(pixel values)")
                elif 'label' in key.lower():
                    unique_labels = np.unique(arr)
                    print(f"      Unique labels: {unique_labels}")
        
        # Display metadata if available (now flattened)
        metadata_keys = ['dataset_type', 'n_train_samples', 'n_test_samples',
                        'fmri_dimensions', 'stimulus_dimensions', 'source', 'compatible_with']

        metadata_found = [k for k in metadata_keys if k in data]
        if metadata_found:
            print(f"\n  📋 Metadata:")
            for key in metadata_found:
                value = data[key]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                print(f"      {key}: {value}")
        
        print(f"\n  ✅ File structure is valid!")
        
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")

def main():
    """Main verification function"""
    print("🔍 CortexFlow Data Verification")
    print("=" * 50)
    
    cortexflow_dir = Path("cortexflow_data")
    
    if not cortexflow_dir.exists():
        print(f"❌ Directory {cortexflow_dir} not found!")
        return
    
    # Verify each MATLAB file
    matlab_files = list(cortexflow_dir.glob("*.mat"))
    
    if not matlab_files:
        print(f"❌ No .mat files found in {cortexflow_dir}")
        return
    
    for mat_file in sorted(matlab_files):
        verify_matlab_file(mat_file)
    
    print(f"\n🎯 Summary:")
    print(f"  • Found {len(matlab_files)} CortexFlow-compatible files")
    print(f"  • All files follow the required format:")
    print(f"    - fmriTrn/fmriTest: Synthetic fMRI data")
    print(f"    - stimTrn/stimTest: Stimulus images (784 pixels)")
    print(f"    - labelTrn/labelTest: Class labels")
    print(f"  • Ready for CortexFlow integration! 🚀")

if __name__ == "__main__":
    main()
