#!/usr/bin/env python3
"""
Debug Crell Dataset Structure
=============================

Investigate the structure of S01.mat to understand why no paradigm rounds are found.
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

def debug_crell_structure():
    """Debug the structure of Crell S01.mat file"""
    
    print("🔍 Debugging Crell Dataset Structure")
    print("=" * 50)
    
    filepath = Path("datasets/S01.mat")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"📁 Loading: {filepath}")
    
    try:
        # Load the .mat file
        data = sio.loadmat(str(filepath))
        
        print(f"\n📊 Top-level keys in S01.mat:")
        for key in data.keys():
            if not key.startswith('__'):
                value = data[key]
                print(f"  {key}: {type(value)} - {getattr(value, 'shape', 'no shape')}")
        
        # Look for paradigm-related keys
        paradigm_keys = [key for key in data.keys() if 'paradigm' in key.lower() or 'Paradigm' in key]
        print(f"\n🔍 Paradigm-related keys: {paradigm_keys}")
        
        # Look for any keys that might contain data
        data_keys = [key for key in data.keys() if not key.startswith('__')]
        print(f"\n📋 All data keys: {data_keys}")
        
        # Examine each key in detail
        for key in data_keys:
            if not key.startswith('__'):
                value = data[key]
                print(f"\n🔍 Examining key: {key}")
                print(f"  Type: {type(value)}")
                print(f"  Shape: {getattr(value, 'shape', 'no shape')}")
                print(f"  Dtype: {getattr(value, 'dtype', 'no dtype')}")
                
                # If it's an array, show some basic info
                if hasattr(value, 'shape') and len(value.shape) > 0:
                    print(f"  Size: {value.size}")
                    if value.size > 0:
                        print(f"  First few elements: {value.flat[:min(5, value.size)]}")
                
                # If it's a structured array, examine its fields
                if hasattr(value, 'dtype') and value.dtype.names:
                    print(f"  Fields: {value.dtype.names}")
                    
                    # Try to access the first element if it exists
                    if value.size > 0:
                        first_elem = value[0, 0] if len(value.shape) >= 2 else value[0]
                        print(f"  First element type: {type(first_elem)}")
                        
                        # If first element has attributes, list them
                        if hasattr(first_elem, 'dtype') and first_elem.dtype.names:
                            print(f"  First element fields: {first_elem.dtype.names}")
                            
                            # Check for EEG-related fields
                            for field in first_elem.dtype.names:
                                if any(term in field.lower() for term in ['eeg', 'brain', 'rda', 'data', 'time', 'marker']):
                                    field_data = first_elem[field]
                                    print(f"    {field}: {type(field_data)} - {getattr(field_data, 'shape', 'no shape')}")
        
        # Try alternative access patterns
        print(f"\n🔍 Trying alternative access patterns...")
        
        # Check if there are any cell arrays
        for key in data_keys:
            value = data[key]
            if hasattr(value, 'dtype') and 'object' in str(value.dtype):
                print(f"  {key} is object array, trying to access elements...")
                try:
                    if value.size > 0:
                        elem = value.item() if value.size == 1 else value[0]
                        print(f"    Element type: {type(elem)}")
                        if hasattr(elem, 'dtype') and elem.dtype.names:
                            print(f"    Element fields: {elem.dtype.names}")
                except Exception as e:
                    print(f"    Error accessing element: {e}")
        
        # Look for any keys containing 'round', 'trial', 'session', etc.
        session_keys = [key for key in data.keys() if any(term in key.lower() for term in ['round', 'trial', 'session', 'run', 'block'])]
        print(f"\n🔍 Session/Round/Trial keys: {session_keys}")
        
        # Check for numeric keys that might be paradigm rounds
        numeric_keys = []
        for key in data.keys():
            if key.isdigit() or (key.startswith('x') and key[1:].isdigit()):
                numeric_keys.append(key)
        print(f"\n🔍 Numeric keys: {numeric_keys}")
        
    except Exception as e:
        print(f"❌ Error loading Crell data: {e}")
        import traceback
        traceback.print_exc()

def test_existing_crell_loader():
    """Test the existing Crell loader to see what it finds"""
    
    print(f"\n🧪 Testing Existing Crell Loader...")
    print("-" * 40)
    
    try:
        import sys
        sys.path.append('.')
        from train_ntvit import CrellDataLoader
        
        loader = CrellDataLoader(
            filepath="datasets/S01.mat",
            stimuli_dir="datasets/crellStimuli",
            max_samples=10
        )
        
        print(f"✓ Existing loader found {len(loader.samples)} samples")
        
        if len(loader.samples) > 0:
            sample = loader.samples[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  EEG shape: {sample['eeg_data'].shape}")
            print(f"  Letter: {sample.get('letter', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Existing loader failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function"""
    
    debug_crell_structure()
    test_existing_crell_loader()
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Check the actual structure found above")
    print(f"  2. Modify CrellDataLoaderRobust to match the real structure")
    print(f"  3. Look for the correct field names for EEG data and markers")

if __name__ == "__main__":
    main()
