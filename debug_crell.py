#!/usr/bin/env python3
"""
Debug Crell dataset loading error
"""

import sys
sys.path.append('.')
import scipy.io
import numpy as np
from pathlib import Path

def debug_crell_slicing():
    print('🔍 Debugging Crell Slicing Error...')
    print('=' * 50)
    
    filepath = Path('datasets/S01.mat')
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return
    
    try:
        data = scipy.io.loadmat(str(filepath))
        print(f'📋 Available keys: {list(data.keys())}')
        
        # Check paradigm data
        for paradigm_key in ['round01_paradigm', 'round02_paradigm']:
            if paradigm_key in data:
                print(f'\n📊 Processing {paradigm_key}...')
                round_data = data[paradigm_key][0, 0]
                
                # Extract data
                eeg_data = round_data['BrainVisionRDA_data'].T  # (64, timepoints) at 500Hz
                eeg_times = round_data['BrainVisionRDA_time'].flatten()
                marker_data = round_data['ParadigmMarker_data'].flatten()
                marker_times = round_data['ParadigmMarker_time'].flatten()
                
                print(f'  EEG data shape: {eeg_data.shape}')
                print(f'  EEG times shape: {eeg_times.shape}')
                print(f'  EEG times type: {type(eeg_times)}')
                print(f'  EEG times dtype: {eeg_times.dtype}')
                print(f'  Marker data shape: {marker_data.shape}')
                print(f'  Marker times shape: {marker_times.shape}')
                print(f'  Marker times type: {type(marker_times)}')
                print(f'  Marker times dtype: {marker_times.dtype}')
                
                # Test a simple search
                test_time = marker_times[0] if len(marker_times) > 0 else 0
                print(f'  Test time: {test_time} (type: {type(test_time)})')
                
                start_idx = np.searchsorted(eeg_times, test_time)
                print(f'  Search result: {start_idx} (type: {type(start_idx)})')
                print(f'  Search result dtype: {start_idx.dtype if hasattr(start_idx, "dtype") else "no dtype"}')
                
                # Test conversion to int
                try:
                    start_idx_int = int(start_idx)
                    print(f'  Converted to int: {start_idx_int} (type: {type(start_idx_int)})')
                except Exception as e:
                    print(f'  ❌ Error converting to int: {e}')
                
                # Test slicing
                try:
                    test_slice = eeg_data[:, start_idx:start_idx+10]
                    print(f'  ✅ Direct slicing works: {test_slice.shape}')
                except Exception as e:
                    print(f'  ❌ Direct slicing failed: {e}')
                
                try:
                    test_slice_int = eeg_data[:, int(start_idx):int(start_idx)+10]
                    print(f'  ✅ Int slicing works: {test_slice_int.shape}')
                except Exception as e:
                    print(f'  ❌ Int slicing failed: {e}')
                
                # Test with numpy.int64 conversion
                try:
                    start_idx_np = np.int64(start_idx)
                    test_slice_np = eeg_data[:, start_idx_np:start_idx_np+10]
                    print(f'  ✅ np.int64 slicing works: {test_slice_np.shape}')
                except Exception as e:
                    print(f'  ❌ np.int64 slicing failed: {e}')
                
                break  # Only test first paradigm
                
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_crell_slicing()
