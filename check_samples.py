#!/usr/bin/env python3
"""
Quick check of CortexFlow sample counts
"""

import scipy.io
import numpy as np

def main():
    print('🔍 Checking CortexFlow conversion results...')
    print('=' * 50)

    # Check MindBigData
    print('📊 MindBigData Dataset:')
    try:
        data_mb = scipy.io.loadmat('cortexflow_data/mindbigdata.mat')
        train_count = data_mb["fmriTrn"].shape[0]
        test_count = data_mb["fmriTest"].shape[0]
        total_mb = train_count + test_count
        
        print(f'  Train samples: {train_count}')
        print(f'  Test samples: {test_count}')
        print(f'  Total samples: {total_mb}')
        print(f'  Train labels: {sorted(np.unique(data_mb["labelTrn"]))}')
        print(f'  Test labels: {sorted(np.unique(data_mb["labelTest"]))}')
        
    except Exception as e:
        print(f'  Error: {e}')
        total_mb = 0

    print()

    # Check Crell
    print('📊 Crell Dataset:')
    try:
        data_crell = scipy.io.loadmat('cortexflow_data/crell.mat')
        train_count = data_crell["fmriTrn"].shape[0]
        test_count = data_crell["fmriTest"].shape[0]
        total_crell = train_count + test_count
        
        print(f'  Train samples: {train_count}')
        print(f'  Test samples: {test_count}')
        print(f'  Total samples: {total_crell}')
        print(f'  Train labels: {sorted(np.unique(data_crell["labelTrn"]))}')
        print(f'  Test labels: {sorted(np.unique(data_crell["labelTest"]))}')
        
    except Exception as e:
        print(f'  Error: {e}')
        total_crell = 0

    print()
    print('✅ Summary:')
    print(f'  MindBigData: {total_mb} samples')
    print(f'  Crell: {total_crell} samples')
    print(f'  Grand Total: {total_mb + total_crell} samples')
    
    # Check if full 50 samples
    if total_mb == 50:
        print(f'  ✅ MindBigData: Full 50 samples confirmed')
    else:
        print(f'  ❌ MindBigData: Expected 50, got {total_mb}')
        
    if total_crell == 50:
        print(f'  ✅ Crell: Full 50 samples confirmed')
    else:
        print(f'  ❌ Crell: Expected 50, got {total_crell}')

if __name__ == "__main__":
    main()
