#!/usr/bin/env python3
"""
Check label distribution in CortexFlow datasets
"""

import scipy.io
import numpy as np

def main():
    print('🔍 Detailed Label Distribution Check...')
    print('=' * 50)

    # MindBigData
    data_mb = scipy.io.loadmat('cortexflow_data/mindbigdata.mat')
    train_labels = data_mb['labelTrn'].flatten()
    test_labels = data_mb['labelTest'].flatten()

    print('📊 MindBigData Label Distribution:')
    print('  Train set (40 samples):')
    for label in range(10):
        count = np.sum(train_labels == label)
        print(f'    Digit {label}: {count} samples')

    print('  Test set (10 samples):')
    for label in range(10):
        count = np.sum(test_labels == label)
        print(f'    Digit {label}: {count} samples')

    print()

    # Crell
    data_crell = scipy.io.loadmat('cortexflow_data/crell.mat')
    train_labels = data_crell['labelTrn'].flatten()
    test_labels = data_crell['labelTest'].flatten()

    print('📊 Crell Label Distribution:')
    letters = ['a','d','e','f','j','n','o','s','t','v']
    
    print('  Train set (40 samples):')
    for label in range(1, 11):
        count = np.sum(train_labels == label)
        letter = letters[label-1]
        print(f'    Letter {letter} (label {label}): {count} samples')

    print('  Test set (10 samples):')
    for label in range(1, 11):
        count = np.sum(test_labels == label)
        letter = letters[label-1]
        print(f'    Letter {letter} (label {label}): {count} samples')

    print()
    print('✅ Verification:')
    
    # Check if all labels represented in test
    mb_test_unique = len(np.unique(test_labels))
    crell_test_unique = len(np.unique(data_crell['labelTest'].flatten()))
    
    print(f'  MindBigData test coverage: {mb_test_unique}/10 digits')
    print(f'  Crell test coverage: {crell_test_unique}/10 letters')
    
    if mb_test_unique == 10:
        print('  ✅ MindBigData: All digits represented in test set')
    else:
        print('  ❌ MindBigData: Missing digits in test set')
        
    if crell_test_unique == 10:
        print('  ✅ Crell: All letters represented in test set')
    else:
        print('  ❌ Crell: Missing letters in test set')

if __name__ == "__main__":
    main()
