#!/usr/bin/env python3
"""
Check original EEG datasets to see total available samples
"""

import sys
sys.path.append('.')
from train_ntvit import MindBigDataLoader, CrellDataLoader
from pathlib import Path
import scipy.io
import numpy as np

def check_mindbigdata():
    print('🔍 Checking MindBigData Original Dataset...')
    print('=' * 50)
    
    datasets_dir = Path('datasets')
    filepath = datasets_dir / "EP1.01.txt"
    stimuli_dir = datasets_dir / "MindbigdataStimuli"
    
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return
    
    print(f'📁 File: {filepath}')
    print(f'📁 Stimuli: {stimuli_dir}')
    
    # Load with no limit to see total available
    loader = MindBigDataLoader(
        filepath=str(filepath),
        stimuli_dir=str(stimuli_dir),
        max_samples=None  # No limit
    )
    
    total_samples = len(loader.samples)
    print(f'📊 Total available samples: {total_samples}')
    
    if total_samples > 0:
        # Check label distribution
        labels = [sample['label'] for sample in loader.samples]
        unique_labels = sorted(set(labels))
        print(f'📈 Available labels: {unique_labels}')
        
        print('📊 Label distribution:')
        for label in unique_labels:
            count = labels.count(label)
            print(f'  Digit {label}: {count} samples')
    
    return total_samples

def check_crell():
    print('\n🔍 Checking Crell Original Dataset...')
    print('=' * 50)
    
    datasets_dir = Path('datasets')
    filepath = datasets_dir / "S01.mat"
    stimuli_dir = datasets_dir / "crellStimuli"
    
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return
    
    print(f'📁 File: {filepath}')
    print(f'📁 Stimuli: {stimuli_dir}')
    
    # Load with no limit to see total available
    loader = CrellDataLoader(
        filepath=str(filepath),
        stimuli_dir=str(stimuli_dir),
        max_samples=None  # No limit
    )
    
    total_samples = len(loader.samples)
    print(f'📊 Total available samples: {total_samples}')
    
    if total_samples > 0:
        # Check label distribution
        labels = [sample['label'] for sample in loader.samples]
        unique_labels = sorted(set(labels))
        print(f'📈 Available labels: {unique_labels}')
        
        letters = ['a','d','e','f','j','n','o','s','t','v']
        print('📊 Label distribution:')
        for label in unique_labels:
            count = labels.count(label)
            letter = letters[label-1] if 1 <= label <= 10 else '?'
            print(f'  Letter {letter} (label {label}): {count} samples')
    
    return total_samples

def check_raw_crell_file():
    print('\n🔍 Checking Raw Crell .mat File...')
    print('=' * 50)
    
    filepath = Path('datasets/S01.mat')
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return
    
    try:
        data = scipy.io.loadmat(str(filepath))
        print(f'📋 Available keys: {list(data.keys())}')
        
        # Check paradigm data
        for key in data.keys():
            if 'paradigm' in key:
                paradigm = data[key]
                print(f'📊 {key}: {paradigm.shape if hasattr(paradigm, "shape") else type(paradigm)}')
                
                if hasattr(paradigm, 'shape') and len(paradigm.shape) > 0:
                    # Count markers
                    markers = paradigm.flatten()
                    unique_markers = np.unique(markers[markers > 0])  # Non-zero markers
                    print(f'  Unique markers: {len(unique_markers)}')
                    print(f'  Total markers: {len(markers[markers > 0])}')
                    
                    # Count letter events (101-110)
                    letter_markers = markers[(markers >= 101) & (markers <= 110)]
                    print(f'  Letter events (101-110): {len(letter_markers)}')
                    
                    if len(letter_markers) > 0:
                        print('  Letter distribution:')
                        letters = ['a','d','e','f','j','n','o','s','t','v']
                        for i, letter in enumerate(letters):
                            marker = 101 + i
                            count = np.sum(letter_markers == marker)
                            print(f'    {letter} (marker {marker}): {count} events')
                            
    except Exception as e:
        print(f'❌ Error reading file: {e}')

def main():
    print('🧠 Original EEG Dataset Analysis')
    print('=' * 60)
    
    # Check MindBigData
    mb_total = check_mindbigdata()
    
    # Check Crell
    crell_total = check_crell()
    
    # Check raw Crell file for more details
    check_raw_crell_file()
    
    print('\n✅ Summary:')
    print('=' * 30)
    if mb_total is not None:
        print(f'📊 MindBigData: {mb_total} total samples available')
        print(f'   Currently using: 50 samples ({"%.1f" % (50/mb_total*100 if mb_total > 0 else 0)}%)')
    
    if crell_total is not None:
        print(f'📊 Crell: {crell_total} total samples available')
        print(f'   Currently using: 50 samples ({"%.1f" % (50/crell_total*100 if crell_total > 0 else 0)}%)')
    
    total_available = (mb_total or 0) + (crell_total or 0)
    print(f'📊 Total available: {total_available} samples')
    print(f'   Currently using: 100 samples ({"%.1f" % (100/total_available*100 if total_available > 0 else 0)}%)')

if __name__ == "__main__":
    main()
