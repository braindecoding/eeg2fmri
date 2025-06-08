#!/usr/bin/env python3
"""
Check raw EEG datasets to see total available data
"""

import scipy.io
import numpy as np
from pathlib import Path

def check_mindbigdata_raw():
    print('🔍 Checking MindBigData Raw File...')
    print('=' * 50)
    
    filepath = Path('datasets/EP1.01.txt')
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return 0
    
    print(f'📁 File: {filepath}')
    
    try:
        # Count lines in file
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f'📊 Total lines in file: {total_lines}')
        
        # Check first few lines to understand format
        print('📋 First 5 lines:')
        for i, line in enumerate(lines[:5]):
            print(f'  {i+1}: {line.strip()[:100]}...')
        
        # Count valid data lines (skip header if any)
        valid_lines = 0
        digit_counts = {str(i): 0 for i in range(10)}
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 7:  # Expected format: [id][event][device][channel][code][size][data]
                try:
                    code = parts[4]  # Code field
                    if code in digit_counts:
                        digit_counts[code] += 1
                        valid_lines += 1
                except:
                    continue
        
        print(f'📊 Valid data lines: {valid_lines}')
        print('📊 Digit distribution:')
        for digit, count in digit_counts.items():
            print(f'  Digit {digit}: {count} samples')
        
        return valid_lines
        
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return 0

def check_crell_raw():
    print('\n🔍 Checking Crell Raw .mat File...')
    print('=' * 50)
    
    filepath = Path('datasets/S01.mat')
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return 0
    
    print(f'📁 File: {filepath}')
    
    try:
        data = scipy.io.loadmat(str(filepath))
        print(f'📋 Available keys: {list(data.keys())}')
        
        total_events = 0
        
        # Check each paradigm
        for key in data.keys():
            if 'paradigm' in key:
                paradigm = data[key]
                print(f'\n📊 {key}:')
                print(f'  Shape: {paradigm.shape if hasattr(paradigm, "shape") else type(paradigm)}')
                
                if hasattr(paradigm, 'shape') and len(paradigm.shape) > 0:
                    # Count markers
                    markers = paradigm.flatten()
                    all_markers = markers[markers > 0]  # Non-zero markers
                    print(f'  Total markers: {len(all_markers)}')
                    
                    # Count letter events (101-110)
                    letter_markers = markers[(markers >= 101) & (markers <= 110)]
                    print(f'  Letter events (101-110): {len(letter_markers)}')
                    total_events += len(letter_markers)
                    
                    if len(letter_markers) > 0:
                        print('  Letter distribution:')
                        letters = ['a','d','e','f','j','n','o','s','t','v']
                        for i, letter in enumerate(letters):
                            marker = 101 + i
                            count = np.sum(letter_markers == marker)
                            print(f'    {letter} (marker {marker}): {count} events')
        
        print(f'\n📊 Total letter events across all paradigms: {total_events}')
        return total_events
        
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return 0

def check_stimuli_files():
    print('\n🔍 Checking Stimulus Files...')
    print('=' * 50)
    
    # Check MindBigData stimuli
    mb_stimuli_dir = Path('datasets/MindbigdataStimuli')
    if mb_stimuli_dir.exists():
        mb_files = list(mb_stimuli_dir.glob('*.jpg'))
        print(f'📁 MindBigData stimuli: {len(mb_files)} files')
        for f in sorted(mb_files):
            print(f'  {f.name}')
    else:
        print('❌ MindBigData stimuli directory not found')
    
    # Check Crell stimuli
    crell_stimuli_dir = Path('datasets/crellStimuli')
    if crell_stimuli_dir.exists():
        crell_files = list(crell_stimuli_dir.glob('*.png'))
        print(f'📁 Crell stimuli: {len(crell_files)} files')
        for f in sorted(crell_files):
            print(f'  {f.name}')
    else:
        print('❌ Crell stimuli directory not found')

def main():
    print('🧠 Raw EEG Dataset Analysis')
    print('=' * 60)
    
    # Check raw files
    mb_total = check_mindbigdata_raw()
    crell_total = check_crell_raw()
    
    # Check stimulus files
    check_stimuli_files()
    
    print('\n✅ Summary:')
    print('=' * 30)
    print(f'📊 MindBigData: {mb_total} total events available')
    print(f'   Currently using: 50 samples ({"%.1f" % (50/mb_total*100 if mb_total > 0 else 0)}%)')
    
    print(f'📊 Crell: {crell_total} total events available')
    print(f'   Currently using: 50 samples ({"%.1f" % (50/crell_total*100 if crell_total > 0 else 0)}%)')
    
    total_available = mb_total + crell_total
    print(f'📊 Total available: {total_available} events')
    print(f'   Currently using: 100 samples ({"%.1f" % (100/total_available*100 if total_available > 0 else 0)}%)')
    
    if total_available > 100:
        print(f'\n💡 Potential for expansion:')
        print(f'   Could generate up to {total_available} samples')
        print(f'   Current utilization: {"%.1f" % (100/total_available*100)}%')

if __name__ == "__main__":
    main()
