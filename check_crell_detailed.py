#!/usr/bin/env python3
"""
Detailed check of Crell dataset
"""

import scipy.io
import numpy as np

def main():
    print('🔍 Detailed Crell Dataset Analysis')
    print('=' * 50)
    
    try:
        data = scipy.io.loadmat('datasets/S01.mat')
        print('📋 Available keys:')
        for key in data.keys():
            if not key.startswith('__'):
                print(f'  {key}: {type(data[key])} - {data[key].shape if hasattr(data[key], "shape") else "no shape"}')

        total_events = 0
        
        # Check paradigm data
        for key in ['round01_paradigm', 'round02_paradigm']:
            if key in data:
                paradigm = data[key]
                print(f'\n📊 {key}:')
                print(f'  Type: {type(paradigm)}')
                print(f'  Shape: {paradigm.shape}')
                
                if paradigm.size > 0:
                    try:
                        if paradigm.shape == (1, 1):
                            inner_data = paradigm[0, 0]
                            print(f'  Inner type: {type(inner_data)}')
                            
                            if hasattr(inner_data, 'shape'):
                                print(f'  Inner shape: {inner_data.shape}')
                                
                                if len(inner_data.shape) > 0:
                                    markers = inner_data.flatten()
                                    all_markers = markers[markers > 0]
                                    unique_markers = np.unique(all_markers)
                                    print(f'  Total markers: {len(all_markers)}')
                                    print(f'  Unique markers: {len(unique_markers)}')
                                    
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
                                            
                    except Exception as e:
                        print(f'  Error accessing inner data: {e}')
        
        print(f'\n✅ Total letter events across all paradigms: {total_events}')
        return total_events
        
    except Exception as e:
        print(f'❌ Error reading Crell file: {e}')
        return 0

if __name__ == "__main__":
    total = main()
    print(f'\n📊 Summary:')
    print(f'  Crell total events: {total}')
    print(f'  Currently using: 50 samples')
    if total > 0:
        print(f'  Utilization: {50/total*100:.1f}%')
        print(f'  Potential expansion: Could use up to {total} events')
