#!/usr/bin/env python3
"""
Test new NT-ViT configuration with:
- MindBigData: 1200 samples balanced per label (120 per digit)
- Crell: Maximum available samples
"""

import sys
sys.path.append('.')
from train_ntvit import MindBigDataLoader, CrellDataLoader, create_data_loaders
from pathlib import Path
import numpy as np

def test_mindbigdata_balanced():
    print('🔍 Testing MindBigData Balanced Loading...')
    print('=' * 50)
    
    datasets_dir = Path('datasets')
    filepath = datasets_dir / "EP1.01.txt"
    stimuli_dir = datasets_dir / "MindbigdataStimuli"
    
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return 0
    
    print(f'📁 File: {filepath}')
    print(f'📁 Stimuli: {stimuli_dir}')
    
    # Test balanced loading with 1200 samples
    print(f'\n🎯 Testing balanced loading (1200 samples, 120 per digit)...')
    loader = MindBigDataLoader(
        filepath=str(filepath),
        stimuli_dir=str(stimuli_dir),
        max_samples=1200,
        balanced_per_label=True
    )
    
    total_samples = len(loader.samples)
    print(f'📊 Total loaded samples: {total_samples}')
    
    if total_samples > 0:
        # Check label distribution
        labels = [sample['label'] for sample in loader.samples]
        print(f'\n📈 Label distribution:')
        for digit in range(10):
            count = labels.count(digit)
            print(f'  Digit {digit}: {count} samples')
        
        # Check if balanced
        unique_counts = list(set([labels.count(d) for d in range(10)]))
        if len(unique_counts) == 1:
            print(f'✅ Perfect balance: {unique_counts[0]} samples per digit')
        else:
            print(f'⚠️  Imbalanced: {unique_counts} samples per digit')
    
    return total_samples

def test_crell_maximum():
    print('\n🔍 Testing Crell Maximum Loading...')
    print('=' * 50)
    
    datasets_dir = Path('datasets')
    filepath = datasets_dir / "S01.mat"
    stimuli_dir = datasets_dir / "crellStimuli"
    
    if not filepath.exists():
        print(f'❌ File not found: {filepath}')
        return 0
    
    print(f'📁 File: {filepath}')
    print(f'📁 Stimuli: {stimuli_dir}')
    
    # Test maximum loading
    print(f'\n🎯 Testing maximum loading (no limit)...')
    loader = CrellDataLoader(
        filepath=str(filepath),
        stimuli_dir=str(stimuli_dir),
        max_samples=None  # No limit
    )
    
    total_samples = len(loader.samples)
    print(f'📊 Total loaded samples: {total_samples}')
    
    if total_samples > 0:
        # Check label distribution
        labels = [sample['label'] for sample in loader.samples]
        unique_labels = sorted(set(labels))
        print(f'📈 Available labels: {unique_labels}')
        
        letters = ['a','d','e','f','j','n','o','s','t','v']
        print('📊 Label distribution:')
        for label in unique_labels:
            count = labels.count(label)
            letter = letters[label] if 0 <= label < len(letters) else '?'
            print(f'  Letter {letter} (label {label}): {count} samples')
    
    return total_samples

def test_data_loaders():
    print('\n🔍 Testing Data Loaders with New Configuration...')
    print('=' * 50)
    
    try:
        train_loaders, val_loaders = create_data_loaders('datasets', batch_size=4)
        
        print(f'✅ Data loaders created successfully!')
        
        # Check MindBigData
        mb_train_size = len(train_loaders['mindbigdata'].dataset.samples)
        mb_val_size = len(val_loaders['mindbigdata'].dataset.samples)
        print(f'📊 MindBigData - Train: {mb_train_size}, Val: {mb_val_size}')
        
        # Check Crell
        crell_train_size = len(train_loaders['crell'].dataset.samples)
        crell_val_size = len(val_loaders['crell'].dataset.samples)
        print(f'📊 Crell - Train: {crell_train_size}, Val: {crell_val_size}')
        
        total_train = mb_train_size + crell_train_size
        total_val = mb_val_size + crell_val_size
        total_all = total_train + total_val
        
        print(f'\n📈 Summary:')
        print(f'  Total training samples: {total_train}')
        print(f'  Total validation samples: {total_val}')
        print(f'  Total samples: {total_all}')
        
        return True
        
    except Exception as e:
        print(f'❌ Error creating data loaders: {e}')
        return False

def main():
    print('🧠 NT-ViT New Configuration Test')
    print('=' * 60)
    
    # Test MindBigData balanced loading
    mb_total = test_mindbigdata_balanced()
    
    # Test Crell maximum loading
    crell_total = test_crell_maximum()
    
    # Test data loaders
    loaders_ok = test_data_loaders()
    
    print('\n✅ Final Summary:')
    print('=' * 30)
    print(f'📊 MindBigData: {mb_total} samples (target: 1200 balanced)')
    print(f'📊 Crell: {crell_total} samples (maximum available)')
    print(f'📊 Data loaders: {"✅ OK" if loaders_ok else "❌ Failed"}')
    
    total_available = mb_total + crell_total
    print(f'📊 Total available: {total_available} samples')
    
    if mb_total >= 1200 and crell_total > 0 and loaders_ok:
        print(f'\n🎉 Configuration ready for training!')
        print(f'   - MindBigData: 1200 balanced samples (120 per digit)')
        print(f'   - Crell: {crell_total} maximum samples')
        print(f'   - Total: {total_available} samples')
    else:
        print(f'\n⚠️  Configuration needs adjustment:')
        if mb_total < 1200:
            print(f'   - MindBigData: Only {mb_total}/1200 samples available')
        if crell_total == 0:
            print(f'   - Crell: No samples loaded')
        if not loaders_ok:
            print(f'   - Data loaders: Failed to create')

if __name__ == "__main__":
    main()
