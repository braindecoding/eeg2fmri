#!/usr/bin/env python3
"""
Test separated NT-ViT training scripts
"""

import sys
sys.path.append('.')
import torch
from pathlib import Path

def test_mindbigdata_script():
    print('🔍 Testing MindBigData Script...')
    print('=' * 50)
    
    try:
        from train_ntvit_mindbigdata import create_mindbigdata_loaders
        
        # Test data loading
        train_loader, val_loader = create_mindbigdata_loaders('datasets', batch_size=2)
        
        print(f'✅ MindBigData loaders created successfully!')
        print(f'📊 Train batches: {len(train_loader)}')
        print(f'📊 Val batches: {len(val_loader)}')
        
        # Test a batch
        sample_batch = next(iter(train_loader))
        print(f'📊 Batch shape: {sample_batch["eeg_data"].shape}')
        print(f'📊 EEG channels: {sample_batch["eeg_data"].shape[1]}')
        
        return True
        
    except Exception as e:
        print(f'❌ MindBigData script failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_crell_script():
    print('\n🔍 Testing Crell Script...')
    print('=' * 50)
    
    try:
        from train_ntvit_crell import create_crell_loaders
        
        # Test data loading
        train_loader, val_loader = create_crell_loaders('datasets', batch_size=2)
        
        print(f'✅ Crell loaders created successfully!')
        print(f'📊 Train batches: {len(train_loader)}')
        print(f'📊 Val batches: {len(val_loader)}')
        
        # Test a batch
        sample_batch = next(iter(train_loader))
        print(f'📊 Batch shape: {sample_batch["eeg_data"].shape}')
        print(f'📊 EEG channels: {sample_batch["eeg_data"].shape[1]}')
        
        return True
        
    except Exception as e:
        print(f'❌ Crell script failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_gpu_availability():
    print('\n🔍 Testing GPU Availability...')
    print('=' * 50)
    
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name()}')
        print(f'📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print(f'📊 CUDA Version: {torch.version.cuda}')
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f'✅ GPU memory allocation test passed')
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'⚠️  GPU memory allocation test failed: {e}')
        
        return True
    else:
        print(f'⚠️  CUDA not available, will use CPU')
        return False

def check_datasets():
    print('\n🔍 Checking Dataset Files...')
    print('=' * 50)
    
    datasets_path = Path('datasets')
    
    # Check MindBigData files
    mb_files = ["EP1.01.txt", "MindbigdataStimuli"]
    mb_ok = True
    
    print('📁 MindBigData files:')
    for file in mb_files:
        path = datasets_path / file
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1e6
                print(f'  ✅ {file}: {size:.1f} MB')
            else:
                count = len(list(path.glob('*'))) if path.is_dir() else 0
                print(f'  ✅ {file}: {count} files')
        else:
            print(f'  ❌ {file}: Missing')
            mb_ok = False
    
    # Check Crell files
    crell_files = ["S01.mat", "crellStimuli"]
    crell_ok = True
    
    print('\n📁 Crell files:')
    for file in crell_files:
        path = datasets_path / file
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1e6
                print(f'  ✅ {file}: {size:.1f} MB')
            else:
                count = len(list(path.glob('*'))) if path.is_dir() else 0
                print(f'  ✅ {file}: {count} files')
        else:
            print(f'  ❌ {file}: Missing')
            crell_ok = False
    
    return mb_ok, crell_ok

def estimate_training_time():
    print('\n🔍 Estimating Training Time...')
    print('=' * 50)
    
    # Rough estimates based on typical training
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        print('🚀 With GPU:')
        print('  MindBigData (~1174 samples, 30 epochs): ~2-3 hours')
        print('  Crell (~320 samples, 30 epochs): ~1-2 hours')
        print('  Total estimated time: ~3-5 hours')
    else:
        print('🐌 With CPU only:')
        print('  MindBigData (~1174 samples, 30 epochs): ~8-12 hours')
        print('  Crell (~320 samples, 30 epochs): ~4-6 hours')
        print('  Total estimated time: ~12-18 hours')
    
    print('\n💡 Recommendations:')
    if gpu_available:
        print('  ✅ GPU detected - proceed with training')
        print('  📊 Suggested batch_size: 4-8 (adjust based on GPU memory)')
    else:
        print('  ⚠️  Consider using GPU for faster training')
        print('  📊 Suggested batch_size: 2-4 (for CPU)')

def main():
    print('🧠 Separated NT-ViT Scripts Test')
    print('=' * 60)
    
    # Check datasets
    mb_ok, crell_ok = check_datasets()
    
    # Test GPU
    gpu_ok = test_gpu_availability()
    
    # Test scripts
    mb_script_ok = False
    crell_script_ok = False
    
    if mb_ok:
        mb_script_ok = test_mindbigdata_script()
    else:
        print('\n⚠️  Skipping MindBigData script test (missing files)')
    
    if crell_ok:
        crell_script_ok = test_crell_script()
    else:
        print('\n⚠️  Skipping Crell script test (missing files)')
    
    # Estimate training time
    estimate_training_time()
    
    # Summary
    print('\n✅ Final Summary:')
    print('=' * 30)
    print(f'📊 MindBigData files: {"✅ OK" if mb_ok else "❌ Missing"}')
    print(f'📊 Crell files: {"✅ OK" if crell_ok else "❌ Missing"}')
    print(f'📊 GPU available: {"✅ Yes" if gpu_ok else "❌ No"}')
    print(f'📊 MindBigData script: {"✅ OK" if mb_script_ok else "❌ Failed"}')
    print(f'📊 Crell script: {"✅ OK" if crell_script_ok else "❌ Failed"}')
    
    if mb_script_ok or crell_script_ok:
        print(f'\n🎉 Ready for training!')
        print(f'📋 Next steps:')
        if mb_script_ok:
            print(f'  1. Run: python train_ntvit_mindbigdata.py')
        if crell_script_ok:
            print(f'  2. Run: python train_ntvit_crell.py')
        print(f'  3. Monitor GPU memory usage during training')
        print(f'  4. Adjust batch_size if needed')
    else:
        print(f'\n⚠️  Fix issues before training')

if __name__ == "__main__":
    main()
