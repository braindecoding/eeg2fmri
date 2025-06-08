#!/usr/bin/env python3
"""
Run NT-ViT Training with GPU Monitoring
=======================================

Automated script to run both MindBigData and Crell training
with GPU memory monitoring and optimization.
"""

import subprocess
import time
import torch
import psutil
from pathlib import Path
import json
from datetime import datetime

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'used': memory_used,
            'cached': memory_cached,
            'total': memory_total,
            'free': memory_total - memory_cached
        }
    return None

def check_system_memory():
    """Check system RAM usage"""
    memory = psutil.virtual_memory()
    return {
        'used': memory.used / 1e9,
        'total': memory.total / 1e9,
        'percent': memory.percent,
        'available': memory.available / 1e9
    }

def run_training_script(script_name, dataset_name):
    """Run training script with monitoring"""
    
    print(f"\n🚀 Starting {dataset_name} Training...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check initial memory
    gpu_mem = check_gpu_memory()
    sys_mem = check_system_memory()
    
    print(f"📊 Initial Memory Status:")
    if gpu_mem:
        print(f"  GPU: {gpu_mem['used']:.1f}GB used / {gpu_mem['total']:.1f}GB total")
    print(f"  RAM: {sys_mem['used']:.1f}GB used / {sys_mem['total']:.1f}GB total ({sys_mem['percent']:.1f}%)")
    
    # Run training
    try:
        print(f"\n🏃 Running: python {script_name}")
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check final memory
        gpu_mem_final = check_gpu_memory()
        sys_mem_final = check_system_memory()
        
        print(f"\n📊 Final Memory Status:")
        if gpu_mem_final:
            print(f"  GPU: {gpu_mem_final['used']:.1f}GB used / {gpu_mem_final['total']:.1f}GB total")
        print(f"  RAM: {sys_mem_final['used']:.1f}GB used / {sys_mem_final['total']:.1f}GB total ({sys_mem_final['percent']:.1f}%)")
        
        if result.returncode == 0:
            print(f"✅ {dataset_name} training completed successfully!")
            print(f"⏱️  Duration: {duration/3600:.1f} hours ({duration/60:.1f} minutes)")
            
            # Save training log
            log_data = {
                'dataset': dataset_name,
                'script': script_name,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_seconds': duration,
                'duration_hours': duration/3600,
                'success': True,
                'initial_memory': {
                    'gpu': gpu_mem,
                    'system': sys_mem
                },
                'final_memory': {
                    'gpu': gpu_mem_final,
                    'system': sys_mem_final
                }
            }
            
            log_file = f"training_log_{dataset_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"📝 Training log saved: {log_file}")
            return True
            
        else:
            print(f"❌ {dataset_name} training failed!")
            print(f"Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name} training timed out (2 hours)")
        return False
    except Exception as e:
        print(f"❌ Error running {dataset_name} training: {e}")
        return False

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cache cleared")

def main():
    """Main training orchestrator"""
    
    print("🧠 NT-ViT Automated Training Pipeline")
    print("=" * 60)
    
    # Check prerequisites
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be slow")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check scripts exist
    scripts = [
        ('train_ntvit_mindbigdata.py', 'MindBigData'),
        ('train_ntvit_crell.py', 'Crell')
    ]
    
    missing_scripts = []
    for script, name in scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing scripts: {missing_scripts}")
        return
    
    # Show training plan
    print(f"\n📋 Training Plan:")
    print(f"  1. MindBigData: ~1174 samples, 30 epochs")
    print(f"  2. Crell: ~640 samples, 30 epochs")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"  Estimated total time: 3-5 hours")
    else:
        print(f"  CPU training - Estimated total time: 12-18 hours")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        return
    
    # Start training
    overall_start = time.time()
    results = {}
    
    # Train MindBigData
    cleanup_gpu_memory()
    results['mindbigdata'] = run_training_script('train_ntvit_mindbigdata.py', 'MindBigData')
    
    # Wait and cleanup between trainings
    if results['mindbigdata']:
        print("\n⏸️  Waiting 30 seconds between trainings...")
        time.sleep(30)
        cleanup_gpu_memory()
    
    # Train Crell
    results['crell'] = run_training_script('train_ntvit_crell.py', 'Crell')
    
    # Final summary
    overall_end = time.time()
    total_duration = overall_end - overall_start
    
    print(f"\n🏁 Training Pipeline Complete!")
    print("=" * 60)
    print(f"⏱️  Total Duration: {total_duration/3600:.1f} hours ({total_duration/60:.1f} minutes)")
    print(f"📊 Results:")
    print(f"  MindBigData: {'✅ Success' if results['mindbigdata'] else '❌ Failed'}")
    print(f"  Crell: {'✅ Success' if results['crell'] else '❌ Failed'}")
    
    # Check output directories
    output_dirs = ['ntvit_mindbigdata_outputs', 'ntvit_crell_outputs']
    print(f"\n📁 Output Directories:")
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            files = list(Path(output_dir).glob('*'))
            print(f"  {output_dir}: {len(files)} files")
            for file in files:
                if file.suffix in ['.pth', '.json']:
                    size = file.stat().st_size / 1e6
                    print(f"    - {file.name}: {size:.1f}MB")
        else:
            print(f"  {output_dir}: Not found")
    
    # Save overall summary
    summary = {
        'pipeline_start': datetime.fromtimestamp(overall_start).isoformat(),
        'pipeline_end': datetime.fromtimestamp(overall_end).isoformat(),
        'total_duration_hours': total_duration/3600,
        'results': results,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }
    
    summary_file = f"training_pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📝 Pipeline summary saved: {summary_file}")
    
    if all(results.values()):
        print(f"\n🎉 All training completed successfully!")
        print(f"📋 Next steps:")
        print(f"  1. Check model outputs in respective directories")
        print(f"  2. Evaluate model performance")
        print(f"  3. Generate synthetic fMRI samples")
    else:
        print(f"\n⚠️  Some training failed - check logs for details")

if __name__ == "__main__":
    main()
