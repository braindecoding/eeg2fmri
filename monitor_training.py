#!/usr/bin/env python3
"""
Monitor NT-ViT Training Progress
===============================

Real-time monitoring of training progress, GPU usage, and output files.
"""

import time
import torch
import psutil
import json
from pathlib import Path
from datetime import datetime
import subprocess

def check_gpu_status():
    """Check GPU memory and utilization"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Try to get GPU utilization using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_util, mem_used_mb, mem_total_mb = result.stdout.strip().split(', ')
                gpu_utilization = int(gpu_util)
                mem_used_nvidia = float(mem_used_mb) / 1024  # Convert MB to GB
                mem_total_nvidia = float(mem_total_mb) / 1024
            else:
                gpu_utilization = None
                mem_used_nvidia = memory_cached
                mem_total_nvidia = memory_total
        except:
            gpu_utilization = None
            mem_used_nvidia = memory_cached
            mem_total_nvidia = memory_total
        
        return {
            'available': True,
            'utilization': gpu_utilization,
            'memory_used': mem_used_nvidia,
            'memory_total': mem_total_nvidia,
            'memory_percent': (mem_used_nvidia / mem_total_nvidia) * 100,
            'torch_allocated': memory_used,
            'torch_cached': memory_cached
        }
    return {'available': False}

def check_system_status():
    """Check system resources"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        'cpu_percent': cpu_percent,
        'memory_used': memory.used / 1e9,
        'memory_total': memory.total / 1e9,
        'memory_percent': memory.percent,
        'memory_available': memory.available / 1e9
    }

def check_training_outputs():
    """Check training output directories and files"""
    output_dirs = {
        'mindbigdata': 'ntvit_mindbigdata_outputs',
        'crell': 'ntvit_crell_outputs'
    }
    
    status = {}
    
    for dataset, output_dir in output_dirs.items():
        dir_path = Path(output_dir)
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            
            # Check for specific files
            checkpoints = list(dir_path.glob('checkpoint_*.pth'))
            best_model = dir_path / 'best_model.pth'
            final_model = dir_path / 'final_model.pth'
            history = dir_path / 'training_history.json'
            
            # Get latest checkpoint info
            latest_checkpoint = None
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            
            # Read training history if available
            training_progress = None
            if history.exists():
                try:
                    with open(history, 'r') as f:
                        training_progress = json.load(f)
                except:
                    pass
            
            status[dataset] = {
                'directory_exists': True,
                'total_files': len(files),
                'checkpoints': len(checkpoints),
                'latest_checkpoint': latest_checkpoint.name if latest_checkpoint else None,
                'best_model_exists': best_model.exists(),
                'final_model_exists': final_model.exists(),
                'history_exists': history.exists(),
                'training_progress': training_progress
            }
        else:
            status[dataset] = {
                'directory_exists': False,
                'total_files': 0,
                'checkpoints': 0,
                'latest_checkpoint': None,
                'best_model_exists': False,
                'final_model_exists': False,
                'history_exists': False,
                'training_progress': None
            }
    
    return status

def check_training_logs():
    """Check for training log files"""
    log_files = list(Path('.').glob('training_log_*.json'))
    pipeline_logs = list(Path('.').glob('training_pipeline_summary_*.json'))
    
    latest_log = None
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    
    latest_pipeline = None
    if pipeline_logs:
        latest_pipeline = max(pipeline_logs, key=lambda x: x.stat().st_mtime)
    
    return {
        'training_logs': len(log_files),
        'pipeline_logs': len(pipeline_logs),
        'latest_log': latest_log.name if latest_log else None,
        'latest_pipeline': latest_pipeline.name if latest_pipeline else None
    }

def display_status():
    """Display comprehensive training status"""
    print(f"\n🧠 NT-ViT Training Monitor - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # GPU Status
    gpu_status = check_gpu_status()
    print(f"🚀 GPU Status:")
    if gpu_status['available']:
        util_str = f"{gpu_status['utilization']}%" if gpu_status['utilization'] is not None else "N/A"
        print(f"  Utilization: {util_str}")
        print(f"  Memory: {gpu_status['memory_used']:.1f}GB / {gpu_status['memory_total']:.1f}GB ({gpu_status['memory_percent']:.1f}%)")
        print(f"  PyTorch: {gpu_status['torch_allocated']:.1f}GB allocated, {gpu_status['torch_cached']:.1f}GB cached")
    else:
        print(f"  ❌ GPU not available")
    
    # System Status
    sys_status = check_system_status()
    print(f"\n💻 System Status:")
    print(f"  CPU: {sys_status['cpu_percent']:.1f}%")
    print(f"  RAM: {sys_status['memory_used']:.1f}GB / {sys_status['memory_total']:.1f}GB ({sys_status['memory_percent']:.1f}%)")
    
    # Training Outputs
    training_status = check_training_outputs()
    print(f"\n📁 Training Outputs:")
    
    for dataset, status in training_status.items():
        print(f"  {dataset.upper()}:")
        if status['directory_exists']:
            print(f"    Files: {status['total_files']}")
            print(f"    Checkpoints: {status['checkpoints']}")
            if status['latest_checkpoint']:
                print(f"    Latest: {status['latest_checkpoint']}")
            print(f"    Best model: {'✅' if status['best_model_exists'] else '❌'}")
            print(f"    Final model: {'✅' if status['final_model_exists'] else '❌'}")
            
            # Show training progress if available
            if status['training_progress']:
                progress = status['training_progress']
                if 'train_losses' in progress and progress['train_losses']:
                    latest_loss = progress['train_losses'][-1]
                    epoch = len(progress['train_losses'])
                    total_epochs = progress.get('num_epochs', 'Unknown')
                    print(f"    Progress: Epoch {epoch}/{total_epochs}, Loss: {latest_loss:.4f}")
                if 'best_val_loss' in progress:
                    print(f"    Best Val Loss: {progress['best_val_loss']:.4f}")
        else:
            print(f"    ❌ Output directory not found")
    
    # Training Logs
    log_status = check_training_logs()
    print(f"\n📝 Training Logs:")
    print(f"  Training logs: {log_status['training_logs']}")
    print(f"  Pipeline logs: {log_status['pipeline_logs']}")
    if log_status['latest_log']:
        print(f"  Latest log: {log_status['latest_log']}")

def monitor_training(interval=30, duration=None):
    """Monitor training with specified interval"""
    
    print(f"🔍 Starting NT-ViT Training Monitor")
    print(f"📊 Update interval: {interval} seconds")
    if duration:
        print(f"⏱️  Monitor duration: {duration} seconds")
    print(f"🛑 Press Ctrl+C to stop monitoring")
    
    start_time = time.time()
    
    try:
        while True:
            display_status()
            
            # Check if duration exceeded
            if duration and (time.time() - start_time) > duration:
                print(f"\n⏱️  Monitoring duration exceeded ({duration}s)")
                break
            
            print(f"\n⏸️  Waiting {interval} seconds for next update...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
    
    print(f"\n✅ Monitoring session ended")

def main():
    """Main monitoring function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Monitor NT-ViT Training Progress')
    parser.add_argument('--interval', '-i', type=int, default=30, 
                       help='Update interval in seconds (default: 30)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Total monitoring duration in seconds (default: unlimited)')
    parser.add_argument('--once', action='store_true',
                       help='Show status once and exit')
    
    args = parser.parse_args()
    
    if args.once:
        display_status()
    else:
        monitor_training(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()
