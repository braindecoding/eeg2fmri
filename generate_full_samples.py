#!/usr/bin/env python3
"""
Generate Full Synthetic fMRI Samples from Trained NT-ViT Models
Uses real EEG data instead of random samples
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')
from train_ntvit import NTViTEEGToFMRI, MindBigDataLoader, CrellDataLoader, generate_synthetic_fmri

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets_dir = Path('datasets')
    output_dir = 'ntvit_outputs'
    
    print(f"🧠 NT-ViT Full Sample Generator")
    print(f"=" * 50)
    print(f"Device: {device}")
    print(f"Datasets: {datasets_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load trained models
    print(f"\n📥 Loading trained models...")
    
    try:
        mindbig_model = NTViTEEGToFMRI(eeg_channels=14, fmri_voxels=15724).to(device)
        mindbig_model.load_state_dict(torch.load('ntvit_outputs_backup/ntvit_mindbigdata_final.pth', map_location=device))
        mindbig_model.eval()
        print(f"✅ Loaded MindBigData model")
        
        crell_model = NTViTEEGToFMRI(eeg_channels=64, fmri_voxels=15724).to(device)
        crell_model.load_state_dict(torch.load('ntvit_outputs_backup/ntvit_crell_final.pth', map_location=device))
        crell_model.eval()
        print(f"✅ Loaded Crell model")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    print(f"\n🧪 Generating synthetic fMRI from full datasets...")
    
    # Process MindBigData
    print(f"\n📊 Processing MindBigData dataset...")
    try:
        mindbig_loader = MindBigDataLoader(
            filepath=str(datasets_dir / "EP1.01.txt"),
            stimuli_dir=str(datasets_dir / "MindbigdataStimuli"),
            max_samples=50
        )
        
        if mindbig_loader.samples:
            # Extract EEG data from all samples
            eeg_data_list = []
            for sample in mindbig_loader.samples:
                eeg_tensor = torch.tensor(sample['eeg_data'], dtype=torch.float32)
                eeg_data_list.append(eeg_tensor)
            
            # Stack into batch tensor
            test_eeg_mindbig = torch.stack(eeg_data_list).to(device)
            print(f"  📈 Loaded {len(test_eeg_mindbig)} MindBigData EEG samples: {test_eeg_mindbig.shape}")
            
            # Generate synthetic fMRI
            generate_synthetic_fmri(
                mindbig_model, test_eeg_mindbig, output_dir, "mindbigdata"
            )
            print(f"  ✅ Generated {len(test_eeg_mindbig)} MindBigData synthetic fMRI samples")
        else:
            print(f"  ⚠️ No MindBigData samples found")
            
    except Exception as e:
        print(f"  ❌ Error processing MindBigData: {e}")
    
    # Process Crell
    print(f"\n📊 Processing Crell dataset...")
    try:
        crell_loader = CrellDataLoader(
            filepath=str(datasets_dir / "S01.mat"),
            stimuli_dir=str(datasets_dir / "crellStimuli"),
            max_samples=50
        )
        
        if crell_loader.samples:
            # Extract EEG data from all samples
            eeg_data_list = []
            for sample in crell_loader.samples:
                eeg_tensor = torch.tensor(sample['eeg_data'], dtype=torch.float32)
                eeg_data_list.append(eeg_tensor)
            
            # Stack into batch tensor
            test_eeg_crell = torch.stack(eeg_data_list).to(device)
            print(f"  📈 Loaded {len(test_eeg_crell)} Crell EEG samples: {test_eeg_crell.shape}")
            
            # Generate synthetic fMRI
            generate_synthetic_fmri(
                crell_model, test_eeg_crell, output_dir, "crell"
            )
            print(f"  ✅ Generated {len(test_eeg_crell)} Crell synthetic fMRI samples")
        else:
            print(f"  ⚠️ No Crell samples found")
            
    except Exception as e:
        print(f"  ❌ Error processing Crell: {e}")
    
    print(f"\n✅ Full sample generation complete!")
    print(f"📁 Check {output_dir}/ for all generated samples")
    
    # Count generated files
    output_path = Path(output_dir)
    npy_files = list(output_path.glob("*_synthetic_fmri_*.npy"))
    json_files = list(output_path.glob("*_synthetic_fmri_*.json"))
    
    print(f"📊 Generated files:")
    print(f"  • {len(npy_files)} .npy files")
    print(f"  • {len(json_files)} .json files")
    print(f"  • Total: {len(npy_files) + len(json_files)} files")

if __name__ == "__main__":
    main()
