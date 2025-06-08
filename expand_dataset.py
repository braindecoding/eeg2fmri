#!/usr/bin/env python3
"""
Expand CortexFlow dataset with more samples
"""

import sys
sys.path.append('.')
from ntvit_to_cortexflow import CortexFlowGenerator

def main():
    print("🚀 Dataset Expansion Options")
    print("=" * 50)
    print("Current: 100 samples (50 per dataset)")
    print()
    print("🔥 AGGRESSIVE EXPANSION (User Requested):")
    print("   MindBigData: 1,200 samples")
    print("   Crell: 320 samples (FULL dataset)")
    print("   Total: 1,520 samples")
    print()

    # Use user-specified aggressive expansion
    mb_samples, crell_samples = 1200, 320
    
    print(f"\n🔄 Generating expanded dataset:")
    print(f"  MindBigData: {mb_samples} samples")
    print(f"  Crell: {crell_samples} samples")
    print(f"  Total: {mb_samples + crell_samples} samples")
    print()
    
    # Create generator with custom sample sizes
    generator = CortexFlowGenerator()
    
    # Temporarily modify the generate_dataset method
    original_method = generator.generate_dataset
    
    def custom_generate_dataset(dataset_type, n_samples=None):
        if dataset_type == "mindbigdata":
            target_samples = mb_samples
        else:
            target_samples = crell_samples
        
        print(f"\n🔄 Generating {dataset_type} dataset ({target_samples} samples)...")
        samples = generator.load_dataset_samples(dataset_type, max_samples=target_samples)
        
        if not samples:
            print(f"  ❌ No samples found for {dataset_type}")
            return None
        
        print(f"  Loaded {len(samples)} original samples")
        
        # Generate synthetic fMRI
        eeg_data_list = []
        for sample in samples:
            eeg_data_list.append(sample['eeg_data'])
        
        if not eeg_data_list:
            print(f"  ❌ No EEG data found for {dataset_type}")
            return None
        
        import torch
        import numpy as np
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if dataset_type == "mindbigdata":
            model = generator.mindbig_model
        else:
            model = generator.crell_model
        
        # Generate in batches to avoid memory issues
        batch_size = 10
        all_synthetic_fmri = []
        
        for i in range(0, len(eeg_data_list), batch_size):
            batch_eeg = eeg_data_list[i:i+batch_size]
            batch_tensor = torch.stack([torch.tensor(eeg, dtype=torch.float32) for eeg in batch_eeg]).to(device)
            
            with torch.no_grad():
                batch_fmri = model(batch_tensor)['synthetic_fmri']
                all_synthetic_fmri.extend(batch_fmri.cpu().numpy())
        
        synthetic_fmri = np.array(all_synthetic_fmri)
        print(f"  Generated synthetic fMRI shape: {synthetic_fmri.shape}")
        
        # Process stimuli and labels
        stimuli_data, labels = generator.process_stimuli_and_labels(samples, dataset_type)
        print(f"  Processed stimuli shape: {stimuli_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Create train/test split
        (fmri_train, fmri_test, stim_train, stim_test, 
         label_train, label_test) = generator.create_train_test_split_by_stimulus(
            synthetic_fmri, stimuli_data, labels, dataset_type
        )
        
        # Create MATLAB data structure
        matlab_data = {
            'fmriTrn': fmri_train.astype(np.float64),
            'fmriTest': fmri_test.astype(np.float64),
            'stimTrn': stim_train.astype(np.uint8),
            'stimTest': stim_test.astype(np.uint8),
            'labelTrn': label_train.astype(np.uint8),
            'labelTest': label_test.astype(np.uint8)
        }
        
        return matlab_data
    
    # Replace method temporarily
    generator.generate_dataset = custom_generate_dataset
    
    # Generate datasets
    try:
        generator.generate_all("cortexflow_data_expanded")
        print(f"\n✅ Expanded dataset generated successfully!")
        print(f"📁 Output: cortexflow_data_expanded/")
        print(f"📊 Total samples: {mb_samples + crell_samples}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
