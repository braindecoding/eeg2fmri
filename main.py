#!/usr/bin/env python3
"""
NT-ViT Complete Implementation - MindBigData + Crell + Stimuli
==============================================================

Complete implementation untuk:
1. MindBigData dataset (EP1.01.txt format)
2. Crell dataset (S01.mat format) 
3. Stimuli folders (MindbigdataStimuli + crellStimuli)
4. NT-ViT architecture dengan Domain Matching
5. Training pipeline untuk EEG → fMRI synthesis

Directory structure expected:
datasets/
├── EP1.01.txt                    # MindBigData EEG data
├── S01.mat                       # Crell EEG data
├── MindbigdataStimuli/           # Digit stimuli (0.jpg, 1.jpg, ...)
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── crellStimuli/                 # Letter stimuli (a.png, d.png, ...)
    ├── a.png
    ├── d.png
    └── ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import math
import torchaudio.transforms as T
from pathlib import Path
import json
import scipy.io
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

# Import NT-ViT components from previous implementation
class NTViTEEGToFMRI(nn.Module):
    """NT-ViT Framework untuk EEG → fMRI conversion"""
    
    def __init__(self, 
                 eeg_channels: int,
                 fmri_voxels: int = 15724,
                 mel_bins: int = 128,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12):
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.fmri_voxels = fmri_voxels
        self.mel_bins = mel_bins
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Core NT-ViT components
        self.spectrogrammer = SpectrogramGenerator(eeg_channels, mel_bins)
        self.generator = NTViTGenerator(mel_bins, fmri_voxels, patch_size, embed_dim, num_heads, num_layers)
        self.domain_matcher = DomainMatchingModule(fmri_voxels, embed_dim)
        
    def forward(self, 
                eeg_data: torch.Tensor,
                target_fmri: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """NT-ViT forward pass"""
        
        # Convert EEG to Mel spectrograms
        mel_spectrograms = self.spectrogrammer(eeg_data)
        
        # Generator - ViT encoding dan decoding
        generator_outputs = self.generator(mel_spectrograms)
        synthetic_fmri = generator_outputs['synthetic_fmri']
        eeg_latent = generator_outputs['latent_representation']
        
        outputs = {
            'synthetic_fmri': synthetic_fmri,
            'mel_spectrograms': mel_spectrograms,
            'eeg_latent': eeg_latent
        }
        
        # Domain Matching (training only)
        if self.training and target_fmri is not None:
            dm_outputs = self.domain_matcher(eeg_latent, target_fmri)
            outputs.update({
                'fmri_latent': dm_outputs['fmri_latent'],
                'domain_alignment_loss': dm_outputs['alignment_loss']
            })
        
        return outputs

# Include all NT-ViT component classes here (SpectrogramGenerator, etc.)
class SpectrogramGenerator(nn.Module):
    """Convert EEG waveforms to Mel spectrograms"""
    
    def __init__(self, eeg_channels: int, mel_bins: int = 128):
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.mel_bins = mel_bins
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=500,
            n_fft=256,
            hop_length=128,
            n_mels=mel_bins,
            f_min=0.5,
            f_max=100.0
        )
        
        # Channel fusion
        self.channel_fusion = nn.Conv2d(eeg_channels, 3, kernel_size=1)
        
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Convert EEG to mel spectrograms"""
        batch_size, channels, time_points = eeg_data.shape
        
        mel_spectrograms = []
        for ch in range(channels):
            channel_data = eeg_data[:, ch, :]
            mel_spec = self.mel_transform(channel_data)
            mel_spectrograms.append(mel_spec)
        
        stacked_mels = torch.stack(mel_spectrograms, dim=1)
        fused_mels = self.channel_fusion(stacked_mels)
        fused_mels = torch.log(fused_mels + 1e-8)
        
        return fused_mels

class NTViTGenerator(nn.Module):
    """NT-ViT Generator with ViT Encoder-Decoder"""
    
    def __init__(self, mel_bins, fmri_voxels, patch_size, embed_dim, num_heads, num_layers):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.fmri_voxels = fmri_voxels
        
        # ViT Encoder
        self.encoder = VisionTransformerEncoder(
            img_size=(mel_bins, mel_bins),
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # ViT Decoder
        self.decoder = VisionTransformerDecoder(
            embed_dim=embed_dim,
            output_dim=fmri_voxels,
            num_heads=num_heads,
            num_layers=6
        )
        
    def forward(self, mel_spectrograms: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ViT encoder-decoder"""
        # Ensure square input
        batch_size, channels, height, width = mel_spectrograms.shape
        
        if height != width:
            max_dim = max(height, width)
            padded = torch.zeros(batch_size, channels, max_dim, max_dim, 
                               device=mel_spectrograms.device, dtype=mel_spectrograms.dtype)
            padded[:, :, :height, :width] = mel_spectrograms
            mel_spectrograms = padded
        
        # ViT Encoder
        latent_representation = self.encoder(mel_spectrograms)
        
        # ViT Decoder
        synthetic_fmri = self.decoder(latent_representation)
        
        return {
            'synthetic_fmri': synthetic_fmri,
            'latent_representation': latent_representation
        }

class VisionTransformerEncoder(nn.Module):
    """Vision Transformer Encoder"""
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Return CLS token
        cls_output = self.norm(x[:, 0])
        return cls_output

class VisionTransformerDecoder(nn.Module):
    """Vision Transformer Decoder for fMRI generation"""
    
    def __init__(self, embed_dim, output_dim, num_heads, num_layers):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Learnable queries
        self.fmri_queries = nn.Parameter(torch.randn(1, output_dim // 64, embed_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 64),
            nn.Tanh()
        )
        
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode latent to fMRI volume"""
        batch_size = encoded.shape[0]
        
        memory = encoded.unsqueeze(1)
        queries = self.fmri_queries.expand(batch_size, -1, -1)
        
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        fmri_chunks = self.output_projection(decoded)
        fmri = fmri_chunks.flatten(1)
        
        # Ensure correct output dimension
        if fmri.shape[1] != self.output_dim:
            if fmri.shape[1] < self.output_dim:
                padding = torch.zeros(batch_size, self.output_dim - fmri.shape[1], 
                                    device=fmri.device, dtype=fmri.dtype)
                fmri = torch.cat([fmri, padding], dim=1)
            else:
                fmri = fmri[:, :self.output_dim]
        
        return fmri

class DomainMatchingModule(nn.Module):
    """Domain Matching module for training alignment"""
    
    def __init__(self, fmri_voxels: int, embed_dim: int):
        super().__init__()
        
        # fMRI encoder
        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_voxels, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Alignment networks
        self.eeg_aligner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.fmri_aligner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, eeg_latent: torch.Tensor, target_fmri: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Domain matching between EEG and fMRI latent spaces"""
        fmri_latent = self.fmri_encoder(target_fmri)
        
        aligned_eeg = self.eeg_aligner(eeg_latent)
        aligned_fmri = self.fmri_aligner(fmri_latent)
        
        alignment_loss = self.compute_contrastive_loss(aligned_eeg, aligned_fmri)
        
        return {
            'fmri_latent': fmri_latent,
            'aligned_eeg': aligned_eeg,
            'aligned_fmri': aligned_fmri,
            'alignment_loss': alignment_loss
        }
    
    def compute_contrastive_loss(self, eeg_features: torch.Tensor, fmri_features: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for domain alignment"""
        eeg_norm = F.normalize(eeg_features, dim=1)
        fmri_norm = F.normalize(fmri_features, dim=1)
        
        similarity = torch.matmul(eeg_norm, fmri_norm.T) / self.temperature
        
        batch_size = eeg_features.shape[0]
        labels = torch.arange(batch_size, device=eeg_features.device)
        
        loss_eeg_to_fmri = F.cross_entropy(similarity, labels)
        loss_fmri_to_eeg = F.cross_entropy(similarity.T, labels)
        
        return (loss_eeg_to_fmri + loss_fmri_to_eeg) / 2

# Dataset loading classes
class MindBigDataLoader:
    """Load and process MindBigData from EP1.01.txt"""
    
    def __init__(self, filepath: str, stimuli_dir: str, max_samples: int = 100):
        self.filepath = filepath
        self.stimuli_dir = Path(stimuli_dir)
        self.max_samples = max_samples
        
        # EPOC channels
        self.epoc_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        
        self.samples = []
        self.load_data()
        
    def load_data(self):
        """Load MindBigData samples"""
        print(f"Loading MindBigData from {self.filepath}...")
        
        signals_by_event = defaultdict(lambda: defaultdict(dict))
        
        try:
            with open(self.filepath, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num > self.max_samples * 500:  # Reasonable limit
                        break
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        device = parts[2]
                        
                        if device == "EP":  # EPOC device only
                            event_id = int(parts[1])
                            channel = parts[3]
                            digit_code = int(parts[4])
                            data_str = parts[6]
                            
                            if 0 <= digit_code <= 9 and channel in self.epoc_channels:
                                try:
                                    data_values = [float(x) for x in data_str.split(',')]
                                    signals_by_event[event_id][channel] = {
                                        'code': digit_code,
                                        'data': np.array(data_values)
                                    }
                                except ValueError:
                                    continue
                                    
        except Exception as e:
            print(f"Error loading MindBigData: {e}")
            return
        
        # Process events
        for event_id, channels_data in signals_by_event.items():
            if len(channels_data) == 14:  # All EPOC channels
                channel_signals = []
                digit_code = None
                
                for channel in self.epoc_channels:
                    if channel in channels_data:
                        sample = channels_data[channel]
                        channel_signals.append(sample['data'])
                        if digit_code is None:
                            digit_code = sample['code']
                    else:
                        break
                
                if len(channel_signals) == 14 and digit_code is not None:
                    # Fixed length processing
                    fixed_length = 512  # 4 seconds at 128Hz
                    processed_signals = []
                    
                    for signal in channel_signals:
                        if len(signal) >= fixed_length:
                            processed_signals.append(signal[:fixed_length])
                        else:
                            # Pad with mean
                            padded = np.full(fixed_length, np.mean(signal))
                            padded[:len(signal)] = signal
                            processed_signals.append(padded)
                    
                    # Load corresponding stimulus image
                    stimulus_path = self.stimuli_dir / f"{digit_code}.jpg"
                    if stimulus_path.exists():
                        stimulus_image = self.load_stimulus_image(stimulus_path)
                        
                        self.samples.append({
                            'eeg_data': np.array(processed_signals),
                            'stimulus_code': digit_code,
                            'stimulus_image': stimulus_image,
                            'dataset_type': 'mindbigdata',
                            'event_id': event_id
                        })
                        
                        if len(self.samples) >= self.max_samples:
                            break
        
        print(f"Loaded {len(self.samples)} MindBigData samples")
    
    def load_stimulus_image(self, image_path: Path) -> np.ndarray:
        """Load and process stimulus image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # Standard size
            img_array = np.array(img) / 255.0  # Normalize
            return img_array.transpose(2, 0, 1)  # CHW format
        except Exception as e:
            print(f"Error loading stimulus {image_path}: {e}")
            return np.zeros((3, 224, 224))

class CrellDataLoader:
    """Load and process Crell dataset from S01.mat"""
    
    def __init__(self, filepath: str, stimuli_dir: str, max_samples: int = 100):
        self.filepath = filepath
        self.stimuli_dir = Path(stimuli_dir)
        self.max_samples = max_samples
        
        # Letter mapping
        self.letter_mapping = {100: 'a', 103: 'd', 104: 'e', 105: 'f', 109: 'j',
                              113: 'n', 114: 'o', 118: 's', 119: 't', 121: 'v'}
        
        self.samples = []
        self.load_data()
        
    def load_data(self):
        """Load Crell samples"""
        print(f"Loading Crell data from {self.filepath}...")
        
        try:
            data = scipy.io.loadmat(self.filepath)
            
            for paradigm_key in ['round01_paradigm', 'round02_paradigm']:
                if paradigm_key not in data:
                    continue
                
                paradigm_data = data[paradigm_key]
                if len(paradigm_data) == 0:
                    continue
                
                round_data = paradigm_data[0, 0]
                
                if 'BrainVisionRDA_data' not in round_data.dtype.names:
                    continue
                
                # Extract data
                eeg_data = round_data['BrainVisionRDA_data'].T  # (64, timepoints)
                eeg_times = round_data['BrainVisionRDA_time'].flatten()
                marker_data = round_data['ParadigmMarker_data'].flatten()
                marker_times = round_data['ParadigmMarker_time'].flatten()
                
                # Extract visual epochs
                visual_epochs = self.extract_visual_epochs(
                    eeg_data, eeg_times, marker_data, marker_times
                )
                
                for epoch_data, letter_code in visual_epochs:
                    letter = list(self.letter_mapping.values())[letter_code]
                    
                    # Load corresponding stimulus
                    stimulus_path = self.stimuli_dir / f"{letter}.png"
                    if stimulus_path.exists():
                        stimulus_image = self.load_stimulus_image(stimulus_path)
                        
                        self.samples.append({
                            'eeg_data': epoch_data,
                            'stimulus_code': letter_code,
                            'stimulus_image': stimulus_image,
                            'dataset_type': 'crell',
                            'letter': letter
                        })
                        
                        if len(self.samples) >= self.max_samples:
                            break
                
                if len(self.samples) >= self.max_samples:
                    break
                    
        except Exception as e:
            print(f"Error loading Crell data: {e}")
            return
        
        print(f"Loaded {len(self.samples)} Crell samples")
    
    def extract_visual_epochs(self, eeg_data, eeg_times, marker_data, marker_times):
        """Extract visual epochs from Crell data"""
        epochs = []
        
        # Find letter presentation events
        letter_events = []
        current_letter = None
        fade_in_time = None
        fade_out_time = None
        
        for marker, marker_time in zip(marker_data, marker_times):
            if marker >= 100:  # Letter code
                current_letter = self.letter_mapping.get(marker, None)
            elif marker == 1:  # Fade in start
                fade_in_time = marker_time
            elif marker == 3:  # Fade out start
                fade_out_time = marker_time
                
                if current_letter is not None and fade_in_time is not None:
                    letter_events.append({
                        'letter': current_letter,
                        'start_time': fade_in_time,
                        'end_time': fade_out_time
                    })
                
                # Reset
                current_letter = None
                fade_in_time = None
                fade_out_time = None
        
        # Extract EEG epochs
        for event in letter_events[:self.max_samples]:
            start_time = event['start_time']
            end_time = event['end_time']
            
            start_idx = np.searchsorted(eeg_times, start_time)
            end_idx = np.searchsorted(eeg_times, end_time)
            
            if end_idx > start_idx and (end_idx - start_idx) > 100:
                epoch_data = eeg_data[:, start_idx:end_idx]
                
                # Resample to fixed length
                target_length = 1000  # 2 seconds at 500Hz
                if epoch_data.shape[1] != target_length:
                    old_indices = np.linspace(0, epoch_data.shape[1]-1, epoch_data.shape[1])
                    new_indices = np.linspace(0, epoch_data.shape[1]-1, target_length)
                    
                    resampled_epoch = np.zeros((64, target_length))
                    for ch in range(64):
                        f = interp1d(old_indices, epoch_data[ch, :], kind='linear')
                        resampled_epoch[ch, :] = f(new_indices)
                    
                    epoch_data = resampled_epoch
                
                # Convert letter to numeric
                letter_to_num = {'a': 0, 'd': 1, 'e': 2, 'f': 3, 'j': 4,
                               'n': 5, 'o': 6, 's': 7, 't': 8, 'v': 9}
                letter_label = letter_to_num.get(event['letter'], 0)
                
                epochs.append((epoch_data, letter_label))
        
        return epochs
    
    def load_stimulus_image(self, image_path: Path) -> np.ndarray:
        """Load and process stimulus image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            return img_array.transpose(2, 0, 1)  # CHW format
        except Exception as e:
            print(f"Error loading stimulus {image_path}: {e}")
            return np.zeros((3, 224, 224))

class EEGFMRIDataset(Dataset):
    """Combined dataset for EEG-fMRI training"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # EEG data
        eeg_data = torch.tensor(sample['eeg_data'], dtype=torch.float32)
        
        # Normalize EEG
        eeg_data = (eeg_data - eeg_data.mean()) / (eeg_data.std() + 1e-8)
        
        # Stimulus image (used to create synthetic fMRI for training)
        stimulus_image = torch.tensor(sample['stimulus_image'], dtype=torch.float32)
        
        # Create synthetic fMRI target (placeholder - in real scenario would be real fMRI)
        synthetic_fmri_target = self.create_synthetic_fmri_target(stimulus_image)
        
        return {
            'eeg_data': eeg_data,
            'stimulus_image': stimulus_image,
            'synthetic_fmri_target': synthetic_fmri_target,
            'stimulus_code': sample['stimulus_code'],
            'dataset_type': sample['dataset_type']
        }
    
    def create_synthetic_fmri_target(self, stimulus_image: torch.Tensor) -> torch.Tensor:
        """Create synthetic fMRI target from stimulus image (placeholder)"""
        # This is a placeholder - in real scenario, you'd have real fMRI data
        # Here we create a simple encoding based on stimulus features
        
        # Simple encoding based on image statistics
        mean_intensity = stimulus_image.mean()
        std_intensity = stimulus_image.std()
        
        # Create pseudo-fMRI based on image features
        fmri_target = torch.randn(15724) * 0.1  # Base noise
        
        # Add patterns based on stimulus
        fmri_target[:1000] += mean_intensity * 0.5  # Visual areas
        fmri_target[1000:2000] += std_intensity * 0.3  # Processing areas
        
        return fmri_target

# Training and evaluation functions
def create_data_loaders(datasets_dir: str, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for MindBigData and Crell"""
    
    datasets_path = Path(datasets_dir)
    
    # Load MindBigData
    mindbig_loader = MindBigDataLoader(
        filepath=str(datasets_path / "EP1.01.txt"),
        stimuli_dir=str(datasets_path / "MindbigdataStimuli"),
        max_samples=50
    )
    
    # Load Crell
    crell_loader = CrellDataLoader(
        filepath=str(datasets_path / "S01.mat"),
        stimuli_dir=str(datasets_path / "crellStimuli"),
        max_samples=50
    )
    
    # Combine samples
    all_samples = mindbig_loader.samples + crell_loader.samples
    
    # Split train/val
    train_samples = all_samples[:int(0.8 * len(all_samples))]
    val_samples = all_samples[int(0.8 * len(all_samples)):]
    
    # Create datasets
    train_dataset = EEGFMRIDataset(train_samples)
    val_dataset = EEGFMRIDataset(val_samples)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Created data loaders:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  MindBigData: {len(mindbig_loader.samples)} samples")
    print(f"  Crell: {len(crell_loader.samples)} samples")
    
    return train_loader, val_loader

def train_ntvit_model(datasets_dir: str, 
                     output_dir: str = "outputs",
                     num_epochs: int = 50,
                     device: str = 'cuda'):
    """Complete training pipeline untuk NT-ViT"""
    
    print("🧠 NT-ViT Training Pipeline")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    train_loader, val_loader = create_data_loaders(datasets_dir, batch_size=4)
    
    # Create models untuk both datasets
    mindbig_model = NTViTEEGToFMRI(eeg_channels=14).to(device)
    crell_model = NTViTEEGToFMRI(eeg_channels=64).to(device)
    
    # Optimizers
    mindbig_optimizer = torch.optim.AdamW(mindbig_model.parameters(), lr=1e-4)
    crell_optimizer = torch.optim.AdamW(crell_model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        mindbig_model.train()
        crell_model.train()
        
        epoch_losses = {'mindbigdata': [], 'crell': []}
        
        for batch_idx, batch in enumerate(train_loader):
            eeg_data = batch['eeg_data'].to(device)
            target_fmri = batch['synthetic_fmri_target'].to(device)
            dataset_types = batch['dataset_type']
            
            # Separate batch by dataset type
            mindbig_mask = [dt == 'mindbigdata' for dt in dataset_types]
            crell_mask = [dt == 'crell' for dt in dataset_types]
            
            # Train MindBigData model
            if any(mindbig_mask):
                mindbig_indices = [i for i, mask in enumerate(mindbig_mask) if mask]
                mindbig_eeg = eeg_data[mindbig_indices]
                mindbig_target = target_fmri[mindbig_indices]
                
                mindbig_optimizer.zero_grad()
                outputs = mindbig_model(mindbig_eeg, mindbig_target)
                
                recon_loss = F.mse_loss(outputs['synthetic_fmri'], mindbig_target)
                domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                total_loss = recon_loss + 0.1 * domain_loss
                
                total_loss.backward()
                mindbig_optimizer.step()
                
                epoch_losses['mindbigdata'].append(total_loss.item())
            
            # Train Crell model
            if any(crell_mask):
                crell_indices = [i for i, mask in enumerate(crell_mask) if mask]
                crell_eeg = eeg_data[crell_indices]
                crell_target = target_fmri[crell_indices]
                
                crell_optimizer.zero_grad()
                outputs = crell_model(crell_eeg, crell_target)
                
                recon_loss = F.mse_loss(outputs['synthetic_fmri'], crell_target)
                domain_loss = outputs.get('domain_alignment_loss', torch.tensor(0.0, device=device))
                total_loss = recon_loss + 0.1 * domain_loss
                
                total_loss.backward()
                crell_optimizer.step()
                
                epoch_losses['crell'].append(total_loss.item())
        
        # Print epoch results
        if epoch_losses['mindbigdata']:
            avg_mindbig_loss = np.mean(epoch_losses['mindbigdata'])
            print(f"  MindBigData loss: {avg_mindbig_loss:.4f}")
        
        if epoch_losses['crell']:
            avg_crell_loss = np.mean(epoch_losses['crell'])
            print(f"  Crell loss: {avg_crell_loss:.4f}")
        
        # Save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(mindbig_model.state_dict(), 
                      output_path / f"ntvit_mindbigdata_epoch_{epoch+1}.pth")
            torch.save(crell_model.state_dict(), 
                      output_path / f"ntvit_crell_epoch_{epoch+1}.pth")
            print(f"  ✓ Saved model checkpoints")
    
    # Final model save
    torch.save(mindbig_model.state_dict(), output_path / "ntvit_mindbigdata_final.pth")
    torch.save(crell_model.state_dict(), output_path / "ntvit_crell_final.pth")
    
    print(f"\n✅ Training complete! Models saved to {output_path}")
    
    return mindbig_model, crell_model

def generate_synthetic_fmri(model: NTViTEEGToFMRI,
                           eeg_data: torch.Tensor,
                           output_path: str,
                           dataset_type: str):
    """Generate synthetic fMRI from trained model"""
    
    model.eval()
    with torch.no_grad():
        outputs = model(eeg_data)
        synthetic_fmri = outputs['synthetic_fmri']
    
    # Save outputs
    for i, fmri in enumerate(synthetic_fmri):
        fmri_numpy = fmri.cpu().numpy()
        
        # Save fMRI
        filename = f"{dataset_type}_synthetic_fmri_{i:03d}.npy"
        filepath = Path(output_path) / filename
        np.save(filepath, fmri_numpy)
        
        # Save metadata
        metadata = {
            'model': 'NT-ViT',
            'dataset_type': dataset_type,
            'fmri_shape': fmri_numpy.shape,
            'fmri_voxels': len(fmri_numpy),
            'value_range': [float(fmri_numpy.min()), float(fmri_numpy.max())],
            'compatible_with': 'MindEye/NSD format'
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved: {filename}")

# Main execution
def main():
    """Main function untuk complete NT-ViT pipeline"""
    
    print("🧠 NT-ViT Complete Implementation")
    print("=" * 50)
    
    # Configuration
    datasets_dir = "datasets"  # Directory containing EP1.01.txt, S01.mat, stimuli folders
    output_dir = "ntvit_outputs"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if datasets exist
    datasets_path = Path(datasets_dir)
    required_files = [
        "EP1.01.txt",
        "S01.mat", 
        "MindbigdataStimuli",
        "crellStimuli"
    ]
    
    missing_files = []
    for file in required_files:
        if not (datasets_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing required files/directories:")
        for file in missing_files:
            print(f"  - {datasets_path / file}")
        print(f"\nPlease ensure you have the correct directory structure:")
        print(f"datasets/")
        print(f"├── EP1.01.txt")
        print(f"├── S01.mat")
        print(f"├── MindbigdataStimuli/")
        print(f"│   ├── 0.jpg")
        print(f"│   ├── 1.jpg")
        print(f"│   └── ...")
        print(f"└── crellStimuli/")
        print(f"    ├── a.png")
        print(f"    ├── d.png")
        print(f"    └── ...")
        return
    
    # Train models
    print(f"\n🚀 Starting training...")
    mindbig_model, crell_model = train_ntvit_model(
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        num_epochs=20,  # Reduced for demo
        device=device
    )
    
    # Generate sample outputs
    print(f"\n🔬 Generating sample synthetic fMRI...")
    
    # Test MindBigData model
    test_eeg_mindbig = torch.randn(3, 14, 512).to(device)
    generate_synthetic_fmri(
        mindbig_model, test_eeg_mindbig, output_dir, "mindbigdata"
    )
    
    # Test Crell model
    test_eeg_crell = torch.randn(3, 64, 1000).to(device)
    generate_synthetic_fmri(
        crell_model, test_eeg_crell, output_dir, "crell"
    )
    
    print(f"\n✅ NT-ViT pipeline complete!")
    print(f"📁 Output files in: {output_dir}/")
    print(f"  • Trained models: ntvit_*.pth")
    print(f"  • Synthetic fMRI: *_synthetic_fmri_*.npy")
    print(f"  • Metadata: *.json")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Use synthetic fMRI files with MindEye for image reconstruction")
    print(f"  2. Evaluate reconstruction quality")
    print(f"  3. Fine-tune models based on results")

if __name__ == "__main__":
    main()