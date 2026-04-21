#!/usr/bin/env python3
"""CLI: Download subset of datasets for baseline evaluation."""

import os
import json
import argparse
from datasets import load_dataset
import torchaudio
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Download baseline evaluation datasets")
    parser.add_argument("--output-dir", type=str, default="data/baseline_eval", help="Where to save datasets")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to pull per dataset")
    return parser.parse_args()

def process_dataset(dataset_name, split, output_dir, samples, audio_col="audio", text_col="text"):
    """Download and process a dataset split into the baseline dir."""
    print(f"Loading {dataset_name} ({split})...")
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    
    count = 0
    with open(metadata_path, 'a', encoding='utf-8') as meta_file:
        for item in ds:
            if count >= samples:
                break
            
            # Extract audio
            if audio_col not in item or text_col not in item:
                continue
                
            audio_data = item[audio_col]
            text = item[text_col]
            
            # The datasets library typically returns a dict with 'array' and 'sampling_rate'
            if isinstance(audio_data, dict) and 'array' in audio_data:
                waveform = torch.tensor(audio_data['array']).unsqueeze(0)
                sr = audio_data['sampling_rate']
            else:
                continue
            
            # Resample strictly to 24000Hz mono since Moshi expects that
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            file_name_short = f"{dataset_name.replace('/', '_')}_{count}.wav"
            file_path = os.path.join(output_dir, file_name_short)
            
            torchaudio.save(file_path, waveform, 24000)
            
            # Write metadata
            meta_obj = {
                "file_name": file_name_short,
                "text": text,
                "dataset": dataset_name
            }
            meta_file.write(json.dumps(meta_obj) + "\n")
            
            count += 1
            
    print(f"Successfully processed {count} samples for {dataset_name} into {output_dir}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean previous metadata
    meta = os.path.join(args.output_dir, "metadata.jsonl")
    if os.path.exists(meta):
        os.remove(meta)
        
    # 1. LibriSpeech (Standard ASR Read Speech)
    process_dataset("librispeech_asr", "test.clean", args.output_dir, args.samples)
    
    # 2. Add any other public HF datasets mimicking your target distribution here
    # Example: A conversational or different read dataset
    # process_dataset("PolyAI/minds14", "en-US.train", args.output_dir, args.samples, audio_col="audio", text_col="english_transcription")
    
    print("Baseline datasets download complete!")

if __name__ == "__main__":
    main()
