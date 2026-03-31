#!/usr/bin/env python3
"""
GCN_MA: Dynamic Network Link Prediction
======================================
Main entry point for training and evaluation.

Usage:
    python main.py --dataset CollegeMsg
    python main.py --dataset CollegeMsg --config configs/config.yaml
    python main.py --ablation
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gcn_ma.data_loader import DynamicNetworkDataset, TrainTestSplitter
from gcn_ma.trainer import GCN_MA_Trainer, run_ablation_study


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='GCN_MA Link Prediction')
    parser.add_argument('--dataset', type=str, default='CollegeMsg',
                        choices=['CollegeMsg', 'Mooc_actions', 'Bitcoinotc', 'EUT'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not exists')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['data']['name'] = args.dataset
    config['data']['data_dir'] = args.data_dir
    
    print("="*60)
    print("GCN_MA: Dynamic Network Link Prediction")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Device: {config['training']['device']}")
    print("="*60)
    
    # Initialize dataset
    print("\n[1/5] Loading dataset...")
    dataset = DynamicNetworkDataset(
        name=args.dataset,
        data_dir=args.data_dir
    )
    
    # Download if requested
    if args.download:
        print("Downloading dataset...")
        dataset.download()
    
    # Load/process data
    try:
        graphs = dataset.load_or_process()
        print(f"Loaded {len(graphs)} graph snapshots")
    except FileNotFoundError:
        print(f"Dataset not found. Run with --download to fetch.")
        print(f"Or manually download from SNAP Stanford and place in {dataset.raw_dir}")
        return
    
    # Split data
    print("\n[2/5] Splitting data...")
    splitter = TrainTestSplitter(graphs, train_ratio=config['data']['train_ratio'])
    
    train_graphs = splitter.get_train_graphs()
    val_graphs = splitter.get_test_graphs()[:2]  # First 2 for validation
    test_graphs = splitter.get_test_graphs()[1:]  # Rest for testing
    
    print(f"Train: {len(train_graphs)} snapshots")
    print(f"Val: {len(val_graphs)} snapshots")
    print(f"Test: {len(test_graphs)} snapshots")
    
    if args.ablation:
        print("\n[3/5] Running ablation study...")
        results = run_ablation_study(train_graphs, val_graphs, test_graphs, config)
        
        print("\n" + "="*60)
        print("ABLATION RESULTS")
        print("="*60)
        for model_name, metrics in results.items():
            print(f"{model_name:20s} | AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f}")
    else:
        print("\n[3/5] Initializing trainer...")
        trainer = GCN_MA_Trainer(config)
        
        print("\n[4/5] Training model...")
        train_results = trainer.train(train_graphs, val_graphs, test_graphs)
        
        print("\n[5/5] Final evaluation on test set...")
        best_epoch = train_results['val_auc'].index(max(train_results['val_auc']))
        print(f"\nBest Epoch: {best_epoch + 1}")
        print(f"Test AUC: {train_results['test_auc'][best_epoch]:.4f}")
        print(f"Test AP: {train_results['test_ap'][best_epoch]:.4f}")
        
        # Save results
        import json
        results_path = Path(args.results_dir)
        results_path.mkdir(exist_ok=True)
        
        results_file = results_path / f"{args.dataset}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'best_epoch': best_epoch + 1,
                'test_auc': train_results['test_auc'][best_epoch],
                'test_ap': train_results['test_ap'][best_epoch],
                'all_results': train_results
            }, f, indent=2)
        print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
