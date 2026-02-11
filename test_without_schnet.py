"""
Simple test script - run this to verify NewChemformer can work without SchNetPack
Run: python test_without_schnet.py
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("Testing NewChemformer WITHOUT SchNetPack (Geometry Disabled)")
print("=" * 80)

print("\n[Step 1/3] Importing NewChemformer...")
try:
    from molbart.models.newchemformer import NewChemformer
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nMake sure you're in the chemformer environment:")
    print("  conda activate chemformer")
    sys.exit(1)

print("\n[Step 2/3] Creating mock config...")
from omegaconf import DictConfig

config = DictConfig({
    "seed": 42,
    "vocabulary_path": "bart_vocab_downstream.json",
    "model_type": "bart",
    "d_model": 128,  # Small for testing
    "n_layers": 2,
    "n_heads": 4,
    "d_feedforward": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "schedule": "cycle",
    "warm_up_steps": 100,
    "clip_grad": 1.0,
    "max_seq_len": 256,
    "task": "backward_prediction",
    "batch_size": 2,
    "n_epochs": 1,
    "acc_batches": 1,
    "n_gpus": 1,
    "n_nodes": 1,
    "train_tokens": None,
    "train_mode": "training",
    "n_beams": 1,
    "mask_prob": 0.1,
    "augmentation_probability": 0.0,
    "model_path": None,
    "data_path": None,
    "datamodule": None,
    "output_directory": "tb_logs_test",
    "activation": "gelu",
    "limit_val_batches": 1.0,
    "check_val_every_n_epoch": 1,
    "data_device": "cpu",
})

print("✓ Config created")

print("\n[Step 3/3] Creating NewChemformer (without SchNet)...")
try:
    # schnet_model=None will disable geometry features
    model = NewChemformer(
        config=config,
        schnet_model=None,  # No SchNet = no geometry
        schnet_dim=128
    )
    print("✓ NewChemformer created successfully")
    print("\n" + "=" * 80)
    print("SUCCESS! NewChemformer works without SchNetPack")
    print("You can now run training with use_geometry=false")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nNext step: Try running actual training:")
print("  cd train")
print("  python train_newchemformer.py data_path=data/uspto_50.pickle use_geometry=false batch_size=2 n_epochs=1")
