#!/usr/bin/env python3
"""
Wrapper script that applies the patch and runs training in the same process
"""
import sys
import os

# Add toy_demo_rubric to path
sys.path.insert(0, os.path.join(os.getcwd(), 'toy_demo_rubric'))

# Import and apply the patch BEFORE importing verl modules
print("=" * 60)
print("🔧 Applying RayPPOTrainer patch...")
import patch_ray_trainer
print("=" * 60)

# Now run the main training
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
