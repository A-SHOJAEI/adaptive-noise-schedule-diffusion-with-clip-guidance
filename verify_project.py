#!/usr/bin/env python
"""Project verification script."""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ MISSING {description}: {filepath}")
        return False

def main():
    """Run verification checks."""
    print("="*80)
    print("PROJECT VERIFICATION")
    print("="*80)
    
    checks = []
    
    # Core files
    print("\n1. Core Files:")
    checks.append(check_file_exists("README.md", "README"))
    checks.append(check_file_exists("LICENSE", "License file"))
    checks.append(check_file_exists("requirements.txt", "Requirements"))
    checks.append(check_file_exists("pyproject.toml", "Package config"))
    checks.append(check_file_exists(".gitignore", "Gitignore"))
    
    # Scripts
    print("\n2. Executable Scripts:")
    checks.append(check_file_exists("scripts/train.py", "Training script"))
    checks.append(check_file_exists("scripts/evaluate.py", "Evaluation script"))
    checks.append(check_file_exists("scripts/predict.py", "Prediction script"))
    
    # Configs
    print("\n3. Configuration Files:")
    checks.append(check_file_exists("configs/default.yaml", "Default config"))
    checks.append(check_file_exists("configs/ablation.yaml", "Ablation config"))
    
    # Source modules
    print("\n4. Source Modules:")
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/__init__.py", "Package init"))
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/models/model.py", "Model implementation"))
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/models/components.py", "Custom components"))
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/data/loader.py", "Data loader"))
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/training/trainer.py", "Trainer"))
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/evaluation/metrics.py", "Metrics"))
    checks.append(check_file_exists("src/adaptive_noise_schedule_diffusion_with_clip_guidance/utils/config.py", "Config utils"))
    
    # Tests
    print("\n5. Test Files:")
    checks.append(check_file_exists("tests/conftest.py", "Test fixtures"))
    checks.append(check_file_exists("tests/test_data.py", "Data tests"))
    checks.append(check_file_exists("tests/test_model.py", "Model tests"))
    checks.append(check_file_exists("tests/test_training.py", "Training tests"))
    
    # Import checks
    print("\n6. Import Checks:")
    try:
        sys.path.insert(0, 'src')
        from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import AdaptiveNoiseDiffusionModel
        print("✓ AdaptiveNoiseDiffusionModel imports successfully")
        checks.append(True)
    except Exception as e:
        print(f"✗ Failed to import AdaptiveNoiseDiffusionModel: {e}")
        checks.append(False)
    
    try:
        from adaptive_noise_schedule_diffusion_with_clip_guidance.models.components import AdaptiveScheduleLoss
        print("✓ AdaptiveScheduleLoss imports successfully")
        checks.append(True)
    except Exception as e:
        print(f"✗ Failed to import AdaptiveScheduleLoss: {e}")
        checks.append(False)
    
    # Config validation
    print("\n7. Configuration Validation:")
    try:
        import yaml
        with open('configs/default.yaml') as f:
            config = yaml.safe_load(f)
        assert config['model']['use_adaptive_schedule'] == True
        assert isinstance(config['training']['learning_rate'], float)
        print("✓ Default config valid (adaptive=True)")
        checks.append(True)
    except Exception as e:
        print(f"✗ Default config invalid: {e}")
        checks.append(False)
    
    try:
        with open('configs/ablation.yaml') as f:
            config = yaml.safe_load(f)
        assert config['model']['use_adaptive_schedule'] == False
        print("✓ Ablation config valid (adaptive=False)")
        checks.append(True)
    except Exception as e:
        print(f"✗ Ablation config invalid: {e}")
        checks.append(False)
    
    # Summary
    print("\n" + "="*80)
    total = len(checks)
    passed = sum(checks)
    print(f"VERIFICATION COMPLETE: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ ALL CHECKS PASSED - Project is complete and ready!")
        return 0
    else:
        print(f"✗ {total - passed} checks failed - please review")
        return 1

if __name__ == "__main__":
    sys.exit(main())
