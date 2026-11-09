"""Check saved checkpoints without repeating any RL training.

Usage: python scripts/verify_models.py
PPO model loading requires stable-baselines3 and gymnasium (pip install -r requirements.txt).
"""
import sys
from pathlib import Path
import zipfile
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import load_dqn, dqn_saving_share


def main():
    missing = []
    for seed in range(5):
        path = ROOT / "models" / f"dqn_best_seed_{seed}.pt"
        if not path.exists():
            missing.append(path.name)
            continue
        q = load_dqn(path)
        high = float(dqn_saving_share(q, [1.0], [1.0])[0])
        low = float(dqn_saving_share(q, [1.0], [0.472])[0])
        assert 0 <= high <= .95 and 0 <= low <= .95
        print(f"DQN seed={seed}: loaded; saving share at asset=1: high={high:.2f}, low={low:.2f}")

    try:
        from stable_baselines3 import PPO
    except ImportError:
        PPO = None
        print("PPO runtime not installed: checking archives only. Install requirements for PPO inference.")

    for seed in range(5):
        path = ROOT / "models" / f"ppo_best_seed_{seed}.zip"
        if not path.exists():
            missing.append(path.name)
            continue
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt PPO checkpoint {path}: {bad}")
            assert {'data', 'policy.pth', '_stable_baselines3_version'}.issubset(archive.namelist())
        if PPO is not None:
            import numpy as np
            model = PPO.load(str(path), device="cpu")
            actions, _ = model.predict(np.array([[1/4.5, 1.0], [1/4.5, 0.472]], dtype=np.float32), deterministic=True)
            assert np.all((actions >= 0) & (actions < 20))
            print(f"PPO seed={seed}: loaded; actions={actions.tolist()}")
        else:
            print(f"PPO seed={seed}: valid SB3 archive (runtime loading not performed)")
    if missing:
        raise FileNotFoundError('Missing model files: ' + ', '.join(missing))
    print("PASS: all 10 model files were found; all available runtime checks passed.")


if __name__ == "__main__":
    main()
