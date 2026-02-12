import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import os


def extract_cte(info: dict):
    if not isinstance(info, dict):
        return None
    for k in ("cte", "CTE", "cross_track_error", "cross_track_err", "crossTrackError"):
        if k in info:
            try:
                return float(info[k])
            except Exception:
                return None
    return None

class CTETrainingLogger(BaseCallback):
    """
    Logs CTE during training only.
    - Writes last CTE and rolling mean(|cte|) to TensorBoard
    - Optionally prints to terminal
    """
    def __init__(self, tb_every_steps=50, print_every_steps=500, verbose=0):
        super().__init__(verbose=verbose)
        self.tb_every_steps = int(tb_every_steps)
        self.print_every_steps = int(print_every_steps)
        self.buf = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)  # VecEnv -> list[dict]
        if infos:
            cte = extract_cte(infos[0])
            if cte is not None:
                self.buf.append(cte)

                # TensorBoard scalars
                if (self.num_timesteps % self.tb_every_steps) == 0:
                    self.logger.record("cte/last", float(cte))
                    # rolling mean of abs(cte)
                    recent = self.buf[-min(len(self.buf), 200):]
                    self.logger.record("cte/abs_mean_200", float(np.mean(np.abs(recent))))

        # Terminal print
        if (self.num_timesteps % self.print_every_steps) == 0 and self.buf:
            recent = self.buf[-min(len(self.buf), 200):]
            mean_abs = float(np.mean(np.abs(recent)))
            last = float(self.buf[-1])
            # print(f"[TRAIN t={self.num_timesteps:07d}] cte_last={last:+.3f}  |cte|_mean(last200)={mean_abs:.3f}")

        return True
    
class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_reward = 0.0
    
    def __reset_reward(self):
        self.ep_reward = 0.0

    def _on_step(self):
        self.ep_reward += float(self.locals['rewards'][0])
        step_rew = float(self.locals['rewards'][0])
        self.logger.record("train/step_reward", step_rew)

        dones = self.locals['dones']
        if dones[0]:
            self.logger.record('rollout/ep_rew_total', self.ep_reward)
            self.__reset_reward()
            
        return True
    

from pathlib import Path
from typing import Optional, Union

from pathlib import Path

from pathlib import Path
from typing import Optional, Union

def find_model_dir(
    keyword: str,
    root: Union[str, Path] = ".",
    verbose: bool = False
) -> Optional[Path]:
    root = Path(root).resolve()
    kw = keyword.casefold()

    if verbose:
        print("Searching under:", root)
        print("Root exists:", root.exists(), "is_dir:", root.is_dir())

    runs_dir = root / "runs"
    if verbose:
        print("runs exists:", runs_dir.exists(), "is_dir:", runs_dir.is_dir())

    for path in root.rglob("*"):
        if path.is_dir() and kw in path.name.casefold():
            if verbose:
                print("FOUND (abs):", path)

            # 🔑 strip the prefix (current dir)
            rel_path = path.relative_to(root)

            if verbose:
                print("FOUND (rel):", rel_path)

            return rel_path

    if verbose:
        print("No match found.")
    return None

from pathlib import Path
import re
from typing import Optional, Union

_CHECKPOINT_PATTERNS = [
    "*checkpoint*.zip",   # your training_checkpoint_XXXX_steps.zip
    "*_steps.zip",        # generic
]

def resolve_model_path_from_run_dir(run_dir: Union[str, Path]) -> Optional[Path]:
    """
    Given a run directory, return the best model path:
      1) final.zip if present
      2) else the checkpoint zip with the largest step count (parsed from filename)
      3) else best_model.zip if present
      4) else None
    """
    run_dir = Path(run_dir)

    # 1) final.zip
    final_zip = run_dir / "final.zip"
    if final_zip.is_file():
        return final_zip

    # (optional) common SB3 eval artifact name
    best_zip = run_dir / "best_model.zip"
    # don't return it yet; use it as fallback after checkpoints

    # 2) checkpoints
    candidates: list[Path] = []
    for pat in _CHECKPOINT_PATTERNS:
        candidates.extend(run_dir.glob(pat))

    # de-dup
    candidates = list(dict.fromkeys([p for p in candidates if p.is_file()]))

    if candidates:
        def step_key(p: Path) -> int:
            name = p.name
            # Prefer explicit "(\d+)_steps"
            m = re.search(r"(\d+)_steps", name)
            if m:
                return int(m.group(1))
            # Otherwise grab the last number in the filename as a fallback
            nums = re.findall(r"\d+", name)
            return int(nums[-1]) if nums else -1

        best_ckpt = max(candidates, key=step_key)
        if step_key(best_ckpt) >= 0:
            return best_ckpt

    # 3) fallback to best_model.zip if present
    if best_zip.is_file():
        return best_zip

    return None