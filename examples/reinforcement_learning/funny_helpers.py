import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
