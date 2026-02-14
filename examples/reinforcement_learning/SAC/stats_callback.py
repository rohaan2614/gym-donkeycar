import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# TODO: Rework this for SAC
class StatsCallback(BaseCallback):
    """
    Assumes a single environment (n_envs == 1).
    Aggregates over a window of `print_every_steps` callback calls,
    then logs windowed stats to TensorBoard.
    """

    def __init__(self, print_every_steps=500, verbose: bool = True):
        super().__init__()
        self.print_every_steps = int(print_every_steps)
        self.verbose = verbose

        self._steer_buf = []
        self._thr_buf = []
        self._cte_buf = []
        self._fwd_buf = []
        self._lap_time_buf = []
        self._lap_count_buf = []
        self._critic_buf = []

    def _on_step(self) -> bool:
        # ----------------------------
        # ACTIONS (clipped only)
        # ----------------------------
        clipped = self.locals.get("actions", None)
        # print(f'Locals: {self.locals}')
        if clipped is None:
            print(f'Locals: {self.locals}')
            raise RuntimeError("actions not found in callback locals.")

        a = np.asarray(clipped, dtype=float)
        steer = float(a[0, 0])
        thr = float(a[0, 1])

        self._steer_buf.append(steer)
        self._thr_buf.append(thr)

        # ----------------------------
        # CRITIC VALUE
        # ----------------------------
        values = self.locals.get("values", None)
        if values is not None:
            try:
                v = values.detach().cpu().numpy()
            except Exception:
                v = np.asarray(values)
            v = np.asarray(v, dtype=float).reshape(-1)
            critic_val = float(v[0]) if v.size > 0 else np.nan
        else:
            critic_val = np.nan

        self._critic_buf.append(critic_val)

        # ----------------------------
        # INFO (env telemetry)
        # ----------------------------
        infos = self.locals.get("infos", None)
        if infos is None or not isinstance(infos, (list, tuple)) or len(infos) < 1 or not isinstance(infos[0], dict):
            raise RuntimeError("infos not found or invalid in callback locals.")

        info = infos[0]
        self._cte_buf.append(float(info.get("cte", np.nan)))
        self._fwd_buf.append(float(info.get("forward_vel", np.nan)))
        self._lap_time_buf.append(float(info.get("last_lap_time", np.nan)))
        self._lap_count_buf.append(float(info.get("lap_count", np.nan)))

        # ----------------------------
        # FLUSH window every N steps
        # ----------------------------
        if self.n_calls % self.print_every_steps == 0:
            steer_mean = float(np.nanmean(self._steer_buf))
            
            thr_mean   = float(np.nanmean(self._thr_buf))
            thr_min    = float(np.nanmin(self._thr_buf))
            thr_max    = float(np.nanmax(self._thr_buf))

            cte_mean = float(np.nanmean(self._cte_buf))
            
            fwd_mean = float(np.nanmean(self._fwd_buf))
            
            lap_time_mean = float(np.nanmean(self._lap_time_buf))
            lap_count_max = int(np.nanmax(self._lap_count_buf))
            
            critic_mean = float(np.nanmean(self._critic_buf))

            # TensorBoard
            self.logger.record("action/steer_mean", steer_mean)
            self.logger.record("action/throttle_mean", thr_mean)
            # self.logger.record("action/throttle_min", thr_min)
            # self.logger.record("action/throttle_max", thr_max)

            self.logger.record("info/cte_mean", cte_mean)
            self.logger.record("info/forward_vel_mean", fwd_mean)
            self.logger.record("info/last_lap_time_mean", lap_time_mean)
            self.logger.record("info/lap_count_max", lap_count_max)

            self.logger.record("train/critic_value_mean", critic_mean)

            if self.verbose:
                print(
                    f"t={self.model.num_timesteps} "
                    f"[ACTION] steer_mean={steer_mean:+.3f} "
                    f"thr_mean={thr_mean:+.3f} thr_min={thr_min:+.3f} thr_max={thr_max:+.3f} "
                    f"| [INFO] cte_mean={cte_mean:+.3f} fwd_mean={fwd_mean:+.3f} "
                    f"lap_time_mean={lap_time_mean:.3f} laps_max={lap_count_max:.0f} "
                    f"| [CRITIC] value_mean={critic_mean:+.3f}"
                )

            # clear buffers
            self._steer_buf.clear()
            self._thr_buf.clear()
            self._cte_buf.clear()
            self._fwd_buf.clear()
            self._lap_time_buf.clear()
            self._lap_count_buf.clear()
            self._critic_buf.clear()

        return True