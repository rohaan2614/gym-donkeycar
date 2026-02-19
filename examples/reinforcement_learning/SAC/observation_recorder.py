# TODO: Fix this
import os
import numpy as np
import imageio

class ObsVideoRecorder:
    def __init__(self, env, video_dir, fps=20, prefix="obs"):
        self.env = env
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)
        self.fps = fps
        self.prefix = prefix
        self.ep_idx = 0
        self.writer = None

    def _to_uint8(self, obs):
        # obs often is HxWxC; convert float->uint8 if needed
        frame = obs
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            # assume 0..1 or 0..255 floats
            mx = frame.max() if frame.size else 1.0
            if mx <= 1.0:
                frame = (frame * 255.0)
            frame = frame.clip(0, 255).astype(np.uint8)

        return frame

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # gym sometimes returns obs,info; handle both
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, info = obs
        else:
            info = {}

        self._close_writer()
        path = os.path.join(self.video_dir, f"{self.prefix}_ep{self.ep_idx:05d}.mp4")
        self.writer = imageio.get_writer(path, fps=self.fps)
        self.writer.append_data(self._to_uint8(obs))
        return (obs, info) if info != {} else obs

    def step(self, action):
        out = self.env.step(action)
        # gym API variants
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out

        if self.writer is not None:
            self.writer.append_data(self._to_uint8(obs))

        if done:
            self._close_writer()
            self.ep_idx += 1

        return out

    def _close_writer(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self):
        self._close_writer()
        return self.env.close()

    def __getattr__(self, name):
        # forward any other attributes (action_space, observation_space, etc.)
        return getattr(self.env, name)
