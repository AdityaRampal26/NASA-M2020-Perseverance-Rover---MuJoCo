#!/usr/bin/env python3
"""
gym_env.py -- a Gymnasium environment for learning to drive Perseverance.

The task is point-to-goal navigation: start at the origin, reach a random goal
a few metres away without tipping or over-stressing the suspension. Actions go
through the same rocker-bogie steering as teleop and autonav (m2020_drive), so
a learned policy commands arcs and turns in place like the real rover.

    pip install "gymnasium>=1.0"

    import gymnasium as gym
    import gym_env                       # registers Perseverance-Nav-v0
    env = gym.make("Perseverance-Nav-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    python scripts/gym_env.py            # quick self-check + a short random drive

Observation (11 floats)
    goal distance (m), goal bearing (sin, cos) in the rover frame,
    roll and pitch (rad), body yaw rate (rad/s), forward speed (m/s),
    left/right rocker and left/right bogie angles (rad)

Action (2 floats in [-1, 1])
    speed   fraction of the 4.2 cm/s top speed, negative = reverse
    turn    fraction of the maximum yaw rate, positive = left

    |speed| below 0.05 turns in place; otherwise the rover drives an arc
    whose radius is speed / yaw rate, clamped so no wheel exceeds its limit.

Reward
    metres of progress toward the goal each step, minus a small tilt penalty;
    +10 for reaching it (within 0.5 m), -10 for exceeding the tilt limit.

Each action is held for `control_dt` seconds (default 0.5 s, 250 physics
steps). The smooth-wheel model is the default because it simulates about
1.5x faster; pass model="m2020_perseverance.xml" for the cleated wheels, or a
course XML for terrain.
"""

import math
import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from m2020_drive import MAX_WHEEL_RATE, WHEEL_RADIUS, Mobility, hold_pose
from terrain_probe import LIMITS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

TOP_SPEED = MAX_WHEEL_RATE * WHEEL_RADIUS        # 0.042 m/s at the centre
CORNER_RADIUS = 1.59                             # m, corner wheel from centre
MAX_YAW_RATE = TOP_SPEED / CORNER_RADIUS         # corners at full rate
SUSPENSION = {"rocker_l": "LEFT_DIFFERENTIAL", "rocker_r": "RIGHT_DIFFERENTIAL",
              "bogie_l": "LEFT_BOGIE", "bogie_r": "RIGHT_BOGIE"}


class PerseveranceNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, model="m2020_perseverance_smooth.xml", control_dt=0.5,
                 goal_range=(3.0, 6.0), goal_tol=0.5, max_steps=600,
                 tilt_limit_deg=None, render_mode=None, camera=None,
                 width=480, height=360):
        path = model if os.path.isabs(model) else os.path.join(MODELS, model)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.mob = Mobility(self.model)
        self.n_sub = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.goal_range = goal_range
        self.goal_tol = goal_tol
        self.max_steps = max_steps
        self.tilt_limit = math.radians(tilt_limit_deg or LIMITS["tilt_deg"])
        self.render_mode = render_mode
        self.camera = camera
        self._size = (height, width)
        self._renderer = None

        self._stowed = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stowed")
        self._qadr = {k: self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for k, j in SUSPENSION.items()}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        pi = math.pi
        lo = np.array([0, -1, -1, -pi, -pi, -10, -1, -pi, -pi, -pi, -pi], np.float32)
        hi = np.array([1e4, 1, 1, pi, pi, 10, 1, pi, pi, pi, pi], np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    # -- state ---------------------------------------------------------
    def _pose(self):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return self.data.qpos[0], self.data.qpos[1], roll, pitch, yaw

    def _tilt(self, roll, pitch):
        # angle between the body z axis and vertical
        return math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch))))

    def _obs(self):
        x, y, roll, pitch, yaw = self._pose()
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - yaw
        v = self.data.qvel[0] * math.cos(yaw) + self.data.qvel[1] * math.sin(yaw)
        susp = [self.data.qpos[self._qadr[k]] for k in SUSPENSION]
        return np.array([dist, math.sin(bearing), math.cos(bearing), roll, pitch,
                         self.data.qvel[5], v, *susp], dtype=np.float32)

    # -- gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stowed)
        hold_pose(self.model, self.data)
        for _ in range(int(2.0 / self.model.opt.timestep)):     # settle
            mujoco.mj_step(self.model, self.data)
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], float)
        else:
            r = self.np_random.uniform(*self.goal_range)
            a = self.np_random.uniform(-math.pi, math.pi)
            x0, y0 = self.data.qpos[0], self.data.qpos[1]
            self.goal = np.array([x0 + r * math.cos(a), y0 + r * math.sin(a)])
        self.steps = 0
        self._dist = self._obs()[0]
        return self._obs(), {"goal": self.goal.copy()}

    def _apply(self, action):
        speed, turn = np.clip(np.asarray(action, float), -1.0, 1.0)
        v = speed * TOP_SPEED
        w = turn * MAX_YAW_RATE
        if abs(speed) < 0.05:
            self.mob.point_turn(self.data.ctrl, w)
        elif abs(w) < 1e-6:
            self.mob.arc(self.data.ctrl, v, None)
        else:
            # keep the outer corner wheel within its rate: |v| + |w| * 1.59
            scale = min(1.0, TOP_SPEED / (abs(v) + abs(w) * CORNER_RADIUS))
            self.mob.arc(self.data.ctrl, v * scale, v / w)

    def step(self, action):
        self._apply(action)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs = self._obs()
        dist = float(obs[0])
        _, _, roll, pitch, _ = self._pose()
        tilt = self._tilt(roll, pitch)
        reward = (self._dist - dist) - 0.01 * tilt
        self._dist = dist

        reached = dist < self.goal_tol
        tipped = tilt > self.tilt_limit or not np.isfinite(obs).all()
        if reached:
            reward += 10.0
        if tipped:
            reward -= 10.0
        terminated = bool(reached or tipped)
        truncated = self.steps >= self.max_steps and not terminated
        info = {"distance": dist, "tilt_deg": math.degrees(tilt),
                "reached": reached, "tipped": tipped}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self._size)
        if self.camera is None:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = self.data.qpos[:3]
            cam.distance, cam.azimuth, cam.elevation = 9.0, 130.0, -25.0
            self._renderer.update_scene(self.data, camera=cam)
        else:
            self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="Perseverance-Nav-v0", entry_point=PerseveranceNavEnv)


def _self_check():
    from gymnasium.utils.env_checker import check_env
    env = PerseveranceNavEnv()
    check_env(env, skip_render_check=True)
    print("gymnasium env_checker: OK")

    # a hand-written controller, to show the env is solvable: turn toward the
    # goal in place, then drive at it
    obs, info = env.reset(seed=1, options={"goal": (2.0, 1.0)})
    total = 0.0
    for t in range(env.max_steps):
        dist, s, c = obs[0], obs[1], obs[2]
        bearing = math.atan2(s, c)
        if abs(bearing) > 0.25:
            action = (0.0, float(np.sign(bearing)))
        else:
            action = (1.0, float(np.clip(2.0 * bearing, -1, 1)))
        obs, r, term, trunc, info = env.step(action)
        total += r
        if term or trunc:
            break
    print(f"heuristic drive: {t + 1} steps ({(t + 1) * 0.5:.0f} s sim), "
          f"reached={info['reached']}, tipped={info['tipped']}, "
          f"final distance {info['distance']:.2f} m, return {total:.2f}")


if __name__ == "__main__":
    _self_check()
