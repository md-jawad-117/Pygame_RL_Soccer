"""
soccer_dqn.py - Pygame RL Soccer  (Deep Q-Network,)
=========================================================
The same soccer environment as soccer_rl.py, but the agent now uses a
**Deep Q-Network (DQN)** instead of a tabular Q-table.

Key differences from tabular Q-learning
----------------------------------------
* **Continuous state**: raw numbers (distances, velocities) fed directly into
  the network - no discretisation bins, no aliasing.
* **Neural network Q-function**: a small MLP maps observations -> Q-values for
  all 5 actions in one forward pass.
* **Experience replay**: transitions are stored in a ring buffer and sampled
  randomly for training, breaking temporal correlations.
* **Target network**: a periodically-frozen copy of the network is used for
  TD targets, stabilising training.

Same RL algorithm underneath - Bellman equation, epsilon-greedy, γ-discounted
returns - just with a network as the function approximator.

Controls (identical to soccer_rl.py)
--------------------------------------
  T       - toggle continuous training
  SPACE   - run exactly one training episode (fully rendered)
  G       - cycle speed mode: SLOW -> FAST -> HYPER
  R       - reset environment
  S       - save model and training state
  L       - load model from disk
  H       - toggle help overlay
  Q / ESC - quit (auto-saves first)
"""

import random
import pickle
import math
from collections import deque
from typing import Tuple, List

import numpy as np
import pygame as pg
import torch
import torch.nn as nn
import torch.optim as optim

import config as C

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DEVICE = torch.device("cpu")   # CPU-only; switch to "cuda" if GPU available

# ---------------------------------------------------------------------------
# Helpers (shared geometry - identical to soccer_rl.py)
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def goal_center() -> np.ndarray:
    return np.array(
        [C.FIELD_W - C.GOAL_THICKNESS // 2, (C.GOAL_Y0 + C.GOAL_Y1) / 2.0],
        dtype=float,)


def epsilon_schedule_dqn(step: int) -> float:
    """Linear epsilon-decay over env steps (not episodes) for DQN."""
    if step >= C.DQN_EPS_DECAY_STEPS:
        return C.DQN_EPS_END
    progress = step / C.DQN_EPS_DECAY_STEPS
    return C.DQN_EPS_START - (C.DQN_EPS_START - C.DQN_EPS_END) * progress

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dqn(agent: "DQNAgent", episode: int, total_steps: int) -> None:
    """Save network weights and training counters."""
    try:
        C.SAVES_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
        }, C.DQN_MODEL_PATH)
        with open(C.DQN_STATE_PATH, "wb") as fh:
            pickle.dump({"episode": episode, "total_steps": total_steps}, fh)
        print(f"[DQN] Saved -> {C.SAVES_DIR}  (ep={episode}, steps={total_steps})")
    except Exception as exc:
        print(f"[DQN] Save failed: {exc}")


def load_dqn(agent: "DQNAgent") -> Tuple[int, int]:
    """Load network weights; return (episode, total_steps) or (0, 0)."""
    episode, total_steps = 0, 0
    if C.DQN_MODEL_PATH.exists():
        try:
            ckpt = torch.load(C.DQN_MODEL_PATH, map_location=DEVICE)
            agent.online.load_state_dict(ckpt["online"])
            agent.target.load_state_dict(ckpt["target"])
            agent.optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[DQN] Model loaded <- {C.DQN_MODEL_PATH}")
        except Exception as exc:
            print(f"[DQN] Model load failed: {exc}")
    else:
        print(f"[DQN] No model found at {C.DQN_MODEL_PATH} - starting fresh.")

    if C.DQN_STATE_PATH.exists():
        try:
            with open(C.DQN_STATE_PATH, "rb") as fh:
                obj = pickle.load(fh)
            episode     = obj.get("episode", 0)
            total_steps = obj.get("total_steps", 0)
            print(f"[DQN] State loaded: ep={episode}, steps={total_steps}")
        except Exception as exc:
            print(f"[DQN] State load failed: {exc}")

    return episode, total_steps

# ---------------------------------------------------------------------------
# Environment  (continuous observations - no discretisation)
# ---------------------------------------------------------------------------

Obs = np.ndarray   # shape (DQN_STATE_DIM,)


class SoccerEnvDQN:
    """
    Soccer environment that returns a **continuous observation vector**
    instead of the discrete state tuple used by the tabular agent.

    Observation (10 floats, all normalised to roughly [-1, 1])
    -----------------------------------------------------------
    0  dx        ball_x - agent_x           / (FIELD_W / 2)
    1  dy        ball_y - agent_y           / (FIELD_H / 2)
    2  gx        goal_cx - ball_x           / (FIELD_W / 2)
    3  gy        goal_cy - ball_y           / (FIELD_H / 2)
    4  vx        ball_vel_x                 / BALL_MAX_SPEED
    5  vy        ball_vel_y                 / BALL_MAX_SPEED
    6  ax        agent_x                    / FIELD_W
    7  ay        agent_y                    / FIELD_H
    8  touching  1.0 if agent touches ball, else 0.0
    9  behind    1.0 if agent is behind ball relative to goal, else 0.0
    """

    def __init__(self) -> None:
        self.agent:    np.ndarray = np.zeros(2, dtype=float)
        self.ball:     np.ndarray = np.zeros(2, dtype=float)
        self.ball_vel: np.ndarray = np.zeros(2, dtype=float)
        self.t:    int  = 0
        self.done: bool = False

        self._prev_ball_goal_dist: float = 0.0
        self._prev_ab_dist:        float = 0.0
        self._prev_ball_x:         float = 0.0
        self._prev_align:          float = 0.0
        self._prev_sweet:          float = 0.0

        self.reset()

    def reset(self) -> Obs:
        self.agent = np.array(
            [C.FIELD_MARGIN + 60, C.FIELD_H // 2 + random.randint(-60, 60)],
            dtype=float,)
        if random.random() < 0.6:
            offset = np.array(
                [random.randint(20, 80), random.randint(-60, 60)], dtype=float)
            self.ball = self.agent + offset
        else:
            self.ball = np.array(
                [
                    C.FIELD_W // 2 - random.randint(150, 250),
                    C.FIELD_H // 2 + random.randint(-120, 120),
                ],
                dtype=float,)
        self.ball = np.clip(
            self.ball,
            [C.FIELD_MARGIN + 40, C.FIELD_MARGIN + 40],
            [C.FIELD_W - 80, C.FIELD_H - 40],)
        self.ball_vel = np.zeros(2, dtype=float)
        self.t    = 0
        self.done = False

        self._prev_ball_goal_dist = self._ball_goal_dist()
        self._prev_ab_dist        = self._agent_ball_dist()
        self._prev_ball_x         = float(self.ball[0])
        self._prev_align          = self._alignment_cos()
        self._prev_sweet          = self._sweet_spot_dist()

        return self._obs()

    # ------------------------------------------------------------------
    # Geometry (identical to SoccerEnv)
    # ------------------------------------------------------------------

    def _agent_rect(self) -> pg.Rect:
        hs = C.AGENT_SIZE // 2
        return pg.Rect(int(self.agent[0]) - hs, int(self.agent[1]) - hs,
                       C.AGENT_SIZE, C.AGENT_SIZE)

    def _ball_rect(self) -> pg.Rect:
        hs = C.BALL_SIZE // 2
        return pg.Rect(int(self.ball[0]) - hs, int(self.ball[1]) - hs,
                       C.BALL_SIZE, C.BALL_SIZE)

    def is_touching(self) -> bool:
        return self._agent_rect().colliderect(self._ball_rect())

    def _ball_goal_dist(self) -> float:
        return float(np.linalg.norm(self.ball - goal_center()))

    def _agent_ball_dist(self) -> float:
        return float(np.linalg.norm(self.ball - self.agent))

    def _alignment_cos(self) -> float:
        gdir  = _norm(goal_center() - self.ball)
        abdir = _norm(self.ball - self.agent)
        return float(np.clip(np.dot(gdir, abdir), -1.0, 1.0))

    def _behind_amount(self) -> float:
        gdir = _norm(goal_center() - self.ball)
        return float(np.dot(self.agent - self.ball, gdir))

    def sweet_spot(self) -> np.ndarray:
        gdir = _norm(goal_center() - self.ball)
        return self.ball - gdir * C.SWEET_DIST

    def _sweet_spot_dist(self) -> float:
        return float(np.linalg.norm(self.sweet_spot() - self.agent))

    def _clamp_ball_speed(self) -> None:
        speed = np.linalg.norm(self.ball_vel)
        if speed > C.BALL_MAX_SPEED:
            self.ball_vel = self.ball_vel / (speed + 1e-9) * C.BALL_MAX_SPEED

    def _scored(self) -> bool:
        x, y = self.ball
        return x >= C.FIELD_W and (C.GOAL_Y0 <= y <= C.GOAL_Y1)

    def _ball_out(self) -> bool:
        x, y = self.ball
        if x < 0 or y < 0 or y > C.FIELD_H:
            return True
        if x > C.FIELD_W and not (C.GOAL_Y0 <= y <= C.GOAL_Y1):
            return True
        return False

    def _ball_unreachable(self) -> bool:
        speed = float(np.linalg.norm(self.ball_vel))
        if speed >= 0.3:
            return False
        x, y = self.ball
        on_left          = x <= C.FIELD_MARGIN + 6
        on_top           = y <= C.FIELD_MARGIN + 6
        on_bottom        = y >= C.FIELD_H - C.FIELD_MARGIN - 6
        on_right_nongoal = (x >= C.FIELD_W - C.FIELD_MARGIN - 6
                            and not (C.GOAL_Y0 <= y <= C.GOAL_Y1))
        return on_left or on_top or on_bottom or on_right_nongoal

    # ------------------------------------------------------------------
    # Continuous observation
    # ------------------------------------------------------------------

    def _obs(self) -> Obs:
        """Return normalised continuous observation vector (length 10)."""
        dx, dy = self.ball - self.agent
        gx, gy = goal_center() - self.ball
        vx, vy = self.ball_vel
        touching = 1.0 if self.is_touching() else 0.0
        behind   = 1.0 if self._behind_amount() <= C.BEHIND_THRESH else 0.0
        return np.array([
            dx / (C.FIELD_W / 2),
            dy / (C.FIELD_H / 2),
            gx / (C.FIELD_W / 2),
            gy / (C.FIELD_H / 2),
            vx / C.BALL_MAX_SPEED,
            vy / C.BALL_MAX_SPEED,
            self.agent[0] / C.FIELD_W,
            self.agent[1] / C.FIELD_H,
            touching,
            behind,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _apply_kick(self) -> float:
        if not self.is_touching():
            return 0.0
        align      = self._alignment_cos()
        behind_amt = self._behind_amount()
        vec = _norm(self.ball - self.agent)
        if align >= C.ALIGN_THRESH and behind_amt <= C.BEHIND_THRESH:
            self.ball_vel += vec * C.KICK_IMPULSE
            return 0.0
        else:
            self.ball_vel += vec * (0.35 * C.KICK_IMPULSE)
            mis = max(0.0, C.ALIGN_THRESH - align)
            return C.MISALIGNED_KICK_PENALTY * (0.5 + mis)

    def step(self, action: int) -> Tuple[Obs, float, bool]:
        if self.done:
            return self._obs(), 0.0, True

        if   action == 0: self.agent[1] -= C.AGENT_SPEED
        elif action == 1: self.agent[1] += C.AGENT_SPEED
        elif action == 2: self.agent[0] -= C.AGENT_SPEED
        elif action == 3: self.agent[0] += C.AGENT_SPEED

        self.agent[0] = np.clip(self.agent[0], 0, C.FIELD_W - C.AGENT_SIZE)
        self.agent[1] = np.clip(self.agent[1], 0, C.FIELD_H - C.AGENT_SIZE)

        kick_penalty = self._apply_kick() if action == 4 else 0.0

        if self.is_touching():
            push = self.ball - self.agent
            n = np.linalg.norm(push)
            if n < 1e-6: push, n = np.array([1.0, 0.0]), 1.0
            self.ball     += (push / n) * 1.5
            self.ball_vel += (push / n) * 0.6

        self.ball     += self.ball_vel
        self.ball_vel *= C.FRICTION
        self._clamp_ball_speed()

        if self.ball[1] <= 0:
            self.ball[1] = 0;          self.ball_vel[1] =  abs(self.ball_vel[1])
        if self.ball[1] >= C.FIELD_H:
            self.ball[1] = C.FIELD_H;  self.ball_vel[1] = -abs(self.ball_vel[1])
        if self.ball[0] <= 0:
            self.ball[0] = 0;          self.ball_vel[0] =  abs(self.ball_vel[0])
        if self.ball[0] >= C.FIELD_W:
            if not (C.GOAL_Y0 <= self.ball[1] <= C.GOAL_Y1):
                self.ball[0] = C.FIELD_W; self.ball_vel[0] = -abs(self.ball_vel[0])

        # Rewards (identical shaping to soccer_rl.py)
        reward = C.STEP_PENALTY
        if self.is_touching():
            reward += C.TOUCH_BONUS

        new_bg = self._ball_goal_dist()
        reward += C.BALL_GOAL_SHAPING * (self._prev_ball_goal_dist - new_bg)
        self._prev_ball_goal_dist = new_bg

        new_bx = float(self.ball[0])
        reward += C.BALL_X_PROGRESS * (new_bx - self._prev_ball_x)
        self._prev_ball_x = new_bx

        if not self.is_touching():
            new_ab = self._agent_ball_dist()
            reward += C.APPROACH_BALL * (self._prev_ab_dist - new_ab)
            self._prev_ab_dist = new_ab
        else:
            self._prev_ab_dist = self._agent_ball_dist()

        align = self._alignment_cos()
        reward += C.ALIGN_SCALE * (align - self._prev_align)
        self._prev_align = align

        behind_amt = self._behind_amount()
        if behind_amt <= C.BEHIND_THRESH:
            reward += C.BACKSIDE_BONUS
        elif behind_amt > 0:
            reward += C.FRONT_PENALTY

        sweet_d = self._sweet_spot_dist()
        reward += C.SWEET_WEIGHT * (self._prev_sweet - sweet_d)
        self._prev_sweet = sweet_d

        reward += kick_penalty

        if self._scored():
            reward += C.GOAL_REWARD
            self.done = True
        elif self._ball_out() or self._ball_unreachable():
            reward += C.OUT_REWARD
            self.done = True

        self.t += 1
        if self.t >= C.MAX_STEPS_PER_EP:
            self.done = True

        return self._obs(), reward, self.done

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular buffer storing (obs, action, reward, next_obs, done)
    transitions for experience replay.

    Random sampling breaks the temporal correlation between consecutive
    transitions, which stabilises neural network training.
    """

    def __init__(self, capacity: int = C.DQN_BUFFER_SIZE) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs: Obs, action: int, reward: float,
             next_obs: Obs, done: bool) -> None:
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32, device=DEVICE),
            torch.tensor(acts,               dtype=torch.long,    device=DEVICE),
            torch.tensor(rews,               dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,              dtype=torch.float32, device=DEVICE),)

    def __len__(self) -> int:
        return len(self.buf)

# ---------------------------------------------------------------------------
# Neural Network (Q-function approximator)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Multi-layer perceptron that maps a continuous observation vector to
    Q-values for each action.

    Architecture: Linear(state_dim -> hidden) -> ReLU -> ... -> Linear(hidden -> n_actions)
    """

    def __init__(
        self,
        state_dim:  int = C.DQN_STATE_DIM,
        hidden_dim: int = C.DQN_HIDDEN_DIM,
        n_layers:   int = C.DQN_N_LAYERS,
        n_actions:  int = C.N_ACTIONS,) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent.

    Maintains two networks:
    * **online**  - updated every ``DQN_TRAIN_FREQ`` steps via gradient descent.
    * **target**  - frozen copy of online; synced every ``DQN_TARGET_UPDATE``
                    steps.  Used for stable TD targets.

    Training uses the standard DQN loss:
        L = E[(r + γ * max_a' Q_target(s', a') − Q_online(s, a))²]
    """

    def __init__(self) -> None:
        self.online    = QNetwork().to(DEVICE)
        self.target    = QNetwork().to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=C.DQN_LR)
        self.buffer    = ReplayBuffer()
        self.loss_fn   = nn.MSELoss()

        self._learn_steps = 0   # counts gradient updates (for target sync)

    def act(self, obs: Obs, eps: float) -> int:
        """epsilon-greedy action selection over continuous observation."""
        if random.random() < eps:
            return random.randrange(C.N_ACTIONS)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.online(t).argmax(dim=1).item())

    def push(self, obs: Obs, action: int, reward: float,
             next_obs: Obs, done: bool) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def learn(self) -> float | None:
        """
        Sample a minibatch and perform one gradient update.
        Returns the scalar loss value, or None if the buffer is too small.
        """
        if len(self.buffer) < C.DQN_TRAIN_START:
            return None

        obs, acts, rews, next_obs, dones = self.buffer.sample(C.DQN_BATCH_SIZE)

        # Current Q-values for taken actions
        q_vals = self.online(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient through target network)
        with torch.no_grad():
            max_next_q = self.target(next_obs).max(dim=1).values
            targets    = rews + C.DQN_GAMMA * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % C.DQN_TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

# ---------------------------------------------------------------------------
# Rendering  (reuses the same draw helpers as soccer_rl.py)
# ---------------------------------------------------------------------------

def _draw_field_markings(screen: pg.Surface) -> None:
    mc = (*C.COL_MARKING, 55)
    surf = pg.Surface((C.FIELD_W, C.FIELD_H), pg.SRCALPHA)
    fm = C.FIELD_MARGIN
    pg.draw.rect(surf, mc, (fm, fm, C.FIELD_W - 2*fm, C.FIELD_H - 2*fm), 2)
    pg.draw.line(surf, mc, (C.FIELD_W // 2, fm), (C.FIELD_W // 2, C.FIELD_H - fm), 2)
    cx, cy = C.FIELD_W // 2, C.FIELD_H // 2
    pg.draw.circle(surf, mc, (cx, cy), 80, 2)
    pg.draw.circle(surf, mc, (cx, cy), 4)
    pb_w, pb_h = 200, 280
    pb_y = (C.FIELD_H - pb_h) // 2
    pg.draw.rect(surf, mc, (fm, pb_y, pb_w, pb_h), 2)
    pg.draw.rect(surf, mc, (C.FIELD_W - fm - pb_w, pb_y, pb_w, pb_h), 2)
    corner_r = 18
    for bx, by, start in [
        (fm, fm, 0), (C.FIELD_W-fm, fm, 90),
        (fm, C.FIELD_H-fm, 270), (C.FIELD_W-fm, C.FIELD_H-fm, 180),
    ]:
        arc_rect = pg.Rect(bx-corner_r, by-corner_r, corner_r*2, corner_r*2)
        pg.draw.arc(surf, mc, arc_rect,
                    math.radians(start), math.radians(start+90), 2)
    screen.blit(surf, (0, 0))


def _draw_alignment_guide(screen: pg.Surface, env: SoccerEnvDQN) -> None:
    align  = env._alignment_cos()
    colour = C.COL_ALIGN_OK if align >= C.ALIGN_THRESH else C.COL_ALIGN_BAD
    sx, sy = env.sweet_spot()
    start  = (int(env.agent[0]), int(env.agent[1]))
    end    = (int(sx), int(sy))
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = max(1.0, math.hypot(dx, dy))
    n_segs = max(1, int(length / 10))
    for i in range(0, n_segs, 2):
        t0 = i / n_segs; t1 = min((i+1)/n_segs, 1.0)
        p0 = (int(start[0]+dx*t0), int(start[1]+dy*t0))
        p1 = (int(start[0]+dx*t1), int(start[1]+dy*t1))
        pg.draw.line(screen, colour, p0, p1, 1)


def _draw_agent_direction(screen: pg.Surface, env: SoccerEnvDQN,
                          last_action: int) -> None:
    ax, ay = int(env.agent[0]), int(env.agent[1])
    hs = C.AGENT_SIZE // 2
    offsets = {0:(0,-1), 1:(0,1), 2:(-1,0), 3:(1,0)}
    if last_action in offsets:
        dx, dy = offsets[last_action]
        tip_x, tip_y = ax + dx*(hs+7), ay + dy*(hs+7)
        px, py = -dy*5, dx*5
        pts = [(tip_x, tip_y),
               (tip_x-dx*8+px, tip_y-dy*8+py),
               (tip_x-dx*8-px, tip_y-dy*8-py)]
        pg.draw.polygon(screen, C.COL_MARKING, pts)
    elif last_action == 4:
        pg.draw.circle(screen, C.COL_AGENT_KICK, (ax, ay), hs+10, 2)


def _draw_hud_panel_dqn(
    screen:       pg.Surface,
    font_lg:      pg.font.Font,
    font_sm:      pg.font.Font,
    episode:      int,
    step:         int,
    ep_reward:    float,
    eps:          float,
    total_steps:  int,
    buf_size:     int,
    last_loss:    float,
    training:     bool,
    speed_name:   str,
    win_window:   deque,
    rew_window:   deque,
    steps_window: deque,
    overlay_help: bool,) -> None:
    px = C.FIELD_W
    pw = C.PANEL_W
    ph = C.FIELD_H

    panel_surf = pg.Surface((pw, ph), pg.SRCALPHA)
    panel_surf.fill(C.COL_HUD_BG)
    screen.blit(panel_surf, (px, 0))
    pg.draw.line(screen, C.COL_DIVIDER, (px, 0), (px, ph), 1)

    def label(text, x, y, col=C.COL_HUD_DIM):
        s = font_sm.render(text, True, col)
        screen.blit(s, (px+x, y)); return y + s.get_height() + 2

    def value(text, x, y, col=C.COL_HUD_TEXT):
        s = font_sm.render(text, True, col)
        screen.blit(s, (px+x, y)); return y + s.get_height() + 2

    def divider(y):
        pg.draw.line(screen, C.COL_DIVIDER, (px+8, y), (px+pw-8, y), 1)
        return y + 8

    y = 10
    s = font_lg.render("RL SOCCER", True, C.COL_HUD_ACCENT)
    screen.blit(s, (px+(pw-s.get_width())//2, y)); y += s.get_height()+4
    s = font_sm.render("Deep Q-Network  ", True, C.COL_HUD_DIM)
    screen.blit(s, (px+(pw-s.get_width())//2, y)); y += s.get_height()+6
    y = divider(y)

    y = label("TRAINING", 8, y)
    y = value(f"Episode   {episode:,}", 8, y)
    y = value(f"Step      {step} / {C.MAX_STEPS_PER_EP}", 8, y)
    y = value(f"Env steps {total_steps:,}", 8, y)
    y = value(f"Epsilon   {eps:.3f}", 8, y)
    y = value(f"Buffer    {buf_size:,}", 8, y)
    loss_txt = f"{last_loss:.4f}" if last_loss > 0 else "warming up"
    y = value(f"Loss      {loss_txt}", 8, y)
    mode_col = C.COL_ALIGN_OK if training else (200, 80, 80)
    mode_txt = "ON  *  " + speed_name if training else "OFF"
    s = font_sm.render(f"Mode      {mode_txt}", True, mode_col)
    screen.blit(s, (px+8, y)); y += s.get_height()+4
    y = divider(y)

    wr   = (sum(win_window)/len(win_window)*100) if win_window else 0.0
    avgr = (sum(rew_window)/len(rew_window))      if rew_window else 0.0
    avgs = (sum(steps_window)/len(steps_window))  if steps_window else 0.0

    y = label("PERFORMANCE", 8, y)
    wr_col = C.COL_ALIGN_OK if wr >= 50 else (C.COL_HUD_ACCENT if wr >= 20 else C.COL_HUD_TEXT)
    s = font_sm.render(f"Win rate  {wr:5.1f}%", True, wr_col)
    screen.blit(s, (px+8, y)); y += s.get_height()+2
    y = value(f"Avg R     {avgr:+.2f}", 8, y)
    y = value(f"Avg steps {avgs:.0f}", 8, y)

    y += 4
    bar_w, bar_h = pw-20, 6
    fill = int(bar_w * min(max(ep_reward, -20.0), 130.0) / 130.0)
    bar_col = C.COL_ALIGN_OK if ep_reward >= 0 else (200, 80, 80)
    pg.draw.rect(screen, (40,40,40), (px+10, y, bar_w, bar_h), border_radius=3)
    if fill > 0:
        pg.draw.rect(screen, bar_col, (px+10, y, fill, bar_h), border_radius=3)
    s = font_sm.render(f"Ep reward  {ep_reward:+.1f}", True, C.COL_HUD_DIM)
    screen.blit(s, (px+8, y+bar_h+2)); y += bar_h+s.get_height()+6
    y = divider(y)

    y = label("CONTROLS", 8, y)
    for key, desc in [("T","Toggle training"),("SPACE","One episode"),
                      ("G","Cycle speed"),("R","Reset"),
                      ("S/L","Save/Load"),("H","Help"),("Q","Quit")]:
        sk = font_sm.render(key,  True, C.COL_HUD_ACCENT)
        sd = font_sm.render(desc, True, C.COL_HUD_DIM)
        screen.blit(sk, (px+8, y)); screen.blit(sd, (px+52, y))
        y += sk.get_height()+1

    if overlay_help:
        lines = [
            "DQN vs Q-Table:",
            "* No state binning - raw floats in",
            "* Neural net: 10 -> 128 -> 128 -> 128 -> 5",
            "* Replay buffer: 100k transitions",
            "* Target network syncs every 500 steps",
            "* Same reward shaping as Q-table ",
            "* Epsilon decays over env steps, not eps",
        ]
        box_h = len(lines)*18+16
        box_y = C.FIELD_H-box_h-10
        ov = pg.Surface((C.FIELD_W-20, box_h), pg.SRCALPHA)
        ov.fill((0,0,0,160))
        screen.blit(ov, (10, box_y))
        for i, line in enumerate(lines):
            col = C.COL_HUD_ACCENT if i == 0 else C.COL_HUD_TEXT
            screen.blit(font_sm.render(line, True, col), (18, box_y+8+i*18))


def draw_scene_dqn(
    screen, font_lg, font_sm, env: SoccerEnvDQN,
    last_action, episode, step, ep_reward, eps,
    total_steps, buf_size, last_loss,
    training, speed_name,
    win_window, rew_window, steps_window, overlay_help,) -> None:
    screen.fill(C.COL_GRASS, pg.Rect(0, 0, C.FIELD_W, C.FIELD_H))
    screen.fill((15,15,15), pg.Rect(C.FIELD_W, 0, C.PANEL_W, C.FIELD_H))

    _draw_field_markings(screen)

    pg.draw.rect(screen, C.COL_GOAL,
                 (C.GOAL_X, C.GOAL_Y0, C.GOAL_THICKNESS, C.GOAL_HEIGHT))
    pg.draw.line(screen, (255,255,100),
                 (C.GOAL_X, C.GOAL_Y0), (C.GOAL_X+C.GOAL_THICKNESS, C.GOAL_Y0), 2)
    pg.draw.line(screen, (255,255,100),
                 (C.GOAL_X, C.GOAL_Y1), (C.GOAL_X+C.GOAL_THICKNESS, C.GOAL_Y1), 2)

    sx, sy = env.sweet_spot()
    pg.draw.circle(screen, C.COL_SWEET, (int(sx), int(sy)), 6)
    pg.draw.circle(screen, (180,180,80), (int(sx), int(sy)), 6, 1)

    _draw_alignment_guide(screen, env)

    b_rect = env._ball_rect()
    shadow = b_rect.move(2, 2)
    sh = pg.Surface((shadow.w, shadow.h), pg.SRCALPHA); sh.fill((0,0,0,60))
    screen.blit(sh, shadow.topleft)
    pg.draw.rect(screen, C.COL_BALL, b_rect, border_radius=3)

    agent_col = C.COL_AGENT_KICK if last_action == 4 else C.COL_AGENT
    a_rect = env._agent_rect()
    pg.draw.rect(screen, agent_col, a_rect, border_radius=4)
    pg.draw.rect(screen, (255,255,255,80), a_rect, 1, border_radius=4)

    _draw_agent_direction(screen, env, last_action)

    _draw_hud_panel_dqn(
        screen, font_lg, font_sm,
        episode, step, ep_reward, eps,
        total_steps, buf_size, last_loss,
        training, speed_name,
        win_window, rew_window, steps_window, overlay_help,)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    """Initialise and run the DQN training loop."""
    pg.init()
    screen = pg.display.set_mode((C.W, C.H))
    pg.display.set_caption("Pygame RL Soccer  *  DQN ")
    font_lg = pg.font.SysFont("consolas", 20, bold=True)
    font_sm = pg.font.SysFont("consolas", 14)

    env   = SoccerEnvDQN()
    agent = DQNAgent()

    win_window   = deque(maxlen=200)
    rew_window   = deque(maxlen=100)
    steps_window = deque(maxlen=100)

    C.SAVES_DIR.mkdir(parents=True, exist_ok=True)
    episode, total_steps = load_dqn(agent)
    print(f"[DQN] Starting from episode={episode:,}, env_steps={total_steps:,}")

    speed_idx = C.SPEED_INDEX_DEFAULT
    speed_name, fps_cap, steps_per_tick = C.SPEEDS[speed_idx]
    clock = pg.time.Clock()

    training     = False
    overlay_help = False
    last_action  = -1
    last_loss    = 0.0

    obs       = env.reset()
    ep_reward = 0.0
    step      = 0

    def end_episode(scored: bool) -> None:
        nonlocal episode
        win_window.append(1 if scored else 0)
        rew_window.append(ep_reward)
        if scored: steps_window.append(step)
        episode += 1
        if episode % C.DQN_AUTOSAVE_EVERY == 0:
            save_dqn(agent, episode, total_steps)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif event.key == pg.K_t:
                    training = not training
                elif event.key == pg.K_g:
                    speed_idx = (speed_idx+1) % len(C.SPEEDS)
                    speed_name, fps_cap, steps_per_tick = C.SPEEDS[speed_idx]
                elif event.key == pg.K_r:
                    obs = env.reset(); ep_reward = 0.0; step = 0; last_action = -1
                elif event.key == pg.K_s:
                    save_dqn(agent, episode, total_steps)
                elif event.key == pg.K_l:
                    load_dqn(agent)
                elif event.key == pg.K_h:
                    overlay_help = not overlay_help
                elif event.key == pg.K_SPACE:
                    obs = env.reset(); ep_reward = 0.0; step = 0; last_action = -1
                    done = False
                    eps_one = epsilon_schedule_dqn(total_steps)
                    while not done:
                        last_action = agent.act(obs, eps_one)
                        next_obs, r, done = env.step(last_action)
                        agent.push(obs, last_action, r, next_obs, done)
                        if total_steps % C.DQN_TRAIN_FREQ == 0:
                            l = agent.learn()
                            if l is not None: last_loss = l
                        obs = next_obs; ep_reward += r; step += 1
                        total_steps += 1
                        eps_one = epsilon_schedule_dqn(total_steps)
                        draw_scene_dqn(
                            screen, font_lg, font_sm, env, last_action,
                            episode, step, ep_reward, eps_one,
                            total_steps, len(agent.buffer), last_loss,
                            True, speed_name,
                            win_window, rew_window, steps_window, overlay_help,)
                        pg.display.flip(); clock.tick(fps_cap)
                        for ev in pg.event.get():
                            if ev.type == pg.QUIT: done = True; running = False
                    end_episode(env._scored())

        if training:
            for _ in range(steps_per_tick):
                if env.done:
                    end_episode(env._scored())
                    obs = env.reset(); ep_reward = 0.0; step = 0; last_action = -1

                eps = epsilon_schedule_dqn(total_steps)
                last_action = agent.act(obs, eps)
                next_obs, r, done = env.step(last_action)
                agent.push(obs, last_action, r, next_obs, done)
                if total_steps % C.DQN_TRAIN_FREQ == 0:
                    l = agent.learn()
                    if l is not None: last_loss = l
                obs = next_obs; ep_reward += r; step += 1
                total_steps += 1

        eps = epsilon_schedule_dqn(total_steps)
        draw_scene_dqn(
            screen, font_lg, font_sm, env, last_action,
            episode, step, ep_reward, eps,
            total_steps, len(agent.buffer), last_loss,
            training, speed_name,
            win_window, rew_window, steps_window, overlay_help,)
        pg.display.flip()
        clock.tick(fps_cap)

    save_dqn(agent, episode, total_steps)
    pg.quit()
    print("[DQN] Exited cleanly.")


if __name__ == "__main__":
    run()
