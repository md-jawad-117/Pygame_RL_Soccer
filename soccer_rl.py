"""
soccer_rl.py - Pygame RL Soccer  (Q-learning,)
====================================================
A single agent (red square) learns to kick a ball into the right-hand goal
using tabular Q-learning with epsilon-greedy exploration.

Key features
------------
* Auto-save / auto-load:  Q-table and training state are persisted to saves/
  every ``AUTOSAVE_EVERY`` episodes and on clean exit.
* Rolling HUD metrics: WinRate (200 ep), AvgReward (100 ep),
  AvgSteps-on-wins (100 ep).
* Three speed modes (press G to cycle): SLOW (60 FPS), FAST (~1 200 FPS),
  HYPER (~5 000 FPS with multiple env steps per render tick).
* Goal-aware reward shaping guides the agent to position itself *behind*
  the ball and align with the ball->goal axis before kicking.

Controls
--------
  T       - toggle continuous training
  SPACE   - run exactly one training episode (fully rendered)
  G       - cycle speed mode: SLOW -> FAST -> HYPER
  R       - reset environment (start a new episode)
  S       - save Q-table and training state immediately
  L       - load Q-table from disk
  H       - toggle help / reward-shaping overlay
  Q / ESC - quit (auto-saves first)
"""

import random
import pickle
from collections import defaultdict, deque
from typing import Tuple, Dict, Any

import numpy as np
import pygame as pg

import config as C

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    """Return *v* normalised to unit length (safe against zero vectors)."""
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _make_bins(lo: float, hi: float, n: int) -> np.ndarray:
    """Return *n + 1* evenly-spaced bin edges covering [lo, hi]."""
    return np.linspace(lo, hi, n + 1)


def _digitize(val: float, edges: np.ndarray) -> int:
    """Map *val* to a bin index in [0, len(edges) - 2]."""
    idx = int(np.digitize([val], edges, right=False)[0]) - 1
    return int(np.clip(idx, 0, len(edges) - 2))


def goal_center() -> np.ndarray:
    """Return the (x, y) centre of the goal mouth."""
    return np.array(
        [C.FIELD_W - C.GOAL_THICKNESS // 2, (C.GOAL_Y0 + C.GOAL_Y1) / 2.0],
        dtype=float,)


def epsilon_schedule(episode: int) -> float:
    """Linear epsilon-decay from EPS_START to EPS_END over EPS_DECAY_EPISODES."""
    if episode >= C.EPS_DECAY_EPISODES:
        return C.EPS_END
    progress = episode / C.EPS_DECAY_EPISODES
    return C.EPS_START - (C.EPS_START - C.EPS_END) * progress


# Pre-compute bin edges once at import time.
_DX_EDGES = _make_bins(C.DX_RANGE[0], C.DX_RANGE[1], C.BINS)
_DY_EDGES = _make_bins(C.DY_RANGE[0], C.DY_RANGE[1], C.BINS)
_GX_EDGES = _make_bins(C.GX_RANGE[0], C.GX_RANGE[1], C.BINS)
_GY_EDGES = _make_bins(C.GY_RANGE[0], C.GY_RANGE[1], C.BINS)
# : ball velocity bin edges
_VX_EDGES = _make_bins(C.VX_RANGE[0], C.VX_RANGE[1], C.VEL_BINS)
_VY_EDGES = _make_bins(C.VY_RANGE[0], C.VY_RANGE[1], C.VEL_BINS)
# : close-range fine bin edges for dx/dy when agent is near the ball
_CDX_EDGES = _make_bins(-C.CLOSE_RANGE, C.CLOSE_RANGE, C.CLOSE_BINS)
_CDY_EDGES = _make_bins(-C.CLOSE_RANGE, C.CLOSE_RANGE, C.CLOSE_BINS)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_q_table(Q: dict, path=C.Q_PATH) -> None:
    """Serialise the Q-table dictionary to *path* using pickle."""
    try:
        C.SAVES_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(dict(Q), fh)
        print(f"[INFO] Q-table saved -> {path}  ({len(Q)} states)")
    except Exception as exc:
        print(f"[WARN] Could not save Q-table: {exc}")


def load_q_table(path=C.Q_PATH) -> dict:
    """Load and return the Q-table dictionary from *path*, or {} if missing."""
    if path.exists():
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            print(f"[INFO] Q-table loaded <- {path}  ({len(obj)} states)")
            return obj
        except Exception as exc:
            print(f"[WARN] Could not load Q-table: {exc}")
    else:
        print(f"[INFO] No Q-table found at {path} - starting fresh.")
    return {}


def save_training_state(episode: int, eps: float, path=C.STATE_PATH) -> None:
    """Persist the episode counter and current epsilon to *path*."""
    try:
        C.SAVES_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"episode": episode, "eps": eps}, fh)
        print(f"[INFO] Training state saved -> {path}")
    except Exception as exc:
        print(f"[WARN] Could not save training state: {exc}")


def load_training_state(path=C.STATE_PATH) -> Tuple[int, float | None]:
    """Return (episode, eps) from *path*, or (0, None) if unavailable."""
    if path.exists():
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            print(f"[INFO] Training state loaded <- {path}: {obj}")
            return obj.get("episode", 0), obj.get("eps", None)
        except Exception as exc:
            print(f"[WARN] Could not load training state: {exc}")
    return 0, None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

State = Tuple[int, int, int, int, int, int, int, int]
# (sx, sy, sgx, sgy, svx, svy, near, touching)
#  sx,  sy  : agent->ball relative position - coarse (16 bins) when far,
#             fine (8 bins over ±CLOSE_RANGE) when near the ball   <- 
#  sgx, sgy : ball->goal relative position    (16 × 16 bins)
#  svx, svy : ball velocity components       ( 4 ×  4 bins)        <- 
#  near     : 1 if agent is within CLOSE_RANGE px of ball          <- 
#             (flags which bin scale sx/sy used - keeps states disjoint)
#  touching : agent touching ball?           (0 or 1)


class SoccerEnv:
    """
    2-D soccer environment for a single RL agent.

    The agent (red square) must navigate to the ball and kick it into the
    goal on the right side of the field.  Episodes end when the ball scores,
    goes out of bounds, or the step limit is reached.

    Coordinate system
    -----------------
    Origin is the top-left corner of the window.  Positive x -> right,
    positive y -> down (standard Pygame convention).
    """

    def __init__(self) -> None:
        self.agent:    np.ndarray = np.zeros(2, dtype=float)
        self.ball:     np.ndarray = np.zeros(2, dtype=float)
        self.ball_vel: np.ndarray = np.zeros(2, dtype=float)
        self.t:   int  = 0
        self.done: bool = False

        # Cached values used for potential-shaping deltas.
        self._prev_ball_goal_dist: float = 0.0
        self._prev_ab_dist:        float = 0.0
        self._prev_ball_x:         float = 0.0
        self._prev_align:          float = 0.0
        self._prev_sweet:          float = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Reset / episode bookkeeping
    # ------------------------------------------------------------------

    def reset(self) -> State:
        """
        Reset the environment to a new episode start.

        Uses a mixed curriculum:
        * 60 % of episodes start with the ball close to the agent.
        * 40 % start with the ball in a random mid-left position.

        Returns
        -------
        State
            The initial discrete state tuple.
        """
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

        # Initialise shaping caches.
        self._prev_ball_goal_dist = self._ball_goal_dist()
        self._prev_ab_dist        = self._agent_ball_dist()
        self._prev_ball_x         = float(self.ball[0])
        self._prev_align          = self._alignment_cos()
        self._prev_sweet          = self._sweet_spot_dist()

        return self.get_state()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _agent_rect(self) -> pg.Rect:
        hs = C.AGENT_SIZE // 2
        return pg.Rect(int(self.agent[0]) - hs, int(self.agent[1]) - hs, C.AGENT_SIZE, C.AGENT_SIZE)

    def _ball_rect(self) -> pg.Rect:
        hs = C.BALL_SIZE // 2
        return pg.Rect(int(self.ball[0]) - hs, int(self.ball[1]) - hs, C.BALL_SIZE, C.BALL_SIZE)

    def is_touching(self) -> bool:
        """Return True if the agent and ball rectangles overlap."""
        return self._agent_rect().colliderect(self._ball_rect())

    def _ball_goal_dist(self) -> float:
        return float(np.linalg.norm(self.ball - goal_center()))

    def _agent_ball_dist(self) -> float:
        return float(np.linalg.norm(self.ball - self.agent))

    def _geom_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the unit vectors (ball->goal direction, agent->ball direction).
        """
        gdir  = _norm(goal_center() - self.ball)   # ball -> goal
        abdir = _norm(self.ball - self.agent)       # agent -> ball
        return gdir, abdir

    def _alignment_cos(self) -> float:
        """
        Cosine of the angle between agent->ball and ball->goal directions.

        A value near +1 means the agent is directly behind the ball relative
        to the goal - the ideal position for a well-directed kick.
        """
        gdir, abdir = self._geom_vectors()
        return float(np.clip(np.dot(gdir, abdir), -1.0, 1.0))

    def _behind_amount(self) -> float:
        """
        Signed projection of (agent - ball) onto the ball->goal unit vector.

        Negative values mean the agent is *behind* the ball (good for kicking).
        Positive values mean the agent is between the ball and the goal (bad).
        """
        gdir = _norm(goal_center() - self.ball)
        return float(np.dot(self.agent - self.ball, gdir))

    def sweet_spot(self) -> np.ndarray:
        """
        The ideal agent position: directly behind the ball relative to the goal,
        at a distance of ``SWEET_DIST`` pixels.
        """
        gdir = _norm(goal_center() - self.ball)
        return self.ball - gdir * C.SWEET_DIST

    def _sweet_spot_dist(self) -> float:
        return float(np.linalg.norm(self.sweet_spot() - self.agent))

    def _clamp_ball_speed(self) -> None:
        speed = np.linalg.norm(self.ball_vel)
        if speed > C.BALL_MAX_SPEED:
            self.ball_vel = self.ball_vel / (speed + 1e-9) * C.BALL_MAX_SPEED

    # ------------------------------------------------------------------
    # Kick logic
    # ------------------------------------------------------------------

    def _apply_kick(self) -> float:
        """
        Execute a KICK action.  Returns any immediate misalignment penalty.

        A *good* kick requires the agent to be:
        1. Behind the ball along the ball->goal axis.
        2. Sufficiently aligned (alignment cosine ≥ ALIGN_THRESH).

        Good kicks receive full KICK_IMPULSE; poor kicks receive a weaker
        nudge plus a penalty proportional to the degree of misalignment.
        """
        if not self.is_touching():
            return 0.0

        align      = self._alignment_cos()
        behind_amt = self._behind_amount()

        vec = _norm(self.ball - self.agent)

        if align >= C.ALIGN_THRESH and behind_amt <= C.BEHIND_THRESH:
            # Well-aligned kick - full power.
            self.ball_vel += vec * C.KICK_IMPULSE
            return 0.0
        else:
            # Weak nudge; penalise proportionally to misalignment.
            self.ball_vel += vec * (0.35 * C.KICK_IMPULSE)
            mis = max(0.0, C.ALIGN_THRESH - align)
            return C.MISALIGNED_KICK_PENALTY * (0.5 + mis)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Apply *action* and advance the simulation by one step.

        Parameters
        ----------
        action : int
            Index into ``ACTIONS`` (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=KICK).

        Returns
        -------
        state : State
            New discrete state.
        reward : float
            Scalar reward for this transition.
        done : bool
            True if the episode has ended.
        info : dict
            Empty auxiliary dict (reserved for future diagnostics).
        """
        if self.done:
            return self.get_state(), 0.0, True, {}

        # --- Agent movement ---
        if   action == 0: self.agent[1] -= C.AGENT_SPEED   # UP
        elif action == 1: self.agent[1] += C.AGENT_SPEED   # DOWN
        elif action == 2: self.agent[0] -= C.AGENT_SPEED   # LEFT
        elif action == 3: self.agent[0] += C.AGENT_SPEED   # RIGHT

        # Clamp agent to full field edges (not the marking line) so it can
        # chase balls that roll near the boundary before going out of bounds.
        self.agent[0] = np.clip(self.agent[0], 0, C.FIELD_W - C.AGENT_SIZE)
        self.agent[1] = np.clip(self.agent[1], 0, C.FIELD_H - C.AGENT_SIZE)

        # Kick (evaluated after movement so position is up-to-date).
        kick_penalty = self._apply_kick() if action == 4 else 0.0

        # Soft push: prevent agent and ball from overlapping.
        if self.is_touching():
            push = self.ball - self.agent
            n = np.linalg.norm(push)
            if n < 1e-6:
                push, n = np.array([1.0, 0.0]), 1.0
            self.ball     += (push / n) * 1.5
            self.ball_vel += (push / n) * 0.6

        # --- Ball physics ---
        self.ball     += self.ball_vel
        self.ball_vel *= C.FRICTION
        self._clamp_ball_speed()

        # Wall collisions (except open right side for goal).
        if self.ball[1] <= 0:
            self.ball[1] = 0;          self.ball_vel[1] =  abs(self.ball_vel[1])
        if self.ball[1] >= C.FIELD_H:
            self.ball[1] = C.FIELD_H;  self.ball_vel[1] = -abs(self.ball_vel[1])
        if self.ball[0] <= 0:
            self.ball[0] = 0;          self.ball_vel[0] =  abs(self.ball_vel[0])
        if self.ball[0] >= C.FIELD_W:
            if not (C.GOAL_Y0 <= self.ball[1] <= C.GOAL_Y1):
                self.ball[0] = C.FIELD_W; self.ball_vel[0] = -abs(self.ball_vel[0])

        # --- Reward computation ---
        reward = C.STEP_PENALTY

        # Touch bonus.
        if self.is_touching():
            reward += C.TOUCH_BONUS

        # Potential-based shaping: ball approaching goal.
        new_bg = self._ball_goal_dist()
        reward += C.BALL_GOAL_SHAPING * (self._prev_ball_goal_dist - new_bg)
        self._prev_ball_goal_dist = new_bg

        # Shaping: ball moving rightward.
        new_bx = float(self.ball[0])
        reward += C.BALL_X_PROGRESS * (new_bx - self._prev_ball_x)
        self._prev_ball_x = new_bx

        # Shaping: agent approaching ball (only when not touching).
        if not self.is_touching():
            new_ab = self._agent_ball_dist()
            reward += C.APPROACH_BALL * (self._prev_ab_dist - new_ab)
            self._prev_ab_dist = new_ab
        else:
            self._prev_ab_dist = self._agent_ball_dist()

        # Goal-aware shaping: alignment improvement.
        align = self._alignment_cos()
        reward += C.ALIGN_SCALE * (align - self._prev_align)
        self._prev_align = align

        # Goal-aware shaping: behind / front of ball.
        behind_amt = self._behind_amount()
        if behind_amt <= C.BEHIND_THRESH:
            reward += C.BACKSIDE_BONUS
        elif behind_amt > 0:
            reward += C.FRONT_PENALTY

        # Goal-aware shaping: sweet-spot proximity.
        sweet_d = self._sweet_spot_dist()
        reward += C.SWEET_WEIGHT * (self._prev_sweet - sweet_d)
        self._prev_sweet = sweet_d

        # Misaligned kick penalty (computed inside _apply_kick).
        reward += kick_penalty

        # --- Terminal conditions ---
        if self._scored():
            reward += C.GOAL_REWARD
            self.done = True
        elif self._ball_out():
            reward += C.OUT_REWARD
            self.done = True
        elif self._ball_unreachable():
            # Ball is stuck against a wall and the agent cannot reach it.
            # End early with the out-of-bounds penalty rather than wasting
            # up to 1500 steps idling.
            reward += C.OUT_REWARD
            self.done = True

        self.t += 1
        if self.t >= C.MAX_STEPS_PER_EP:
            self.done = True

        return self.get_state(), reward, self.done, {}

    # ------------------------------------------------------------------
    # Terminal checks
    # ------------------------------------------------------------------

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
        """
        Return True when the ball has rolled so close to a non-goal boundary
        that it is effectively stopped there and the agent cannot dislodge it.

        This prevents the agent from idling for up to 1500 steps whenever the
        ball gets wedged in a corner or against a side wall far from the goal.
        The threshold is 6 px inside each wall edge - tight enough to catch
        genuinely stuck situations without triggering during normal play.
        """
        x, y = self.ball
        vx, vy = self.ball_vel
        speed = float(np.linalg.norm(self.ball_vel))

        # Ball is essentially stopped (friction has killed its velocity).
        ball_stopped = speed < 0.3

        # Ball is pressed against a non-goal boundary.
        on_left   = x <= C.FIELD_MARGIN + 6
        on_top    = y <= C.FIELD_MARGIN + 6
        on_bottom = y >= C.FIELD_H - C.FIELD_MARGIN - 6
        # Right wall only counts as stuck if it's outside the goal mouth.
        on_right_nongaol = (
            x >= C.FIELD_W - C.FIELD_MARGIN - 6
            and not (C.GOAL_Y0 <= y <= C.GOAL_Y1))

        stuck = on_left or on_top or on_bottom or on_right_nongaol
        return ball_stopped and stuck

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------

    def get_state(self) -> State:
        """
        Return the current discrete state as an 8-tuple  ().

        Components
        ----------
        sx, sy   : relative ball position (agent -> ball).
                   When the agent is within CLOSE_RANGE px of the ball
                   ('near' == 1), finer bins covering ±CLOSE_RANGE are used
                   so nearby positions are cleanly separated - eliminating
                   the oscillation / confusion that coarse bins cause up close.
                   When far ('near' == 0), the full-range coarse bins are used.
        sgx, sgy : relative goal position (ball -> goal centre), [0, BINS-1]
        svx, svy : discretised ball velocity components, [0, VEL_BINS-1]
        near     : 1 if agent is within CLOSE_RANGE px of ball, else 0
                   (included in the tuple so far-states and close-states are
                   always disjoint - same sx value means different things in
                   each regime, so 'near' keeps them separate in the Q-table)
        c        : 1 if agent is currently touching ball, else 0
        """
        dx, dy = self.ball - self.agent
        gx, gy = goal_center() - self.ball
        vx, vy = self.ball_vel

        ab_dist = float(np.hypot(dx, dy))
        near = 1 if ab_dist <= C.CLOSE_RANGE else 0
        c    = 1 if self.is_touching() else 0

        if near:
            # Fine bins: ±CLOSE_RANGE px mapped to CLOSE_BINS buckets.
            sx = _digitize(float(np.clip(dx, -C.CLOSE_RANGE, C.CLOSE_RANGE)), _CDX_EDGES)
            sy = _digitize(float(np.clip(dy, -C.CLOSE_RANGE, C.CLOSE_RANGE)), _CDY_EDGES)
        else:
            # Coarse bins: full field range.
            sx = _digitize(float(np.clip(dx, C.DX_RANGE[0], C.DX_RANGE[1])), _DX_EDGES)
            sy = _digitize(float(np.clip(dy, C.DY_RANGE[0], C.DY_RANGE[1])), _DY_EDGES)

        sgx = _digitize(float(np.clip(gx, C.GX_RANGE[0], C.GX_RANGE[1])), _GX_EDGES)
        sgy = _digitize(float(np.clip(gy, C.GY_RANGE[0], C.GY_RANGE[1])), _GY_EDGES)
        svx = _digitize(float(np.clip(vx, C.VX_RANGE[0], C.VX_RANGE[1])), _VX_EDGES)
        svy = _digitize(float(np.clip(vy, C.VY_RANGE[0], C.VY_RANGE[1])), _VY_EDGES)

        return sx, sy, sgx, sgy, svx, svy, near, c


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearner:
    """
    Tabular Q-learning agent with epsilon-greedy action selection.

    The Q-table is a ``defaultdict`` mapping discrete states to a numpy array
    of Q-values (one per action).  Unknown states are initialised to 0.
    """

    def __init__(self) -> None:
        self.Q: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(C.N_ACTIONS, dtype=np.float32))

    def act(self, state: State, eps: float, prev_action: int = -1) -> int:
        """
        Return an action using epsilon-greedy policy with action inertia  ().

        Parameters
        ----------
        state : State
            Current discrete environment state.
        eps : float
            Exploration probability (0 = fully greedy, 1 = fully random).
        prev_action : int
            The action chosen on the previous step (-1 = no prior action).
            When the top two Q-values are within ACTION_INERTIA of each other,
            the previous action is preferred over switching - this eliminates
            the oscillation / direction-flip behaviour seen when the agent is
            close to the ball and Q-values are nearly tied.
        """
        if random.random() < eps:
            return random.randrange(C.N_ACTIONS)

        q = self.Q[state]
        best = int(np.argmax(q))

        # Inertia: if previous action's Q-value is within the margin of the
        # best action, keep doing what we were doing instead of switching.
        if (
            prev_action >= 0
            and prev_action != best
            and q[prev_action] >= q[best] - C.ACTION_INERTIA):
            return prev_action

        return best

    def update(self, s: State, a: int, r: float, sp: State, done: bool) -> None:
        """
        Perform a single Q-learning (TD) update.

        Q(s, a) <- Q(s, a) + α * [r + γ * max_a' Q(s', a') − Q(s, a)]
        """
        qsa    = self.Q[s][a]
        target = r if done else r + C.GAMMA * float(np.max(self.Q[sp]))
        self.Q[s][a] = (1 - C.ALPHA) * qsa + C.ALPHA * target

    def save(self, path=C.Q_PATH) -> None:
        """Save Q-table to *path*."""
        save_q_table(self.Q, path)

    def load(self, path=C.Q_PATH) -> None:
        """Load Q-table from *path* (no-op if file is missing)."""
        obj = load_q_table(path)
        self.Q = defaultdict(lambda: np.zeros(C.N_ACTIONS, dtype=np.float32), obj)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_field_markings(screen: pg.Surface) -> None:
    """
    Draw static soccer-pitch markings onto *screen*.

    Markings are drawn in a dimmed white so they are visible but do not
    distract from the agents.  All coordinates are in field space (left
    FIELD_W pixels of the window).
    """
    mc = (*C.COL_MARKING, 55)     # (R, G, B, A) - semi-transparent
    surf = pg.Surface((C.FIELD_W, C.FIELD_H), pg.SRCALPHA)

    fm = C.FIELD_MARGIN

    # Outer boundary
    pg.draw.rect(surf, mc, (fm, fm, C.FIELD_W - 2*fm, C.FIELD_H - 2*fm), 2)

    # Centre vertical line
    pg.draw.line(surf, mc, (C.FIELD_W // 2, fm), (C.FIELD_W // 2, C.FIELD_H - fm), 2)

    # Centre circle
    cx, cy = C.FIELD_W // 2, C.FIELD_H // 2
    pg.draw.circle(surf, mc, (cx, cy), 80, 2)
    pg.draw.circle(surf, mc, (cx, cy), 4)   # centre spot

    # Left penalty box  (~200 px wide, centred vertically)
    pb_w, pb_h = 200, 280
    pb_x = fm
    pb_y = (C.FIELD_H - pb_h) // 2
    pg.draw.rect(surf, mc, (pb_x, pb_y, pb_w, pb_h), 2)

    # Right penalty box (goal side)
    pg.draw.rect(surf, mc, (C.FIELD_W - fm - pb_w, pb_y, pb_w, pb_h), 2)

    # Corner arcs
    corner_r = 18
    corners = [
        (fm, fm,                         90, 0),    # top-left
        (C.FIELD_W - fm, fm,             90, 180),  # top-right (but capped at field edge)
        (fm, C.FIELD_H - fm,             90, 270),  # bottom-left
        (C.FIELD_W - fm, C.FIELD_H - fm, 90, 90),  # bottom-right
    ]
    for (bx, by, _arc, start_angle) in corners:
        arc_rect = pg.Rect(bx - corner_r, by - corner_r, corner_r * 2, corner_r * 2)
        pg.draw.arc(surf, mc, arc_rect, np.radians(start_angle), np.radians(start_angle + 90), 2)

    screen.blit(surf, (0, 0))


def _draw_agent_direction(
    screen: pg.Surface,
    env: SoccerEnv,
    last_action: int,) -> None:
    """
    Draw a directional indicator on the agent showing its last action.

    * Movement actions (UP/DOWN/LEFT/RIGHT): small filled triangle pointing
      in the direction of motion.
    * KICK action: the agent is drawn in a brighter colour (handled in the
      main draw call) and a small burst ring is drawn around it.
    """
    ax, ay = int(env.agent[0]), int(env.agent[1])
    hs = C.AGENT_SIZE // 2

    ACTION_OFFSETS = {
        0: (0, -1),   # UP
        1: (0,  1),   # DOWN
        2: (-1, 0),   # LEFT
        3: (1,  0),   # RIGHT
    }

    if last_action in ACTION_OFFSETS:
        dx, dy = ACTION_OFFSETS[last_action]
        tip_x = ax + dx * (hs + 7)
        tip_y = ay + dy * (hs + 7)
        # Build a perpendicular offset for the triangle base.
        perp_x, perp_y = -dy * 5, dx * 5
        pts = [
            (tip_x, tip_y),
            (tip_x - dx*8 + perp_x, tip_y - dy*8 + perp_y),
            (tip_x - dx*8 - perp_x, tip_y - dy*8 - perp_y),
        ]
        pg.draw.polygon(screen, C.COL_MARKING, pts)

    elif last_action == 4:  # KICK
        # Burst ring (shrinks each frame but we just draw one ring per step).
        pg.draw.circle(screen, C.COL_AGENT_KICK, (ax, ay), hs + 10, 2)


def _draw_alignment_guide(screen: pg.Surface, env: SoccerEnv) -> None:
    """
    Draw a coloured guide line from the agent to the sweet-spot position.

    The colour indicates whether the agent is well-aligned:
    * Green  - aligned (cosine ≥ ALIGN_THRESH) and ready to kick.
    * Orange - not yet aligned.
    """
    align = env._alignment_cos()
    colour = C.COL_ALIGN_OK if align >= C.ALIGN_THRESH else C.COL_ALIGN_BAD

    sx, sy = env.sweet_spot()
    start = (int(env.agent[0]), int(env.agent[1]))
    end   = (int(sx), int(sy))

    # Dashed line: draw short segments.
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, np.hypot(dx, dy))
    step_px = 10
    n_segs  = int(length / step_px)
    for i in range(0, n_segs, 2):
        t0 = i / n_segs
        t1 = min((i + 1) / n_segs, 1.0)
        p0 = (int(start[0] + dx * t0), int(start[1] + dy * t0))
        p1 = (int(start[0] + dx * t1), int(start[1] + dy * t1))
        pg.draw.line(screen, colour, p0, p1, 1)


def _draw_hud_panel(
    screen:       pg.Surface,
    font_lg:      pg.font.Font,
    font_sm:      pg.font.Font,
    episode:      int,
    step:         int,
    ep_reward:    float,
    eps:          float,
    training:     bool,
    speed_name:   str,
    win_window:   deque,
    rew_window:   deque,
    steps_window: deque,
    q_states:     int,
    overlay_help: bool,) -> None:
    """
    Render a semi-transparent stats panel on the right side of the window.

    The panel is split into three sections separated by divider lines:
    1. Training state (episode, step, epsilon, mode, speed).
    2. Performance metrics (win rate, avg reward, avg steps to goal).
    3. Keyboard shortcuts reference.
    """
    px = C.FIELD_W   # panel starts here
    pw = C.PANEL_W
    ph = C.FIELD_H

    # Background
    panel_surf = pg.Surface((pw, ph), pg.SRCALPHA)
    panel_surf.fill(C.COL_HUD_BG)
    screen.blit(panel_surf, (px, 0))

    # Thin separator between field and panel.
    pg.draw.line(screen, C.COL_DIVIDER, (px, 0), (px, ph), 1)

    def label(text: str, x: int, y: int, font=font_sm, col=C.COL_HUD_DIM) -> int:
        surf = font.render(text, True, col)
        screen.blit(surf, (px + x, y))
        return y + surf.get_height() + 2

    def value(text: str, x: int, y: int, font=font_sm, col=C.COL_HUD_TEXT) -> int:
        surf = font.render(text, True, col)
        screen.blit(surf, (px + x, y))
        return y + surf.get_height() + 2

    def divider(y: int) -> int:
        pg.draw.line(screen, C.COL_DIVIDER, (px + 8, y), (px + pw - 8, y), 1)
        return y + 8

    y = 10

    # ── Title ──────────────────────────────────────────────────────────
    surf = font_lg.render("RL SOCCER", True, C.COL_HUD_ACCENT)
    screen.blit(surf, (px + (pw - surf.get_width()) // 2, y))
    y += surf.get_height() + 4
    surf = font_sm.render("Q-Learning  ", True, C.COL_HUD_DIM)
    screen.blit(surf, (px + (pw - surf.get_width()) // 2, y))
    y += surf.get_height() + 6
    y = divider(y)

    # ── Training state ─────────────────────────────────────────────────
    y = label("TRAINING", 8, y, font_sm, C.COL_HUD_DIM)
    y = value(f"Episode   {episode:,}", 8, y)
    y = value(f"Step      {step} / {C.MAX_STEPS_PER_EP}", 8, y)
    y = value(f"Epsilon   {eps:.3f}", 8, y)
    y = value(f"Q-states  {q_states:,}", 8, y)

    mode_col = C.COL_ALIGN_OK if training else (200, 80, 80)
    mode_txt = "ON  *  " + speed_name if training else "OFF"
    surf = font_sm.render(f"Mode      {mode_txt}", True, mode_col)
    screen.blit(surf, (px + 8, y)); y += surf.get_height() + 4
    y = divider(y)

    # ── Performance ────────────────────────────────────────────────────
    wr   = (sum(win_window) / len(win_window) * 100) if win_window else 0.0
    avgr = (sum(rew_window) / len(rew_window))        if rew_window else 0.0
    avgs = (sum(steps_window) / len(steps_window))    if steps_window else 0.0

    y = label("PERFORMANCE", 8, y, font_sm, C.COL_HUD_DIM)
    wr_col = C.COL_ALIGN_OK if wr >= 50 else (C.COL_HUD_ACCENT if wr >= 20 else C.COL_HUD_TEXT)
    surf = font_sm.render(f"Win rate  {wr:5.1f}%", True, wr_col)
    screen.blit(surf, (px + 8, y)); y += surf.get_height() + 2
    y = value(f"Avg R     {avgr:+.2f}", 8, y)
    y = value(f"Avg steps {avgs:.0f}", 8, y)

    # Current episode reward mini-bar.
    y += 4
    bar_w = pw - 20
    bar_h = 6
    max_r = 130.0
    fill = int(bar_w * min(max(ep_reward, -20.0), max_r) / max_r)
    bar_col = C.COL_ALIGN_OK if ep_reward >= 0 else (200, 80, 80)
    pg.draw.rect(screen, (40, 40, 40), (px + 10, y, bar_w, bar_h), border_radius=3)
    if fill > 0:
        pg.draw.rect(screen, bar_col, (px + 10, y, fill, bar_h), border_radius=3)
    surf = font_sm.render(f"Ep reward  {ep_reward:+.1f}", True, C.COL_HUD_DIM)
    screen.blit(surf, (px + 8, y + bar_h + 2)); y += bar_h + surf.get_height() + 6
    y = divider(y)

    # ── Controls ───────────────────────────────────────────────────────
    y = label("CONTROLS", 8, y, font_sm, C.COL_HUD_DIM)
    keys = [
        ("T",     "Toggle training"),
        ("SPACE", "One episode"),
        ("G",     "Cycle speed"),
        ("R",     "Reset episode"),
        ("S / L", "Save / Load"),
        ("H",     "Help overlay"),
        ("Q/Esc", "Quit"),
    ]
    for key, desc in keys:
        surf_k = font_sm.render(key, True, C.COL_HUD_ACCENT)
        surf_d = font_sm.render(desc, True, C.COL_HUD_DIM)
        screen.blit(surf_k, (px + 8, y))
        screen.blit(surf_d, (px + 52, y))
        y += surf_k.get_height() + 1

    # ── Help / reward overlay (toggled with H) ─────────────────────────
    if overlay_help:
        help_lines = [
            "Reward shaping ():",
            "* Approach ball  (+)  when not touching",
            "* Touch bonus   (+0.5) on contact",
            "* Ball -> goal   shaping  ×2.0",
            "* Ball moves right  +0.015/px",
            "* Behind ball    +0.20 / step",
            "* In front ball  -0.12 / step",
            "* Sweet-spot approach  ×0.08",
            "* Misaligned kick  penalty ×0.6",
            "* Goal  +120   *  Out  -8",
            "State: pos(16) + goal(16) + vel(4) + touch - ",
        ]
        box_h = len(help_lines) * 18 + 16
        box_y = C.FIELD_H - box_h - 10
        ov = pg.Surface((C.FIELD_W - 20, box_h), pg.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        screen.blit(ov, (10, box_y))
        for i, line in enumerate(help_lines):
            col = C.COL_HUD_ACCENT if i == 0 else C.COL_HUD_TEXT
            surf = font_sm.render(line, True, col)
            screen.blit(surf, (18, box_y + 8 + i * 18))


def draw_scene(
    screen:       pg.Surface,
    font_lg:      pg.font.Font,
    font_sm:      pg.font.Font,
    env:          SoccerEnv,
    last_action:  int,
    episode:      int,
    step:         int,
    ep_reward:    float,
    eps:          float,
    training:     bool,
    speed_name:   str,
    win_window:   deque,
    rew_window:   deque,
    steps_window: deque,
    q_states:     int,
    overlay_help: bool,) -> None:
    """
    Render a complete frame: field, markings, agents, overlays, and HUD panel.
    """
    # ── Field background ──────────────────────────────────────────────
    field_rect = pg.Rect(0, 0, C.FIELD_W, C.FIELD_H)
    screen.fill(C.COL_GRASS, field_rect)
    # Panel area background (very dark).
    screen.fill((15, 15, 15), pg.Rect(C.FIELD_W, 0, C.PANEL_W, C.FIELD_H))

    # ── Field markings ────────────────────────────────────────────────
    _draw_field_markings(screen)

    # ── Goal ─────────────────────────────────────────────────────────
    pg.draw.rect(screen, C.COL_GOAL, (C.GOAL_X, C.GOAL_Y0, C.GOAL_THICKNESS, C.GOAL_HEIGHT))
    # Goal post highlights.
    pg.draw.line(screen, (255, 255, 100), (C.GOAL_X, C.GOAL_Y0), (C.GOAL_X + C.GOAL_THICKNESS, C.GOAL_Y0), 2)
    pg.draw.line(screen, (255, 255, 100), (C.GOAL_X, C.GOAL_Y1), (C.GOAL_X + C.GOAL_THICKNESS, C.GOAL_Y1), 2)

    # ── Sweet-spot marker ────────────────────────────────────────────
    sx, sy = env.sweet_spot()
    pg.draw.circle(screen, C.COL_SWEET, (int(sx), int(sy)), 6)
    pg.draw.circle(screen, (180, 180, 80), (int(sx), int(sy)), 6, 1)

    # ── Alignment guide line ─────────────────────────────────────────
    _draw_alignment_guide(screen, env)

    # ── Ball ─────────────────────────────────────────────────────────
    b_rect = env._ball_rect()
    pg.draw.rect(screen, C.COL_BALL, b_rect, border_radius=3)
    # Ball shadow (subtle depth cue).
    shadow = b_rect.move(2, 2)
    shadow_surf = pg.Surface((shadow.w, shadow.h), pg.SRCALPHA)
    shadow_surf.fill((0, 0, 0, 60))
    screen.blit(shadow_surf, shadow.topleft)

    # ── Agent ────────────────────────────────────────────────────────
    agent_col = C.COL_AGENT_KICK if last_action == 4 else C.COL_AGENT
    a_rect = env._agent_rect()
    pg.draw.rect(screen, agent_col, a_rect, border_radius=4)
    # Thin white border on agent.
    pg.draw.rect(screen, (255, 255, 255, 80), a_rect, 1, border_radius=4)

    # ── Direction indicator ──────────────────────────────────────────
    _draw_agent_direction(screen, env, last_action)

    # ── HUD panel ────────────────────────────────────────────────────
    _draw_hud_panel(
        screen, font_lg, font_sm,
        episode, step, ep_reward, eps,
        training, speed_name,
        win_window, rew_window, steps_window,
        q_states, overlay_help,)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Initialise Pygame, create the environment and agent, then enter the
    main event / render loop.
    """
    pg.init()
    screen = pg.display.set_mode((C.W, C.H))
    pg.display.set_caption("Pygame RL Soccer  *  Q-Learning ")

    font_lg = pg.font.SysFont("consolas", 20, bold=True)
    font_sm = pg.font.SysFont("consolas", 14)

    env   = SoccerEnv()
    agent = QLearner()

    # Rolling metric windows.
    win_window   = deque(maxlen=200)
    rew_window   = deque(maxlen=100)
    steps_window = deque(maxlen=100)

    # ── Auto-load checkpoint ──────────────────────────────────────────
    C.SAVES_DIR.mkdir(parents=True, exist_ok=True)
    agent.load()
    episode_0, eps_0 = load_training_state()
    episode = episode_0 or 0
    eps     = eps_0 if eps_0 is not None else epsilon_schedule(episode)
    print(f"[INFO] Starting from episode={episode:,}, eps={eps:.3f}")

    # Speed profile.
    speed_idx = C.SPEED_INDEX_DEFAULT
    speed_name, fps_cap, steps_per_tick = C.SPEEDS[speed_idx]
    clock = pg.time.Clock()

    training     = False
    overlay_help = False
    last_action  = -1   # -1 = no prior action (inertia disabled for first step)

    ep_reward = 0.0
    step      = 0
    s         = env.get_state()

    def end_episode(scored: bool) -> None:
        """Update metrics and increment episode counter."""
        nonlocal episode, eps
        win_window.append(1 if scored else 0)
        rew_window.append(ep_reward)
        if scored:
            steps_window.append(step)
        episode += 1
        eps = epsilon_schedule(episode)
        if episode % C.AUTOSAVE_EVERY == 0:
            agent.save(); save_training_state(episode, eps)

    running = True
    while running:
        # ── Event handling ────────────────────────────────────────────
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    running = False

                elif event.key == pg.K_t:
                    training = not training

                elif event.key == pg.K_g:
                    speed_idx = (speed_idx + 1) % len(C.SPEEDS)
                    speed_name, fps_cap, steps_per_tick = C.SPEEDS[speed_idx]

                elif event.key == pg.K_r:
                    s = env.reset(); ep_reward = 0.0; step = 0; last_action = -1

                elif event.key == pg.K_s:
                    agent.save(); save_training_state(episode, eps)

                elif event.key == pg.K_l:
                    agent.load()

                elif event.key == pg.K_h:
                    overlay_help = not overlay_help

                elif event.key == pg.K_SPACE:
                    # Run exactly one fully-rendered training episode.
                    s = env.reset(); ep_reward = 0.0; step = 0
                    done = False
                    eps_one = max(epsilon_schedule(episode), C.EPS_END)
                    while not done:
                        last_action = agent.act(s, eps_one, last_action)
                        sp, r, done, _ = env.step(last_action)
                        agent.update(s, last_action, r, sp, done)
                        s = sp; ep_reward += r; step += 1
                        draw_scene(
                            screen, font_lg, font_sm, env, last_action,
                            episode, step, ep_reward, eps_one,
                            True, speed_name,
                            win_window, rew_window, steps_window,
                            len(agent.Q), overlay_help,)
                        pg.display.flip()
                        clock.tick(fps_cap)
                        for ev in pg.event.get():
                            if ev.type == pg.QUIT:
                                done = True; running = False
                    end_episode(env._scored())

        # ── Continuous training ───────────────────────────────────────
        if training:
            for _ in range(steps_per_tick):
                if env.done:
                    end_episode(env._scored())
                    s = env.reset(); ep_reward = 0.0; step = 0; last_action = -1

                last_action = agent.act(s, eps, last_action)
                sp, r, done, _ = env.step(last_action)
                agent.update(s, last_action, r, sp, done)
                s = sp; ep_reward += r; step += 1

        # ── Render ───────────────────────────────────────────────────
        draw_scene(
            screen, font_lg, font_sm, env, last_action,
            episode, step, ep_reward, eps,
            training, speed_name,
            win_window, rew_window, steps_window,
            len(agent.Q), overlay_help,)
        pg.display.flip()
        clock.tick(fps_cap)

    # ── Clean-up save ─────────────────────────────────────────────────
    agent.save()
    save_training_state(episode, eps)
    pg.quit()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    run()
