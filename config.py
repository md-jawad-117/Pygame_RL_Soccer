"""
config.py - Central configuration for Pygame RL Soccer.

Edit values here to tune the environment, physics, or learning algorithm.
All paths are relative so the project works on any machine.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent
SAVES_DIR:    Path = PROJECT_ROOT / "saves"
Q_PATH:       Path = SAVES_DIR / "q_table.pkl"
STATE_PATH:   Path = SAVES_DIR / "training_state.pkl"
DQN_MODEL_PATH: Path = SAVES_DIR / "dqn_model.pt"
DQN_STATE_PATH: Path = SAVES_DIR / "dqn_training_state.pkl"

# ---------------------------------------------------------------------------
# Window and field
# ---------------------------------------------------------------------------
PANEL_W: int = 220
FIELD_W: int = 900
FIELD_H: int = 600
W: int = FIELD_W + PANEL_W
H: int = FIELD_H

FIELD_MARGIN: int = 30

# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------
GOAL_HEIGHT:    int = 180
GOAL_THICKNESS: int = 8
GOAL_X:         int = FIELD_W - GOAL_THICKNESS
GOAL_Y0:        int = (FIELD_H - GOAL_HEIGHT) // 2
GOAL_Y1:        int = GOAL_Y0 + GOAL_HEIGHT

# ---------------------------------------------------------------------------
# Agent and ball physics
# ---------------------------------------------------------------------------
AGENT_SIZE:     int   = 20
BALL_SIZE:      int   = 12
AGENT_SPEED:    int   = 5
FRICTION:       float = 0.965
BALL_MAX_SPEED: float = 11.5
KICK_IMPULSE:   float = 11.0

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ACTIONS   = ["UP", "DOWN", "LEFT", "RIGHT", "KICK"]
N_ACTIONS: int = len(ACTIONS)

# ---------------------------------------------------------------------------
# Speed profiles
# ---------------------------------------------------------------------------
SPEEDS = [
    ("SLOW",  60,   1),
    ("FAST",  1200, 4),
    ("HYPER", 0,    20),
]
SPEED_INDEX_DEFAULT: int = 0
HYPER_RENDER_EVERY:  int = 20

# ---------------------------------------------------------------------------
# Q-learning hyperparameters
# ---------------------------------------------------------------------------
GAMMA:              float = 0.985
ALPHA:              float = 0.25
EPS_START:          float = 1.0
EPS_END:            float = 0.08
EPS_DECAY_EPISODES: int   = 25_000
MAX_STEPS_PER_EP:   int   = 1_500
AUTOSAVE_EVERY:     int   = 100

# Q-learning state discretisation
BINS:     int = 16
VEL_BINS: int = 4

DX_RANGE = (-FIELD_W // 2, FIELD_W // 2)
DY_RANGE = (-FIELD_H // 2, FIELD_H // 2)
GX_RANGE = (-FIELD_W // 2, FIELD_W // 2)
GY_RANGE = (-FIELD_H // 2, FIELD_H // 2)
VX_RANGE = (-BALL_MAX_SPEED, BALL_MAX_SPEED)
VY_RANGE = (-BALL_MAX_SPEED, BALL_MAX_SPEED)

CLOSE_RANGE: float = 80.0
CLOSE_BINS:  int   = 8
ACTION_INERTIA: float = 0.15

# ---------------------------------------------------------------------------
# Reward terms (shared by both agents)
# ---------------------------------------------------------------------------
STEP_PENALTY:  float = -0.004
GOAL_REWARD:   float = +120.0
OUT_REWARD:    float = -8.0
TOUCH_BONUS:   float = +0.5

BALL_GOAL_SHAPING: float = 2.0
BALL_X_PROGRESS:   float = 0.015
APPROACH_BALL:     float = 0.08

SWEET_DIST:   float = 10.0
SWEET_WEIGHT: float = 0.08
ALIGN_SCALE:  float = 0.6
ALIGN_THRESH: float = 0.65
BACKSIDE_BONUS:          float = +0.20
FRONT_PENALTY:           float = -0.12
BEHIND_THRESH:           float = -5.0
MISALIGNED_KICK_PENALTY: float = -0.6

# ---------------------------------------------------------------------------
# DQN hyperparameters
# ---------------------------------------------------------------------------
DQN_STATE_DIM:       int   = 10
DQN_HIDDEN_DIM:      int   = 128
DQN_N_LAYERS:        int   = 3
DQN_GAMMA:           float = 0.99
DQN_LR:              float = 1e-3
DQN_BATCH_SIZE:      int   = 256
DQN_BUFFER_SIZE:     int   = 100_000
DQN_TARGET_UPDATE:   int   = 500
DQN_TRAIN_START:     int   = 1_000
DQN_TRAIN_FREQ:      int   = 4
DQN_EPS_START:       float = 1.0
DQN_EPS_END:         float = 0.05
DQN_EPS_DECAY_STEPS: int   = 150_000
DQN_AUTOSAVE_EVERY:  int   = 200

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COL_GRASS      = (18,  90,  40)
COL_MARKING    = (255, 255, 255)
COL_GOAL       = (250, 250,  50)
COL_AGENT      = (220,  50,  50)
COL_AGENT_KICK = (255, 160,  80)
COL_BALL       = (245, 245, 245)
COL_SWEET      = (255, 255, 180)
COL_ALIGN_OK   = ( 80, 220,  80)
COL_ALIGN_BAD  = (220, 160,  40)
COL_HUD_BG     = ( 10,  10,  10, 180)
COL_HUD_TEXT   = (230, 230, 230)
COL_HUD_DIM    = (140, 140, 140)
COL_HUD_ACCENT = ( 90, 190, 255)
COL_DIVIDER    = ( 60,  60,  60)
