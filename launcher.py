"""
launcher.py - Pygame RL Soccer launcher.

Run this file to pick an agent and start training or watching.

    python launcher.py

Or launch directly:

    python soccer_rl.py    # Tabular Q-learning
    python soccer_dqn.py   # Deep Q-Network
"""

import pygame as pg
import sys
import config as C

BG        = (15,  20,  30)
PANEL_COL = (25,  32,  45)
BORDER    = (50,  60,  80)
ACCENT    = (90, 190, 255)
TEXT_MAIN = (230, 230, 230)
TEXT_DIM  = (130, 130, 150)
GREEN     = (80,  210,  80)
ORANGE    = (255, 165,  50)

W, H = 760, 420


def _badge(screen, font, path, col, x, y):
    exists = path.exists()
    c   = col if exists else TEXT_DIM
    txt = "checkpoint found" if exists else "no checkpoint - fresh start"
    screen.blit(font.render(txt, True, c), (x, y))


def run_launcher() -> None:
    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("Pygame RL Soccer")
    clock  = pg.time.Clock()

    font_xl = pg.font.SysFont("consolas", 28, bold=True)
    font_lg = pg.font.SysFont("consolas", 18, bold=True)
    font_md = pg.font.SysFont("consolas", 14)
    font_sm = pg.font.SysFont("consolas", 12)

    card_w, card_h = 330, 260
    gap   = 20
    total = 2 * card_w + gap
    sx    = (W - total) // 2
    card_y = 110

    card_ql  = pg.Rect(sx,            card_y, card_w, card_h)
    card_dqn = pg.Rect(sx+card_w+gap, card_y, card_w, card_h)

    CARDS = [
        (
            card_ql, "Q-Learning", "Tabular agent",
            C.Q_PATH, GREEN, "[1]",
            [
                "Discrete state space",
                "16 spatial bins + 4 velocity bins",
                "Close-range fine bins",
                "Action inertia smoothing",
                "~92% win rate after training",
            ],
        ),
        (
            card_dqn, "Deep Q-Network", "Neural network agent",
            C.DQN_MODEL_PATH, ORANGE, "[2]",
            [
                "Continuous state - 10 raw floats",
                "MLP: 10 -> 128 -> 128 -> 128 -> 5",
                "Experience replay (100k transitions)",
                "Target network (synced every 500 steps)",
                "Reaches 100% win rate",
            ],
        ),
    ]

    choice = None
    while choice is None:
        mx, my = pg.mouse.get_pos()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit(); sys.exit()
            elif event.type == pg.KEYDOWN:
                if   event.key == pg.K_1: choice = "ql"
                elif event.key == pg.K_2: choice = "dqn"
                elif event.key in (pg.K_ESCAPE, pg.K_q):
                    pg.quit(); sys.exit()
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if card_ql.collidepoint(mx, my):  choice = "ql"
                if card_dqn.collidepoint(mx, my): choice = "dqn"

        screen.fill(BG)

        title = font_xl.render("PYGAME  RL  SOCCER", True, ACCENT)
        screen.blit(title, ((W - title.get_width()) // 2, 20))
        sub = font_md.render("Select an agent to train or watch", True, TEXT_DIM)
        screen.blit(sub, ((W - sub.get_width()) // 2, 58))
        pg.draw.line(screen, BORDER, (40, 84), (W-40, 84), 1)

        for card, label, sublabel, badge_path, badge_col, key, bullets in CARDS:
            hov        = card.collidepoint(mx, my)
            border_col = badge_col if hov else BORDER
            bg_col     = (35, 45, 65) if hov else PANEL_COL

            pg.draw.rect(screen, bg_col,     card, border_radius=10)
            pg.draw.rect(screen, border_col, card, 2, border_radius=10)

            cx = card.x + 14
            cy = card.y + 12

            ks = font_sm.render(key, True, TEXT_DIM)
            screen.blit(ks, (card.right - ks.get_width() - 10, cy))

            ls = font_lg.render(label, True, badge_col if hov else TEXT_MAIN)
            screen.blit(ls, (cx, cy)); cy += ls.get_height() + 2

            ss = font_sm.render(sublabel, True, TEXT_DIM)
            screen.blit(ss, (cx, cy)); cy += ss.get_height() + 6

            pg.draw.line(screen, BORDER, (cx, cy), (card.right-14, cy), 1)
            cy += 7

            for b in bullets:
                bs = font_sm.render(b, True, TEXT_MAIN if hov else TEXT_DIM)
                screen.blit(bs, (cx, cy)); cy += bs.get_height() + 4

            cy += 4
            _badge(screen, font_sm, badge_path, badge_col, cx, cy)

        foot = font_sm.render(
            "Click a card  or  press  1 / 2  to launch  -  ESC to exit",
            True, TEXT_DIM,
        )
        screen.blit(foot, ((W - foot.get_width()) // 2, H - 22))

        pg.display.flip()
        clock.tick(60)

    pg.quit()

    if choice == "ql":
        import soccer_rl
        soccer_rl.run()
    else:
        import soccer_dqn
        soccer_dqn.run()


if __name__ == "__main__":
    run_launcher()
