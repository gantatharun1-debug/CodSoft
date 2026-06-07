# 🎮 Tic-Tac-Toe AI — Minimax + Alpha-Beta Pruning

An unbeatable Tic-Tac-Toe AI built with **Python**, powered by the **Minimax algorithm with Alpha-Beta Pruning**. Available as a step-by-step Jupyter Notebook and a polished **Streamlit** web application with a cyberpunk-themed UI.

> **AI Goal:** Never lose — always win or draw (on Hard difficulty).

---

## 📁 Project Structure

```
├── TicTacToe_AI.ipynb       # Master notebook — algorithm walkthrough + terminal game
└── tictactoe_app.py         # Streamlit web application with full UI
```

---

## ✨ Features

- **Minimax Algorithm** — AI recursively evaluates all possible future game states
- **Alpha-Beta Pruning** — Optimised Minimax that skips provably irrelevant branches, making the AI significantly faster
- **3 Difficulty Levels** — Easy (75% random), Medium (40% random), Hard (full Minimax — never loses)
- **First-move Choice** — Player can choose whether they or the AI goes first
- **Winning Cell Highlight** — The three winning cells pulse with a glow animation
- **Persistent Score Tracker** — Tracks wins, losses, and draws across games within a session
- **Performance Benchmark** — Notebook Section 10 measures and compares speed with vs. without pruning
- **AI vs AI Self-Play Test** — Notebook Section 9 runs 50 simulated games to verify 0% loss rate on Hard

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|-----------|----------------|
| Core AI Logic | Python standard library (`math`, `random`) |
| Web Application | `streamlit` |
| UI Fonts & Styling | Google Fonts (Orbitron, Rajdhani), custom CSS |
| Notebook Environment | Jupyter / Google Colab |

No external ML libraries or datasets are required.

---

## ⚙️ Installation

```bash
# 1. Clone / download the project
git clone <your-repo-url>
cd tictactoe-ai

# 2. Install dependencies
pip install streamlit
```

No additional packages needed for the notebook — only Python's standard library is used.

---

## 🚀 Usage

### Option A — Streamlit Web App

```bash
streamlit run tictactoe_app.py
```

Opens at **http://localhost:8501**

**UI Features:**
- Click any empty cell to place your move (X)
- AI responds immediately as O
- Difficulty and first-move dropdowns reset the board on change
- **🔄 New Game** starts a fresh round while keeping scores
- **Reset Score** zeroes the scoreboard
- Expand **ℹ️ How to Play** for in-app instructions

### Option B — Jupyter / Google Colab Notebook

Open `TicTacToe_AI.ipynb` and run all cells top-to-bottom with `Shift+Enter`.

| Section | What It Covers |
|---------|----------------|
| Section 1 | Imports (`math`, `time`, `random`) |
| Section 2 | Board creation, display, and position guide |
| Section 3 | Win detection, draw detection, move helpers |
| Section 4 | Minimax algorithm with Alpha-Beta Pruning |
| Section 5 | AI move selector with difficulty levels |
| Section 6 | Human input handler with validation |
| Section 7 | Main game loop with score tracker |
| Section 8 | Interactive terminal game (select difficulty → play) |
| Section 9 | AI vs AI self-play test (50 games, Hard vs Random) |
| Section 10 | Alpha-Beta pruning speed comparison benchmark |
| Section 11 | Summary, project stats, and key concepts |

---

## 🧠 How the AI Works

### Board Representation

The board is a flat list of 9 cells (indices 0–8):

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

### Minimax Algorithm

The AI simulates every possible future game state in a recursive game tree:

- **AI (O) is the Maximising player** — tries to reach the highest score
- **Human (X) is the Minimising player** — tries to reach the lowest score

**Terminal state scores:**

| Outcome | Score |
|---------|-------|
| AI wins | `+10 − depth` (prefer faster wins) |
| Human wins | `depth − 10` (prefer slower losses) |
| Draw | `0` |

### Alpha-Beta Pruning

Two bounds are tracked at every node:
- **Alpha** — best score the AI can guarantee (starts at −∞)
- **Beta** — best score the Human can guarantee (starts at +∞)

When `beta ≤ alpha`, the remaining branches in that subtree are pruned — they can never influence the final decision. This dramatically reduces the number of nodes evaluated without changing the result.

### Difficulty Levels

| Level | Behaviour |
|-------|-----------|
| Easy | 75% random move, 25% Minimax |
| Medium | 40% random move, 60% Minimax |
| Hard | Always Minimax — **cannot lose** |

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| Board size | 3 × 3 = 9 cells |
| Possible unique games | 255,168 |
| Algorithm | Minimax + Alpha-Beta Pruning |
| AI loss rate on Hard | **0%** |
| External ML libraries | None |
| GPU required | No |
| Dataset required | No |

---

## 🎨 Streamlit UI Design

The web app uses a dark cyberpunk aesthetic:

- **Colour palette** — deep dark background (`#0a0a0f`) with cyan (`#00d4ff`) for the player and magenta (`#ff006e`) for the AI
- **Typography** — Orbitron (futuristic monospace) for scores and title, Rajdhani for labels
- **Board cells** — hover glow on empty cells; filled cells have distinct neon borders and text shadows; winning cells pulse with a CSS keyframe animation
- **Score cards** — frosted-glass panels with per-player accent colours

---

## 🏗️ Code Architecture

### Notebook (`TicTacToe_AI.ipynb`)

```
create_board / print_board
        ↓
check_winner / is_draw / is_game_over / get_empty_cells
        ↓
minimax(board, depth, is_maximizing, alpha, beta)   ← core AI
        ↓
best_move(board, difficulty)                         ← difficulty wrapper
        ↓
human_move / play_game                               ← game loop
        ↓
auto_play_test / minimax_no_pruning                  ← testing & benchmarks
```

### Streamlit App (`tictactoe_app.py`)

```
Game Logic (check_winner, minimax, best_move)
        ↓
Session State (board, turn, score, difficulty, first_move)
        ↓
handle_click(pos)  →  do_ai_move()                   ← event handlers
        ↓
Render (title → scoreboard → status banner → grid → controls)
```

---

## 📌 Key Concepts

- **Minimax** — exhaustive game-tree search that guarantees optimal play
- **Alpha-Beta Pruning** — branch-and-bound optimisation that reduces search space without sacrificing correctness
- **Depth penalty** — preferring wins in fewer moves makes the AI play more aggressively
- **Recursion** — each minimax call spawns child calls for every available position, unwinding on terminal states

---

## 🚀 Possible Extensions

- Add an **undo** button to take back the last move
- Extend to a **4×4 or 5×5 board** with depth-limited Minimax and a heuristic evaluation function
- Replace Minimax with a **Reinforcement Learning** agent (Q-learning or policy gradient)
- Add **multiplayer** (Human vs Human) mode
- Track win/draw statistics across sessions using a database or local file

---

## 📄 License

This project is for educational purposes. Feel free to adapt and extend it for your own learning or projects.
