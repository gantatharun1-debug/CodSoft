import math
import random
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tic-Tac-Toe AI",
    page_icon="🎮",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a0533 0%, #0a0a0f 50%, #001a33 100%) !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 1rem 3rem !important; max-width: 600px !important; }

/* ── Title ── */
.game-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #00d4ff, #7b2fff, #ff006e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
    filter: drop-shadow(0 0 20px rgba(123,47,255,0.5));
}
.game-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    text-align: center;
    color: #4a4a6a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Score Board ── */
.scoreboard {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 1rem 0;
}
.score-card {
    flex: 1;
    max-width: 130px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 8px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.score-card.you   { border-color: rgba(0,212,255,0.3);  background: rgba(0,212,255,0.06); }
.score-card.ai    { border-color: rgba(255,0,110,0.3);  background: rgba(255,0,110,0.06); }
.score-card.draw  { border-color: rgba(123,47,255,0.3); background: rgba(123,47,255,0.06); }
.score-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6a6a8a;
    margin-bottom: 4px;
}
.score-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}
.you  .score-value { color: #00d4ff; text-shadow: 0 0 12px rgba(0,212,255,0.5); }
.ai   .score-value { color: #ff006e; text-shadow: 0 0 12px rgba(255,0,110,0.5); }
.draw .score-value { color: #7b2fff; text-shadow: 0 0 12px rgba(123,47,255,0.5); }

/* ── Status Banner ── */
.status-banner {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
    letter-spacing: 2px;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 1rem 0;
    text-transform: uppercase;
}
.status-your-turn  { background: rgba(0,212,255,0.1);  color: #00d4ff; border: 1px solid rgba(0,212,255,0.25); }
.status-ai-turn    { background: rgba(255,0,110,0.1);  color: #ff006e; border: 1px solid rgba(255,0,110,0.25); }
.status-you-win    { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.5);  font-size: 1.2rem; }
.status-ai-win     { background: rgba(255,0,110,0.15); color: #ff006e; border: 1px solid rgba(255,0,110,0.5);  font-size: 1.2rem; }
.status-draw       { background: rgba(123,47,255,0.15);color: #b36fff; border: 1px solid rgba(123,47,255,0.5); font-size: 1.2rem; }

/* ── Board ── */
.board-wrap {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    max-width: 320px;
    margin: 1rem auto;
    padding: 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
}
.cell {
    aspect-ratio: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 2px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.03);
    min-height: 90px;
    user-select: none;
}
.cell.empty {
    color: transparent;
    border-color: rgba(255,255,255,0.08);
}
.cell.empty:hover {
    background: rgba(0,212,255,0.08);
    border-color: rgba(0,212,255,0.3);
    transform: scale(1.04);
    box-shadow: 0 0 16px rgba(0,212,255,0.15);
}
.cell.x {
    color: #00d4ff;
    border-color: rgba(0,212,255,0.35);
    background: rgba(0,212,255,0.06);
    text-shadow: 0 0 20px rgba(0,212,255,0.6);
    box-shadow: inset 0 0 20px rgba(0,212,255,0.05), 0 0 10px rgba(0,212,255,0.1);
}
.cell.o {
    color: #ff006e;
    border-color: rgba(255,0,110,0.35);
    background: rgba(255,0,110,0.06);
    text-shadow: 0 0 20px rgba(255,0,110,0.6);
    box-shadow: inset 0 0 20px rgba(255,0,110,0.05), 0 0 10px rgba(255,0,110,0.1);
}
.cell.win {
    animation: pulse-win 1s ease-in-out infinite alternate;
}
.cell.x.win { background: rgba(0,212,255,0.18); border-color: rgba(0,212,255,0.8); box-shadow: 0 0 24px rgba(0,212,255,0.4); }
.cell.o.win { background: rgba(255,0,110,0.18); border-color: rgba(255,0,110,0.8); box-shadow: 0 0 24px rgba(255,0,110,0.4); }
@keyframes pulse-win {
    from { transform: scale(1); }
    to   { transform: scale(1.06); }
}

/* ── Controls ── */
.controls-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin: 1rem 0 0.5rem;
    flex-wrap: wrap;
}
.diff-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #4a4a6a;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 4px;
}

/* ── Streamlit button overrides ── */
div.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.05) !important;
    color: #c0c0d0 !important;
    transition: all 0.2s ease !important;
    font-size: 0.8rem !important;
    padding: 0.45rem 1rem !important;
}
div.stButton > button:hover {
    background: rgba(123,47,255,0.2) !important;
    border-color: rgba(123,47,255,0.5) !important;
    color: #b36fff !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Cell buttons (board) ── */
.cell-btn > div.stButton > button {
    width: 90px !important;
    height: 90px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    padding: 0 !important;
    border-radius: 10px !important;
    line-height: 1 !important;
}

/* ── Select box ── */
div.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #c0c0d0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

WINNING_COMBOS = [
    [0,1,2],[3,4,5],[6,7,8],   # rows
    [0,3,6],[1,4,7],[2,5,8],   # columns
    [0,4,8],[2,4,6]            # diagonals
]

def check_winner(board, player):
    return any(all(board[i] == player for i in combo) for combo in WINNING_COMBOS)

def get_winning_cells(board, player):
    for combo in WINNING_COMBOS:
        if all(board[i] == player for i in combo):
            return combo
    return []

def is_draw(board):
    return ' ' not in board and not check_winner(board,'X') and not check_winner(board,'O')

def is_game_over(board):
    return check_winner(board,'X') or check_winner(board,'O') or is_draw(board)

def get_empty(board):
    return [i for i,c in enumerate(board) if c == ' ']

def minimax(board, depth, is_max, alpha, beta):
    if check_winner(board,'O'): return 10 - depth
    if check_winner(board,'X'): return depth - 10
    if is_draw(board):          return 0

    if is_max:
        best = -math.inf
        for pos in get_empty(board):
            board[pos] = 'O'
            best = max(best, minimax(board, depth+1, False, alpha, beta))
            board[pos] = ' '
            alpha = max(alpha, best)
            if beta <= alpha: break
        return best
    else:
        best = math.inf
        for pos in get_empty(board):
            board[pos] = 'X'
            best = min(best, minimax(board, depth+1, True, alpha, beta))
            board[pos] = ' '
            beta = min(beta, best)
            if beta <= alpha: break
        return best

def best_move(board, difficulty):
    empty = get_empty(board)
    if not empty:
        return None
    # Easy: 75% random
    if difficulty == 'Easy' and random.random() < 0.75:
        return random.choice(empty)
    # Medium: 40% random
    if difficulty == 'Medium' and random.random() < 0.40:
        return random.choice(empty)
    # Hard: full minimax
    best_sc, best_pos = -math.inf, None
    for pos in empty:
        board[pos] = 'O'
        sc = minimax(board, 0, False, -math.inf, math.inf)
        board[pos] = ' '
        if sc > best_sc:
            best_sc, best_pos = sc, pos
    return best_pos


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        'board':       [' '] * 9,
        'turn':        'X',          # X = human, O = AI
        'game_over':   False,
        'winner':      None,         # 'X', 'O', 'Draw', None
        'win_cells':   [],
        'score':       {'You': 0, 'AI': 0, 'Draws': 0},
        'difficulty':  'Hard',
        'first_move':  'You',
        'message':     '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_board():
    st.session_state.board     = [' '] * 9
    st.session_state.game_over = False
    st.session_state.winner    = None
    st.session_state.win_cells = []
    st.session_state.message   = ''
    # who goes first
    if st.session_state.first_move == 'AI':
        st.session_state.turn = 'O'
        do_ai_move()
    else:
        st.session_state.turn = 'X'

def do_ai_move():
    if st.session_state.game_over:
        return
    pos = best_move(st.session_state.board, st.session_state.difficulty)
    if pos is not None:
        st.session_state.board[pos] = 'O'
        if check_winner(st.session_state.board, 'O'):
            st.session_state.game_over = True
            st.session_state.winner    = 'O'
            st.session_state.win_cells = get_winning_cells(st.session_state.board, 'O')
            st.session_state.score['AI'] += 1
        elif is_draw(st.session_state.board):
            st.session_state.game_over = True
            st.session_state.winner    = 'Draw'
            st.session_state.score['Draws'] += 1
        else:
            st.session_state.turn = 'X'

def handle_click(pos):
    if st.session_state.game_over:
        return
    if st.session_state.board[pos] != ' ':
        return
    if st.session_state.turn != 'X':
        return

    # Human move
    st.session_state.board[pos] = 'X'
    if check_winner(st.session_state.board, 'X'):
        st.session_state.game_over = True
        st.session_state.winner    = 'X'
        st.session_state.win_cells = get_winning_cells(st.session_state.board, 'X')
        st.session_state.score['You'] += 1
        return
    if is_draw(st.session_state.board):
        st.session_state.game_over = True
        st.session_state.winner    = 'Draw'
        st.session_state.score['Draws'] += 1
        return

    # AI move
    st.session_state.turn = 'O'
    do_ai_move()


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════════════════════

init_state()

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="game-title">Tic · Tac · Toe</div>', unsafe_allow_html=True)
st.markdown('<div class="game-subtitle">AI · Minimax · Alpha-Beta Pruning</div>', unsafe_allow_html=True)

# ── Score Board ────────────────────────────────────────────────────────────────
sc = st.session_state.score
st.markdown(f"""
<div class="scoreboard">
  <div class="score-card you">
    <div class="score-label">You (X)</div>
    <div class="score-value">{sc['You']}</div>
  </div>
  <div class="score-card draw">
    <div class="score-label">Draws</div>
    <div class="score-value">{sc['Draws']}</div>
  </div>
  <div class="score-card ai">
    <div class="score-label">AI (O)</div>
    <div class="score-value">{sc['AI']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Status Banner ──────────────────────────────────────────────────────────────
if st.session_state.game_over:
    w = st.session_state.winner
    if w == 'X':
        st.markdown('<div class="status-banner status-you-win">🎉 You Win!</div>', unsafe_allow_html=True)
    elif w == 'O':
        st.markdown('<div class="status-banner status-ai-win">🤖 AI Wins!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-banner status-draw">🤝 It\'s a Draw!</div>', unsafe_allow_html=True)
else:
    if st.session_state.turn == 'X':
        st.markdown('<div class="status-banner status-your-turn">⚡ Your Turn — Click a Cell</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-banner status-ai-turn">🤖 AI is Thinking…</div>', unsafe_allow_html=True)

# ── Game Board ─────────────────────────────────────────────────────────────────
board     = st.session_state.board
win_cells = st.session_state.win_cells

# Cell symbol display mapping
SYMBOLS = {'X': 'X', 'O': 'O', ' ': '·'}
COLORS  = {'X': '#00d4ff', 'O': '#ff006e', ' ': '#1e1e2e'}

cols_grid = [st.columns(3) for _ in range(3)]

for row in range(3):
    for col in range(3):
        pos   = row * 3 + col
        cell  = board[pos]
        is_win = pos in win_cells

        with cols_grid[row][col]:
            # Style the button label
            symbol = SYMBOLS[cell]
            color  = COLORS[cell]

            if cell == ' ' and not st.session_state.game_over:
                # Clickable empty cell
                if st.button('·', key=f'cell_{pos}',
                             use_container_width=True,
                             help=f'Position {pos}'):
                    handle_click(pos)
                    st.rerun()
            else:
                # Filled cell or game over — show symbol, disabled
                btn_label = symbol if cell != ' ' else '·'
                st.button(btn_label, key=f'cell_{pos}',
                          use_container_width=True,
                          disabled=True)

# ── Controls Row ───────────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([2, 2, 2])

with c1:
    st.markdown('<div class="diff-label">Difficulty</div>', unsafe_allow_html=True)
    diff = st.selectbox(
        'diff', ['Easy', 'Medium', 'Hard'],
        index=['Easy','Medium','Hard'].index(st.session_state.difficulty),
        label_visibility='collapsed',
        key='diff_select'
    )
    if diff != st.session_state.difficulty:
        st.session_state.difficulty = diff
        reset_board()
        st.rerun()

with c2:
    st.markdown('<div class="diff-label">First Move</div>', unsafe_allow_html=True)
    first = st.selectbox(
        'first', ['You', 'AI'],
        index=['You','AI'].index(st.session_state.first_move),
        label_visibility='collapsed',
        key='first_select'
    )
    if first != st.session_state.first_move:
        st.session_state.first_move = first
        reset_board()
        st.rerun()

with c3:
    st.markdown('<div class="diff-label">Actions</div>', unsafe_allow_html=True)
    if st.button('🔄  New Game', use_container_width=True):
        reset_board()
        st.rerun()

# ── Reset All Scores ───────────────────────────────────────────────────────────
col_l, col_r = st.columns([3,1])
with col_r:
    if st.button('Reset Score', use_container_width=True):
        st.session_state.score = {'You': 0, 'AI': 0, 'Draws': 0}
        reset_board()
        st.rerun()

# ── How to Play ────────────────────────────────────────────────────────────────
with st.expander('ℹ️  How to Play'):
    st.markdown("""
<div style="font-family:'Rajdhani',sans-serif; color:#8888aa; font-size:0.9rem; line-height:1.8;">
<b style="color:#00d4ff">You are X · AI is O</b><br><br>
• Click any empty cell to place your move<br>
• First to align 3 symbols (row / column / diagonal) wins<br>
• <b style="color:#00d4ff">Easy</b> — AI plays mostly random moves<br>
• <b style="color:#b36fff">Medium</b> — AI plays smart moves 60% of the time<br>
• <b style="color:#ff006e">Hard</b> — AI uses full Minimax + Alpha-Beta Pruning and <u>never loses</u>
</div>
""", unsafe_allow_html=True)
