"""
Counterpart — Streamlit version
================================
Same model wiring as server.py (enginev1 + board_to_matrix + move_to_int),
but the whole app — board, rules, and UI — runs in Streamlit instead of a
separate HTML/JS frontend + Flask backend. Deploy this one file (plus
engine.pt / move_to_int.json) to share.streamlit.io and you're done.
"""

import json
import os
import random

import chess
import numpy as np
import streamlit as st
import torch
from torch import nn

st.set_page_config(page_title="Counterpart", page_icon="♞", layout="centered")


# ======================================================================
# MODEL ARCHITECTURE — must match what you trained in chess_engine.ipynb
# ======================================================================
class enginev1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layerstack = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.layerstack(x)


# ======================================================================
# LOAD MODEL + move_to_int (cached so this only runs once per session)
# ======================================================================
@st.cache_resource
def load_model_and_moves():
    engine_path = "engine.pt"
    move_map_path = "move_to_int.json"
    hf_repo_id = os.environ.get("MODEL_REPO_ID")  # e.g. "yourusername/counterpart-engine"

    if not os.path.exists(engine_path):
        if not hf_repo_id:
            raise RuntimeError(
                "engine.pt not found locally, and MODEL_REPO_ID isn't set. "
                "Either add engine.pt via Git LFS, or upload it to a Hugging "
                "Face model repo and set MODEL_REPO_ID as a secret."
            )
        from huggingface_hub import hf_hub_download
        engine_path = hf_hub_download(repo_id=hf_repo_id, filename="engine.pt")

    if not os.path.exists(move_map_path):
        if hf_repo_id:
            from huggingface_hub import hf_hub_download
            move_map_path = hf_hub_download(repo_id=hf_repo_id, filename="move_to_int.json")
        else:
            raise RuntimeError("move_to_int.json not found locally, and no MODEL_REPO_ID set.")

    with open(move_map_path) as f:
        move_to_int = json.load(f)
    int_to_move = {v: k for k, v in move_to_int.items()}

    model = enginev1(num_classes=len(move_to_int))
    model.load_state_dict(torch.load(engine_path, map_location="cpu"))
    model.eval()
    return model, int_to_move


model, int_to_move = load_model_and_moves()


# ======================================================================
# board_to_matrix — identical encoding to training, do not change
# ======================================================================
def board_to_matrix(board: chess.Board):
    matrix = torch.zeros((13, 8, 8))
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color_offset = 0 if piece.color == chess.BLACK else 6
        matrix[piece_color_offset + piece_type, row, col] = 1
        matrix[12, 0, 0] = 1 if board.turn == chess.WHITE else 0
    return matrix


def predict_move(board: chess.Board) -> str:
    legal_moves = [m.uci() for m in board.legal_moves]
    with torch.no_grad():
        x = board_to_matrix(board).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    for idx in np.argsort(probs)[::-1]:
        candidate = int_to_move.get(idx)
        if candidate in legal_moves:
            return candidate
    return random.choice(legal_moves)  # shouldn't happen, safety net


# ======================================================================
# GAME STATE
# ======================================================================
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.player_color = chess.WHITE
    st.session_state.last_move = None
    st.session_state.error_msg = None
    st.session_state.selected = None
    st.session_state.pending_promo = None


def new_game(player_color):
    st.session_state.board = chess.Board()
    st.session_state.player_color = player_color
    st.session_state.last_move = None
    st.session_state.error_msg = None
    st.session_state.selected = None
    st.session_state.pending_promo = None


# ======================================================================
# SIDEBAR — controls
# ======================================================================
with st.sidebar:
    st.header("Game")
    color_choice = st.radio("Play as", ["White", "Black"], horizontal=True)
    if st.button("New game", use_container_width=True):
        new_game(chess.WHITE if color_choice == "White" else chess.BLACK)
        st.rerun()

    st.divider()
    st.caption("Move history")
    board = st.session_state.board
    san_list = []
    temp_board = chess.Board()
    for mv in board.move_stack:
        san_list.append(temp_board.san(mv))
        temp_board.push(mv)
    rows = [san_list[i:i+2] for i in range(0, len(san_list), 2)]
    for i, row in enumerate(rows, start=1):
        cols = st.columns([1, 3, 3])
        cols[0].write(f"{i}.")
        cols[1].write(row[0] if len(row) > 0 else "")
        cols[2].write(row[1] if len(row) > 1 else "")


# ======================================================================
# If it's the model's turn, let it move before rendering
# ======================================================================
board = st.session_state.board
if not board.is_game_over() and board.turn != st.session_state.player_color:
    with st.spinner("Model is thinking..."):
        uci = predict_move(board)
        move = chess.Move.from_uci(uci)
        board.push(move)
        st.session_state.last_move = move


# ======================================================================
# MAIN — clickable board
# ======================================================================
st.title("♞ Counterpart")
st.caption("A board for you and the model you trained. Click a piece, then click where it goes.")

st.markdown("""
<style>
div.stButton > button {
    font-size: 30px;
    height: 52px;
    width: 100%;
    padding: 0;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)

player_is_white = st.session_state.player_color == chess.WHITE
files = "abcdefgh"
ranks_order = list(range(8, 0, -1)) if player_is_white else list(range(1, 9))
files_order = list(files) if player_is_white else list(files[::-1])

selected = st.session_state.selected
pending_promo = st.session_state.pending_promo
interactive = (not board.is_game_over()
               and board.turn == st.session_state.player_color
               and pending_promo is None)

legal_targets = set()
if selected:
    for m in board.legal_moves:
        if chess.square_name(m.from_square) == selected:
            legal_targets.add(chess.square_name(m.to_square))

clicked_square = None
for rank in ranks_order:
    label_col, *sq_cols = st.columns([0.4] + [1]*8)
    label_col.markdown(f"<div style='padding-top:14px'>{rank}</div>", unsafe_allow_html=True)
    for i, file in enumerate(files_order):
        square_name = file + str(rank)
        piece = board.piece_at(chess.parse_square(square_name))
        label = piece.unicode_symbol() if piece else "\u00A0"
        is_selected = (square_name == selected)
        is_target = square_name in legal_targets
        btn_type = "primary" if (is_selected or is_target) else "secondary"
        if sq_cols[i].button(label, key=f"sq_{square_name}", type=btn_type,
                              use_container_width=True, disabled=not interactive):
            clicked_square = square_name

label_col, *sq_cols = st.columns([0.4] + [1]*8)
label_col.write("")
for i, file in enumerate(files_order):
    sq_cols[i].markdown(f"<div style='text-align:center'>{file}</div>", unsafe_allow_html=True)

# --- handle a click -----------------------------------------------------
if clicked_square and interactive:
    if selected is None:
        piece = board.piece_at(chess.parse_square(clicked_square))
        if piece and piece.color == st.session_state.player_color:
            st.session_state.selected = clicked_square
        st.rerun()
    else:
        if clicked_square == selected:
            st.session_state.selected = None
            st.rerun()
        matching = [m for m in board.legal_moves
                    if chess.square_name(m.from_square) == selected
                    and chess.square_name(m.to_square) == clicked_square]
        if matching:
            if any(m.promotion for m in matching):
                st.session_state.pending_promo = {"from": selected, "to": clicked_square}
                st.session_state.selected = None
            else:
                board.push(matching[0])
                st.session_state.last_move = matching[0]
                st.session_state.selected = None
            st.rerun()
        else:
            piece = board.piece_at(chess.parse_square(clicked_square))
            st.session_state.selected = clicked_square if (piece and piece.color == st.session_state.player_color) else None
            st.rerun()

# --- promotion picker -----------------------------------------------------
if pending_promo:
    st.write("Promote to:")
    promo_cols = st.columns(4)
    promo_map = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
    promo_glyphs = {"Q": "♕", "R": "♖", "B": "♗", "N": "♘"} if player_is_white else {"Q": "♛", "R": "♜", "B": "♝", "N": "♞"}
    for i, letter in enumerate(["Q", "R", "B", "N"]):
        if promo_cols[i].button(promo_glyphs[letter], key=f"promo_{letter}", use_container_width=True):
            move = chess.Move(
                chess.parse_square(pending_promo["from"]),
                chess.parse_square(pending_promo["to"]),
                promotion=promo_map[letter],
            )
            board.push(move)
            st.session_state.last_move = move
            st.session_state.pending_promo = None
            st.rerun()

# --- status ---------------------------------------------------------------
if board.is_checkmate():
    winner = "White" if board.turn == chess.BLACK else "Black"
    st.success(f"Checkmate — {winner} wins")
elif board.is_stalemate():
    st.info("Draw by stalemate")
elif board.is_insufficient_material():
    st.info("Draw — insufficient material")
elif board.is_check():
    st.warning("Check!")
else:
    turn_name = "White" if board.turn == chess.WHITE else "Black"
    you_or_model = "Your move — tap a piece" if board.turn == st.session_state.player_color else "Model's move"
    st.write(f"**{turn_name} to move** — {you_or_model}")

st.caption("FEN: " + board.fen())
