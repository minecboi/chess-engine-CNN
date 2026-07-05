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
import chess.svg
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


def new_game(player_color):
    st.session_state.board = chess.Board()
    st.session_state.player_color = player_color
    st.session_state.last_move = None
    st.session_state.error_msg = None


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
# MAIN — board + status + move input
# ======================================================================
st.title("♞ Counterpart")
st.caption("A board for you and the model you trained.")

svg = chess.svg.board(
    board,
    orientation=st.session_state.player_color,
    lastmove=st.session_state.last_move,
    size=430,
)
st.markdown(f'<div style="display:flex;justify-content:center">{svg}</div>', unsafe_allow_html=True)

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
    you_or_model = "Your move" if board.turn == st.session_state.player_color else "Model's move"
    st.write(f"**{turn_name} to move** — {you_or_model}")

if st.session_state.error_msg:
    st.error(st.session_state.error_msg)
    st.session_state.error_msg = None

if not board.is_game_over() and board.turn == st.session_state.player_color:
    with st.form("move_form", clear_on_submit=True):
        move_input = st.text_input(
            "Your move (UCI notation, e.g. e2e4 — add a letter for promotion, e.g. e7e8q)"
        )
        submitted = st.form_submit_button("Play move")
        if submitted and move_input:
            try:
                move = chess.Move.from_uci(move_input.strip().lower())
                if move in board.legal_moves:
                    board.push(move)
                    st.session_state.last_move = move
                    st.rerun()
                else:
                    st.session_state.error_msg = f"'{move_input}' isn't a legal move in this position."
                    st.rerun()
            except ValueError:
                st.session_state.error_msg = f"Couldn't parse '{move_input}' — use UCI like e2e4."
                st.rerun()

st.caption("FEN: " + board.fen())
