"""
Counterpart — model server
===========================
This is the bridge between your enginev1 chess model and the website.

Wired specifically to match chess_engine.ipynb:
  - the enginev1 architecture (3 conv layers + 2 linear layers)
  - board_to_matrix() for encoding a position exactly as it was trained
  - move_to_int / int_to_move for turning class indices into UCI moves

The website POSTs the current position (FEN) to /api/move and expects
back a JSON object with a single legal move in UCI notation, e.g.:
    {"move": "e2e4"}
"""

import json
import random

import numpy as np
import torch
from torch import nn
import chess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ======================================================================
# 0. BEFORE RUNNING THIS: save these two things from your notebook
# ======================================================================
# At the end of chess_engine.ipynb, after training, add and run:
#
#   torch.save(model.state_dict(), "engine.pt")
#
#   import json
#   with open("move_to_int.json", "w") as f:
#       json.dump(move_to_int, f)
#
# Then put both engine.pt and move_to_int.json in this same folder as
# server.py.


# ======================================================================
# 1. MODEL ARCHITECTURE — copied from your notebook so state_dict loads
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
# 2. LOAD move_to_int, THEN THE MODEL (num_classes depends on the dict)
# ======================================================================
with open("move_to_int.json") as f:
    move_to_int = json.load(f)

int_to_move = {v: k for k, v in move_to_int.items()}
num_classes = len(move_to_int)

model = enginev1(num_classes=num_classes)
model.load_state_dict(torch.load("engine.pt", map_location="cpu"))
model.to("cpu")
model.eval()


# ======================================================================
# 3. board_to_matrix — identical encoding to training, do not change
# ======================================================================
def board_to_matrix(board: chess.Board):
    matrix = torch.zeros((13, 8, 8))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color_offset = 0 if piece.color == chess.BLACK else 6
        matrix[piece_color_offset + piece_type, row, col] = 1
        matrix[12, 0, 0] = 1 if board.turn == chess.WHITE else 0
    return matrix


# ======================================================================
# 4. predict_move — same ranked-probability approach as your notebook
# ======================================================================
def predict_move(fen: str, legal_moves: list) -> str:
    board = chess.Board(fen)

    with torch.no_grad():
        x = board_to_matrix(board).unsqueeze(0)  # add batch dimension
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked_indices = np.argsort(probs)[::-1]  # most confident first
    for idx in ranked_indices:
        candidate = int_to_move.get(idx)
        if candidate in legal_moves:
            return candidate

    # Shouldn't happen (there's always at least one legal move covered
    # somewhere in a well-trained vocabulary), but just in case:
    return random.choice(legal_moves)


# ======================================================================
# API routes — you shouldn't need to change anything below this line
# ======================================================================
@app.route("/api/move", methods=["POST"])
def get_move():
    data = request.get_json(force=True, silent=True) or {}
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "fen is required"}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "invalid fen"}), 400

    legal_moves = [m.uci() for m in board.legal_moves]
    if not legal_moves:
        return jsonify({"error": "no legal moves — game is already over"}), 400

    try:
        move = predict_move(fen, legal_moves)
    except Exception as exc:
        print(f"[predict_move] raised an exception, falling back to random: {exc}")
        move = None

    if move not in legal_moves:
        print(f"[predict_move] returned an illegal or missing move ({move!r}), using a random legal move instead")
        move = random.choice(legal_moves)

    return jsonify({"move": move})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
