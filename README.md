# Counterpart — play your own chess model

- **`index.html`** — the website. Open it in a browser once the server below is running.
- **`server.py`** — wired specifically to your `enginev1` model, `board_to_matrix` encoding, and `move_to_int` dictionary from `chess_engine.ipynb`.

## 1. In your notebook, save the two things the server needs

At the end of `chess_engine.ipynb`, after training finishes, add a new cell and run it:

```python
torch.save(model.state_dict(), "engine.pt")

import json
with open("move_to_int.json", "w") as f:
    json.dump(move_to_int, f)
```

Copy both `engine.pt` and `move_to_int.json` into this same folder, next to `server.py`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

(This installs the CPU build of PyTorch by default. If you trained on GPU and want GPU inference too, install the matching CUDA build from pytorch.org instead — CPU is enough for just making moves.)

## 3. Run the server

```bash
python server.py
```

Starts on `http://localhost:5000`. It will load `engine.pt` and `move_to_int.json` from this folder automatically — no code changes needed, since it already matches your notebook's architecture and encoding exactly.

## 4. Open the website

Open `index.html` in your browser. The "Your model" panel already points at `http://localhost:5000/api/move` — click **Test connection**, then **New game**.

## How a move gets picked

Same approach as the `predict_move` cell in your notebook: the board is encoded with `board_to_matrix`, the model outputs a probability over every move in `move_to_int`, and the server walks down that list from most to least confident until it finds one that's actually legal in the current position — then plays it.

## If something looks off

- **"Couldn't reach your model" in the browser** — the server isn't running, or `engine.pt` / `move_to_int.json` aren't in the folder. Check the terminal running `server.py` for the error.
- **Moves seem randomly bad** — check the training was completed with enough epochs/data; the server is doing exactly what your notebook's `predict_move` did, so as-trained quality carries over directly.
- **`size mismatch` error on load** — means `move_to_int.json` doesn't match the `engine.pt` it's paired with (different `num_classes`). Make sure both were saved from the same training run.

## Making it a real website (deployed online)

Right now this only runs on your own computer. To get a public URL anyone can open:

### The app is now a single deployable unit

`server.py` serves both the API **and** the website itself (via `/`), so you only need to deploy one thing — no separate frontend hosting required. The site's "Your model" endpoint already defaults to a relative `/api/move`, so it works automatically wherever it's hosted, with no code changes.

### Recommended: Render (free tier, easiest for a Python + PyTorch app)

1. Push this whole folder (`index.html`, `server.py`, `requirements.txt`, `Procfile`, `engine.pt`, `move_to_int.json`) to a GitHub repo.
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn server:app` (already in the included `Procfile`, Render picks it up automatically)
5. Deploy. Render gives you a URL like `https://your-app.onrender.com` — that's your live website.

Note: Render's free tier spins the server down after inactivity, so the first request after a while takes ~30–60 seconds to wake up.

### Alternatives

- **Railway** (railway.app) — same idea as Render, very similar setup, usage-based free credits.
- **Fly.io** — a bit more setup (Dockerfile-based) but generous free tier and stays warm longer.
- **PythonAnywhere** — good if you want something simpler than Docker/git-based deploys, free tier available.
- **Hugging Face Spaces** (Gradio/Docker SDK) — popular for ML demos specifically, free, but wants a `Dockerfile` or specific app format.

All of these need your model files (`engine.pt`, `move_to_int.json`) included in the repo/deploy — check the file size limits of whichever platform you pick if your model is large.

### A note on cost/performance

This runs your PyTorch model on the server's CPU for every move, which is fine for a small CNN like `enginev1` but means each move request takes a moment. Free tiers are fine for sharing with friends; if you want it always-instantly-on for lots of visitors, a small paid tier on any of the above removes the cold-start delay.
