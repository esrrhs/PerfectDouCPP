# Serves the public DouZero-ADP agents over one TCP connection.
# One request line in, one action line out. Pass is "-".
# CPU only: the trainer already owns the GPU.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Fields, tab-separated:
#   pos hand other last bomb nums played lastdict seq legal
# Card groups use comma-separated DouZero values. "-" is empty.
# played / lastdict are landlord;landlord_down;landlord_up.
# seq / legal join actions with "/".
import argparse
import importlib.util
import socket
from pathlib import Path

import numpy as np
import torch
from douzero.env.env import get_obs


def cards(tok):
    if tok is None or tok == "" or tok == "-":
        return []
    return [int(x) for x in tok.split(",") if x != ""]


def groups(tok):
    return [cards(p) for p in tok.split("/")]


def triples(tok):
    parts = tok.split(";")
    while len(parts) < 3:
        parts.append("-")
    return {
        "landlord": cards(parts[0]),
        "landlord_down": cards(parts[1]),
        "landlord_up": cards(parts[2]),
    }


def nums(tok):
    a = [int(x) for x in tok.split()]
    return {
        "landlord": a[0],
        "landlord_down": a[1],
        "landlord_up": a[2],
    }


class InfoSet:
    pass


def build(line):
    f = line.split("\t")
    if len(f) != 10:
        raise ValueError("expected 10 fields, got %d" % len(f))
    info = InfoSet()
    info.player_position = f[0]
    info.player_hand_cards = cards(f[1])
    info.other_hand_cards = cards(f[2])
    info.last_move = cards(f[3])
    info.bomb_num = int(f[4])
    info.num_cards_left_dict = nums(f[5])
    info.played_cards = triples(f[6])
    info.last_move_dict = triples(f[7])
    info.card_play_action_seq = groups(f[8]) if f[8] != "" else []
    info.legal_actions = groups(f[9])
    return info


def encode(action):
    if not action:
        return "-"
    return ",".join(str(c) for c in action)


def load_model(pos, path):
    # Import the model file directly. douzero.dmc pulls in GitPython, which
    # this eval does not need.
    root = Path(get_obs.__code__.co_filename).resolve().parents[1] / "dmc" / "models.py"
    spec = importlib.util.spec_from_file_location("dz_models", root)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    model = models.model_dict[pos]()
    pretrained = torch.load(path, map_location="cpu", weights_only=False)
    state = model.state_dict()
    state.update({k: v for k, v in pretrained.items() if k in state})
    model.load_state_dict(state)
    model.eval()
    return model


def choose(model, infoset):
    if len(infoset.legal_actions) == 1:
        return infoset.legal_actions[0]
    obs = get_obs(infoset)
    with torch.inference_mode():
        z = torch.from_numpy(obs["z_batch"]).float()
        x = torch.from_numpy(obs["x_batch"]).float()
        values = model.forward(z, x, return_value=True)["values"].detach().cpu().numpy()
    return infoset.legal_actions[int(np.argmax(values, axis=0)[0])]


def serve(port, ckpt_dir):
    agents = {
        pos: load_model(pos, ckpt_dir + "/" + pos + ".ckpt")
        for pos in ("landlord", "landlord_down", "landlord_up")
    }
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    print("ready", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            buf = b""
            try:
                while True:
                    while b"\n" not in buf:
                        chunk = conn.recv(1 << 20)
                        if not chunk:
                            break
                        buf += chunk
                    if not buf:
                        break
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode().strip()
                    if text == "" or text == "QUIT":
                        break
                    action = choose(agents[text.split("\t", 1)[0]], build(text))
                    conn.sendall((encode(action) + "\n").encode())
            except Exception as e:
                pass
            finally:
                conn.close()
    finally:
        srv.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--ckpt-dir", required=True)
    a = p.parse_args()
    serve(a.port, a.ckpt_dir)
