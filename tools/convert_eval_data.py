#!/usr/bin/env python3
"""Convert DouZero eval_data.pkl into a compact binary format for C++ fast loading.

Format:
Header (8 bytes):
  char magic[4] = "EVAL"
  uint32_t num_decks
Per Deck (57 bytes):
  uint8_t landlord[20]       (rank index 0..14)
  uint8_t landlord_down[17]  (rank index 0..14)
  uint8_t landlord_up[17]    (rank index 0..14)
  uint8_t bottom[3]          (rank index 0..14)
"""

import argparse
import os
import pickle
import struct
import sys

# DouZero card value -> PerfectDouCPP rank index (0..14)
# 3..14 -> 0..11, 17(2) -> 12, 20(BlackJoker) -> 13, 30(RedJoker) -> 14
def val_to_rank(v):
    if v == 17:
        return 12
    elif v == 20:
        return 13
    elif v == 30:
        return 14
    else:
        return v - 3


def convert(pkl_path, bin_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    num_decks = len(data)
    with open(bin_path, "wb") as f:
        f.write(b"EVAL")
        f.write(struct.pack("<I", num_decks))

        for item in data:
            l_cards = bytes(val_to_rank(v) for v in item["landlord"])
            d_cards = bytes(val_to_rank(v) for v in item["landlord_down"])
            u_cards = bytes(val_to_rank(v) for v in item["landlord_up"])
            b_cards = bytes(val_to_rank(v) for v in item["three_landlord_cards"])
            assert len(l_cards) == 20, f"Expected 20 landlord cards, got {len(l_cards)}"
            assert len(d_cards) == 17, f"Expected 17 down cards, got {len(d_cards)}"
            assert len(u_cards) == 17, f"Expected 17 up cards, got {len(u_cards)}"
            assert len(b_cards) == 3, f"Expected 3 bottom cards, got {len(b_cards)}"
            f.write(l_cards)
            f.write(d_cards)
            f.write(u_cards)
            f.write(b_cards)

    print(f"Successfully converted {num_decks} decks from {pkl_path} to {bin_path} ({os.path.getsize(bin_path)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DouZero eval_data.pkl to binary format")
    parser.add_argument("input", nargs="?", default="data/eval_data.pkl", help="Input .pkl file path")
    parser.add_argument("output", nargs="?", default="", help="Output .bin file path (optional)")
    args = parser.parse_args()

    out_path = args.output
    if not out_path:
        if args.input.endswith(".pkl"):
            out_path = args.input[:-4] + ".bin"
        else:
            out_path = args.input + ".bin"

    convert(args.input, out_path)
