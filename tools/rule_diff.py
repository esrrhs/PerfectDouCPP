"""Compare C++ legal moves with DouZero's generator on random hands."""
import random
import subprocess
import sys

from douzero.env.game import GameEnv
from douzero.env.move_generator import MovesGener

VALUES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]


class Dummy:
    def act(self, infoset):
        return []


def deal_counts(rng):
    deck = []
    for r in range(13):
        deck.extend([r] * 4)
    deck.extend([13, 14])
    rng.shuffle(deck)
    hand = deck[:20]
    counts = [0] * 15
    for r in hand:
        counts[r] += 1
    return counts


def to_dz(counts):
    cards = []
    for r, n in enumerate(counts):
        cards.extend([VALUES[r]] * n)
    return cards


def key(cards):
    return tuple(sorted(cards))


def dz_legal(hand_counts, rival_counts):
    env = GameEnv(
        {"landlord": Dummy(), "landlord_up": Dummy(), "landlord_down": Dummy()}
    )
    env.acting_player_position = "landlord"
    env.info_sets["landlord"].player_hand_cards = to_dz(hand_counts)
    rival = to_dz(rival_counts)
    env.card_play_action_seq = [] if sum(rival_counts) == 0 else [rival]
    moves = env.get_legal_card_play_actions()
    return {key(m) for m in moves}


def dz_lead(hand_counts):
    mg = MovesGener(to_dz(hand_counts))
    return {key(m) for m in mg.gen_moves()}


def main():
    exe = sys.argv[1]
    rng = random.Random(1)
    cases = []
    lines = ["400"]
    for i in range(200):
        hand = deal_counts(rng)
        cases.append(("L", hand, None))
        lines.append("L " + " ".join(str(x) for x in hand))
    for i in range(200):
        hand = deal_counts(rng)
        rival = deal_counts(rng)
        # A rival play must be a real move. Use one lead move from that hand.
        lead = list(dz_lead(rival))
        if not lead:
            continue
        move = list(rng.choice(lead))
        rc = [0] * 15
        inv = {v: i for i, v in enumerate(VALUES)}
        for v in move:
            rc[inv[v]] += 1
        cases.append(("F", hand, rc))
        lines.append(
            "F "
            + " ".join(str(x) for x in hand)
            + " "
            + " ".join(str(x) for x in rc)
        )
    lines[0] = str(len(cases))
    proc = subprocess.run(
        [exe], input="\n".join(lines) + "\n", text=True, capture_output=True
    )
    if proc.returncode != 0:
        print(proc.stderr)
        sys.exit(proc.returncode)
    out = proc.stdout.splitlines()
    pos = 0
    mismatches = 0
    max_n = 0
    for kind, hand, rival in cases:
        n = int(out[pos])
        pos += 1
        got = set()
        for _ in range(n):
            row = out[pos]
            pos += 1
            if row == "":
                got.add(tuple())
            else:
                got.add(tuple(int(x) for x in row.split(",")))
        max_n = max(max_n, n)
        if kind == "L":
            expect = dz_lead(hand)
        else:
            expect = dz_legal(hand, rival)
        if got != expect:
            mismatches += 1
            if mismatches <= 8:
                print("mismatch", kind)
                print(" only cpp", sorted(got - expect)[:6])
                print(" only dz ", sorted(expect - got)[:6])
    print(f"cases={len(cases)} mismatches={mismatches} max_moves={max_n}")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
