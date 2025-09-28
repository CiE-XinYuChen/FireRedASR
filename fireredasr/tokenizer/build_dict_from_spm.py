#!/usr/bin/env python3

import argparse
import os
import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument("--spm_model", type=str, required=True,
                    help="Path to sentencepiece model (.model)")
parser.add_argument("--out_dict", type=str, required=True,
                    help="Output dict.txt file with '<token> <id>' per line")
parser.add_argument("--special_tokens", type=str, nargs="*", default=["<pad>", "<unk>", "<sos>", "<eos>"],
                    help="Special tokens to prepend in order")


def main(args):
    os.makedirs(os.path.dirname(args.out_dict), exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    # Build deterministic mapping: specials first, then all SPM pieces
    with open(args.out_dict, "w", encoding="utf8") as f:
        idx = 0
        for t in args.special_tokens:
            f.write(f"{t} {idx}\n")
            idx += 1
        for p in pieces:
            if p in args.special_tokens:
                continue
            f.write(f"{p} {idx}\n")
            idx += 1
    print(f"Wrote dict to {args.out_dict}. size={idx}")


if __name__ == "__main__":
    main(parser.parse_args())

