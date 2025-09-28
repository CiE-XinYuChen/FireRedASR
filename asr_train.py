#!/usr/bin/env python3

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="ASR training entry (AED supported)")
    parser.add_argument("--asr_type", type=str, default="aed", choices=["aed"],
                        help="ASR type to train. Currently only 'aed' is supported.")
    # parse only the high-level option and forward the rest
    args, remaining = parser.parse_known_args()

    if args.asr_type == "aed":
        from fireredasr.train import train_aed as train_mod
        # Forward remaining args to the AED trainer's CLI
        sys.argv = [sys.argv[0]] + remaining
        train_mod.main()
    else:
        raise NotImplementedError("Only AED training is available in this repo.")


if __name__ == "__main__":
    main()

