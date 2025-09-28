#!/usr/bin/env python3

import argparse
import os
import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, required=True,
                    help="Path to a plain text file with one line per utterance text")
parser.add_argument("--model_prefix", type=str, required=True,
                    help="Output prefix for SPM model (e.g., data/spm_bpe1000)")
parser.add_argument("--vocab_size", type=int, default=1000)
parser.add_argument("--character_coverage", type=float, default=1.0)
parser.add_argument("--model_type", type=str, default="bpe",
                    choices=["bpe", "unigram", "char", "word"])


def main(args):
    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)
    # Normalize to uppercase to match project tokenizer behavior
    # (ChineseCharEnglishSpmTokenizer uppercases input before SPM)
    norm_path = args.input_text + ".upper"
    with open(args.input_text, "r", encoding="utf8") as fin, \
         open(norm_path, "w", encoding="utf8") as fout:
        for line in fin:
            fout.write(line.strip().upper() + "\n")

    cmd = (
        f"--input={norm_path} "
        f"--model_prefix={args.model_prefix} "
        f"--vocab_size={args.vocab_size} "
        f"--character_coverage={args.character_coverage} "
        f"--model_type={args.model_type}"
    )
    spm.SentencePieceTrainer.Train(cmd)
    print(f"Trained SPM at {args.model_prefix}.model")


if __name__ == "__main__":
    main(parser.parse_args())

