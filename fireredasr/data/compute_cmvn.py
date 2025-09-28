#!/usr/bin/env python3

import argparse
import os
import numpy as np
import kaldiio

from fireredasr.data.dataset_kaldi import read_wav_scp
from fireredasr.data.asr_feat import KaldifeatFbank


parser = argparse.ArgumentParser()
parser.add_argument("--wav_scp", type=str, required=True,
                    help="Kaldi-style wav.scp: <utt> <wav_path>")
parser.add_argument("--out_cmvn", type=str, required=True,
                    help="Output path of Kaldi CMVN stats (e.g., cmvn.ark)")
parser.add_argument("--num_mel_bins", type=int, default=80)
parser.add_argument("--frame_length", type=int, default=25)
parser.add_argument("--frame_shift", type=int, default=10)


def main(args):
    assert os.path.exists(args.wav_scp), f"Not found: {args.wav_scp}"
    wavs = read_wav_scp(args.wav_scp)
    fbank = KaldifeatFbank(num_mel_bins=args.num_mel_bins,
                           frame_length=args.frame_length,
                           frame_shift=args.frame_shift,
                           dither=0.0)

    feat_dim = args.num_mel_bins
    sum_vec = np.zeros((feat_dim,), dtype=np.float64)
    sum_sq_vec = np.zeros((feat_dim,), dtype=np.float64)
    frame_count = 0.0

    for i, (utt, wav) in enumerate(wavs.items()):
        feat = fbank(wav)
        if feat.shape[0] == 0:
            continue
        sum_vec += feat.sum(axis=0)
        sum_sq_vec += (feat * feat).sum(axis=0)
        frame_count += feat.shape[0]
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1} / {len(wavs)} wavs")

    assert frame_count > 0, "No frames accumulated; check wav.scp"
    # Kaldi CMVN stats: 2 x (dim + 1), last column stores frame_count
    stats = np.zeros((2, feat_dim + 1), dtype=np.float64)
    stats[0, :feat_dim] = sum_vec
    stats[0, feat_dim] = frame_count
    stats[1, :feat_dim] = sum_sq_vec
    stats[1, feat_dim] = 0.0

    os.makedirs(os.path.dirname(args.out_cmvn), exist_ok=True)
    kaldiio.save_mat(args.out_cmvn, stats)
    print(f"Saved CMVN to {args.out_cmvn}. Frames={frame_count:.0f}")


if __name__ == "__main__":
    main(parser.parse_args())

