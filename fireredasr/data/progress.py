#!/usr/bin/env python3

"""
LibriSpeech preprocessing helper.

Features:
- Optional: Convert FLAC to WAV (16kHz, mono, 16-bit) via ffmpeg
- Generate Kaldi-style manifests: wav.scp, text (uppercase)
- Align wav.scp/text by uttid (keep intersection)
- Aggregate train texts and train SentencePiece model
- Build dict.txt from SPM model (prepend <pad>, <unk>, <sos>, <eos>)
- Compute CMVN stats (Kaldi format cmvn.ark) on training wav.scp

Usage example:
  python fireredasr/data/progress.py \
    --librispeech_root /data/LibriSpeech \
    --out_dir data/libri \
    --convert_to_wav 1 --ffmpeg ffmpeg \
    --spm_vocab_size 1000
"""

import argparse
import os
import re
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sentencepiece as spm
import kaldiio

from fireredasr.data.asr_feat import KaldifeatFbank


def run_cmd(cmd: List[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def find_files(root: str, pattern: str) -> List[str]:
    return glob(str(Path(root) / "**" / pattern), recursive=True)


def convert_flac_to_wav(root: str, ffmpeg: str = "ffmpeg", sr: int = 16000):
    flacs = find_files(root, "*.flac")
    print(f"[convert] found {len(flacs)} flac files under {root}")
    for i, f in enumerate(flacs, 1):
        wav = f[:-5] + ".wav"
        out_dir = os.path.dirname(wav)
        os.makedirs(out_dir, exist_ok=True)
        # -y overwrite, 16kHz mono 16-bit PCM
        run_cmd([ffmpeg, "-loglevel", "error", "-y", "-i", f,
                 "-ar", str(sr), "-ac", "1", "-acodec", "pcm_s16le",
                 "-f", "wav", wav])
        if i % 500 == 0:
            print(f"[convert] processed {i}/{len(flacs)}")


def collect_wavs(root: str, splits: List[str]) -> Dict[str, str]:
    utt2wav: Dict[str, str] = {}
    for sp in splits:
        wavs = find_files(os.path.join(root, sp), "*.wav")
        for w in wavs:
            uid = os.path.basename(w)[:-4]
            if uid in utt2wav and utt2wav[uid] != w:
                print(f"[warn] duplicate uttid {uid}, keep first: {utt2wav[uid]} vs {w}")
                continue
            utt2wav[uid] = os.path.abspath(w)
    return utt2wav


def collect_texts(root: str, splits: List[str], uppercase: bool = True) -> Dict[str, str]:
    utt2txt: Dict[str, str] = {}
    for sp in splits:
        trans_files = find_files(os.path.join(root, sp), "*.trans.txt")
        for tf in trans_files:
            with open(tf, "r", encoding="utf8") as fin:
                for line in fin:
                    cols = line.strip().split()
                    if not cols:
                        continue
                    utt = cols[0]
                    txt = "" if len(cols) == 1 else " ".join(cols[1:])
                    if uppercase:
                        txt = txt.upper()
                    utt2txt[utt] = txt
    return utt2txt


def write_manifest(out_dir: str, name: str, utt2wav: Dict[str, str], utt2txt: Dict[str, str]):
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    wav_scp = os.path.join(out_dir, name, "wav.scp")
    text = os.path.join(out_dir, name, "text")

    ids = sorted(set(utt2wav.keys()) & set(utt2txt.keys()))
    miss_wav = sorted(set(utt2txt.keys()) - set(utt2wav.keys()))
    miss_txt = sorted(set(utt2wav.keys()) - set(utt2txt.keys()))
    if miss_wav:
        print(f"[warn] {name}: {len(miss_wav)} ids in text but missing wav")
    if miss_txt:
        print(f"[warn] {name}: {len(miss_txt)} ids in wav but missing text")
    print(f"[manifest] {name}: kept {len(ids)} utts after alignment")

    with open(wav_scp, "w", encoding="utf8") as fw, \
         open(text, "w", encoding="utf8") as ft:
        for uid in ids:
            fw.write(f"{uid} {utt2wav[uid]}\n")
            ft.write(f"{uid} {utt2txt[uid]}\n")
    return wav_scp, text


def build_train_text(text_path: str, out_path: str):
    with open(text_path, "r", encoding="utf8") as fin, \
         open(out_path, "w", encoding="utf8") as fout:
        for line in fin:
            cols = line.strip().split()
            if not cols:
                continue
            fout.write("" if len(cols) == 1 else " ".join(cols[1:]))
            fout.write("\n")


def train_spm(input_text: str, model_prefix: str, vocab_size: int,
              model_type: str = "bpe", character_coverage: float = 1.0):
    # Ensure uppercase input is used (project tokenizer uppercases internally)
    norm_path = input_text + ".upper"
    with open(input_text, "r", encoding="utf8") as fin, \
         open(norm_path, "w", encoding="utf8") as fout:
        for line in fin:
            fout.write(line.strip().upper() + "\n")
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    cmd = (
        f"--input={norm_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--model_type={model_type}"
    )
    spm.SentencePieceTrainer.Train(cmd)
    print(f"[spm] trained {model_prefix}.model")


def build_dict_from_spm(spm_model: str, out_dict: str,
                        special_tokens: List[str] = ["<pad>", "<unk>", "<sos>", "<eos>"]):
    os.makedirs(os.path.dirname(out_dict), exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    with open(out_dict, "w", encoding="utf8") as f:
        idx = 0
        for t in special_tokens:
            f.write(f"{t} {idx}\n"); idx += 1
        for p in pieces:
            if p in special_tokens:
                continue
            f.write(f"{p} {idx}\n"); idx += 1
    print(f"[dict] wrote {out_dict}")


def compute_cmvn_kaldi(wav_scp: str, out_cmvn: str, num_mel_bins=80, frame_length=25, frame_shift=10):
    from fireredasr.data.dataset_kaldi import read_wav_scp
    utt2wav = read_wav_scp(wav_scp)
    fbank = KaldifeatFbank(num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           dither=0.0)
    feat_dim = num_mel_bins
    sum_vec = np.zeros((feat_dim,), dtype=np.float64)
    sum_sq_vec = np.zeros((feat_dim,), dtype=np.float64)
    frame_count = 0.0
    for i, (utt, wav) in enumerate(utt2wav.items(), 1):
        feat = fbank(wav)
        if feat.shape[0] == 0:
            continue
        sum_vec += feat.sum(axis=0)
        sum_sq_vec += (feat * feat).sum(axis=0)
        frame_count += feat.shape[0]
        if i % 500 == 0:
            print(f"[cmvn] processed {i}/{len(utt2wav)} wavs")
    assert frame_count > 0, "No frames for CMVN; check wav.scp"
    stats = np.zeros((2, feat_dim + 1), dtype=np.float64)
    stats[0, :feat_dim] = sum_vec
    stats[0, feat_dim] = frame_count
    stats[1, :feat_dim] = sum_sq_vec
    stats[1, feat_dim] = 0.0
    os.makedirs(os.path.dirname(out_cmvn), exist_ok=True)
    kaldiio.save_mat(out_cmvn, stats)
    print(f"[cmvn] saved to {out_cmvn}, frames={frame_count:.0f}")


def parse_splits(x: str) -> List[str]:
    return [s for s in re.split(r"[ ,]+", x.strip()) if s]


def main():
    ap = argparse.ArgumentParser(description="LibriSpeech preprocessing pipeline")
    ap.add_argument("--librispeech_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/libri")
    ap.add_argument("--train_splits", type=str, default="train-clean-100,train-clean-360,train-other-500")
    ap.add_argument("--dev_splits", type=str, default="dev-clean,dev-other")
    ap.add_argument("--convert_to_wav", type=int, default=1)
    ap.add_argument("--ffmpeg", type=str, default="ffmpeg")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--uppercase", type=int, default=1)

    # SPM / dict
    ap.add_argument("--spm_vocab_size", type=int, default=1000)
    ap.add_argument("--spm_model_type", type=str, default="bpe", choices=["bpe", "unigram", "char", "word"])
    ap.add_argument("--character_coverage", type=float, default=1.0)
    ap.add_argument("--no_spm", action="store_true")

    # CMVN
    ap.add_argument("--no_cmvn", action="store_true")
    ap.add_argument("--num_mel_bins", type=int, default=80)
    ap.add_argument("--frame_length", type=int, default=25)
    ap.add_argument("--frame_shift", type=int, default=10)

    args = ap.parse_args()

    train_splits = parse_splits(args.train_splits)
    dev_splits = parse_splits(args.dev_splits)

    # Step 1: Convert FLAC -> WAV
    if args.convert_to_wav:
        convert_flac_to_wav(args.librispeech_root, ffmpeg=args.ffmpeg, sr=args.sample_rate)
    else:
        print("[convert] skipped")

    # Step 2: Build wav.scp/text for train & dev
    tr_wav = collect_wavs(args.librispeech_root, train_splits)
    tr_txt = collect_texts(args.librispeech_root, train_splits, uppercase=bool(args.uppercase))
    de_wav = collect_wavs(args.librispeech_root, dev_splits)
    de_txt = collect_texts(args.librispeech_root, dev_splits, uppercase=bool(args.uppercase))

    tr_wav_scp, tr_text = write_manifest(args.out_dir, "train", tr_wav, tr_txt)
    de_wav_scp, de_text = write_manifest(args.out_dir, "dev", de_wav, de_txt)

    # Step 3: Aggregate train text for SPM
    tr_text_all = os.path.join(args.out_dir, "train_text.txt")
    build_train_text(tr_text, tr_text_all)

    # Step 4: Train SPM + build dict
    spm_prefix = os.path.join(args.out_dir, f"spm_{args.spm_model_type}{args.spm_vocab_size}")
    if not args.no_spm:
        train_spm(tr_text_all, spm_prefix, args.spm_vocab_size, args.spm_model_type, args.character_coverage)
        dict_path = os.path.join(args.out_dir, "dict.txt")
        build_dict_from_spm(spm_prefix + ".model", dict_path)
    else:
        print("[spm] skipped")

    # Step 5: Compute CMVN on training set
    if not args.no_cmvn:
        out_cmvn = os.path.join(args.out_dir, "cmvn.ark")
        compute_cmvn_kaldi(tr_wav_scp, out_cmvn, args.num_mel_bins, args.frame_length, args.frame_shift)
    else:
        print("[cmvn] skipped")

    print("[done] preprocessing completed")


if __name__ == "__main__":
    main()

