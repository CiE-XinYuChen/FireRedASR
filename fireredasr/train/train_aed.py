#!/usr/bin/env python3

import argparse
import os
import time
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fireredasr.data.dataset_kaldi import load_kaldi_manifest
from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer


def pad_list(xs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs]) if n_batch > 0 else 0
    if n_batch == 0:
        return torch.empty(0)
    pad = xs[0].new_full([n_batch, max_len], pad_value)
    for i, x in enumerate(xs):
        pad[i, : x.size(0)] = x
    return pad


def build_model_args(odim: int, idim: int, pad_id: int, sos_id: int, eos_id: int,
                     n_layers_enc=24, n_layers_dec=12, n_head=16, d_model=1024,
                     residual_dropout=0.1, dropout_rate=0.1, kernel_size=33,
                     pe_maxlen=5000):
    args = SimpleNamespace()
    # decoder special ids
    args.sos_id = sos_id
    args.eos_id = eos_id
    args.pad_id = pad_id
    # dims
    args.idim = idim
    args.odim = odim
    args.n_layers_enc = n_layers_enc
    args.n_layers_dec = n_layers_dec
    args.n_head = n_head
    args.d_model = d_model
    args.residual_dropout = residual_dropout
    args.dropout_rate = dropout_rate
    args.kernel_size = kernel_size
    args.pe_maxlen = pe_maxlen
    return args


def forward_teacher_forcing(model: FireRedAsrAed,
                            feats: torch.Tensor,
                            feat_lens: torch.Tensor,
                            ys_in_pad: torch.Tensor,
                            ys_out_pad: torch.Tensor,
                            pad_id: int) -> torch.Tensor:
    enc_out, enc_lens, enc_mask = model.encoder(feats, feat_lens)
    # Build trg mask
    tgt_mask = model.decoder.ignored_target_position_is_0(ys_in_pad, pad_id)
    # Embed + pe
    dec_input = model.decoder.tgt_word_emb(ys_in_pad) * model.decoder.scale 
    dec_input = dec_input + model.decoder.positional_encoding(ys_in_pad)
    dec_input = model.decoder.dropout(dec_input)

    x = dec_input
    for dec_layer in model.decoder.layer_stack:
        x = dec_layer.forward(x, enc_out, tgt_mask, enc_mask, cache=None)
    x = model.decoder.layer_norm_out(x)
    logits = model.decoder.tgt_word_prj(x)  # (N, L, V)
    # Return CE loss computed outside
    return logits


def tokens_to_ys(texts: List[str], tokenizer: ChineseCharEnglishSpmTokenizer,
                 sos_id: int, eos_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ys_in, ys_out = [], []
    for t in texts:
        _, token_ids = tokenizer.tokenize(t)
        inp = torch.tensor([sos_id] + token_ids, dtype=torch.long)
        out = torch.tensor(token_ids + [eos_id], dtype=torch.long)
        ys_in.append(inp)
        ys_out.append(out)
    return pad_list(ys_in, tokenizer.dict.get("<pad>", 0)), \
           pad_list(ys_out, tokenizer.dict.get("<pad>", 0))


def run_epoch(model, optimizer, criterion, feat_extractor,
              batch_items: List[Tuple[str, str, str]],
              tokenizer, pad_id, sos_id, eos_id,
              device, batch_size=8, train=True):
    model.train(mode=train)
    total_loss = 0.0
    total_tok = 0
    start = 0
    while start < len(batch_items):
        batch = batch_items[start:start + batch_size]
        start += batch_size
        uttids = [x[0] for x in batch]
        wavs = [x[1] for x in batch]
        texts = [x[2] for x in batch]

        feats, feat_lens, _ = feat_extractor(wavs)
        ys_in_pad, ys_out_pad = tokens_to_ys(texts, tokenizer, sos_id, eos_id)

        feats = feats.to(device)
        feat_lens = feat_lens.to(device)
        ys_in_pad = ys_in_pad.to(device)
        ys_out_pad = ys_out_pad.to(device)

        with torch.set_grad_enabled(train):
            logits = forward_teacher_forcing(model, feats, feat_lens, ys_in_pad, ys_out_pad, pad_id)
            # Flatten for CE
            N, L, V = logits.size()
            loss = criterion(logits.view(N*L, V), ys_out_pad.view(N*L))
            # token count (exclude pads)
            n_tok = int((ys_out_pad != pad_id).sum().item())

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.detach().item()
        total_tok += max(n_tok, 1)

    return total_loss / max(total_tok, 1)


def save_checkpoint(save_dir: str, model: FireRedAsrAed, args_ns: SimpleNamespace):
    os.makedirs(save_dir, exist_ok=True)
    pkg = {
        "args": args_ns,
        "model_state_dict": model.state_dict(),
    }
    out = os.path.join(save_dir, "model.pth.tar")
    torch.save(pkg, out)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    # Data manifests
    parser.add_argument("--train_wav_scp", type=str, required=True)
    parser.add_argument("--train_text", type=str, required=True)
    parser.add_argument("--valid_wav_scp", type=str, required=True)
    parser.add_argument("--valid_text", type=str, required=True)
    parser.add_argument("--cmvn", type=str, required=True, help="Kaldi CMVN stats (computed by compute_cmvn.py)")
    # Tokenizer / vocab
    parser.add_argument("--dict", dest="dict_path", type=str, required=True)
    parser.add_argument("--spm_model", type=str, required=True)
    parser.add_argument("--pad_token", type=str, default="<pad>")
    parser.add_argument("--unk_token", type=str, default="<unk>")
    parser.add_argument("--sos_token", type=str, default="<sos>")
    parser.add_argument("--eos_token", type=str, default="<eos>")
    # Model
    parser.add_argument("--idim", type=int, default=80)
    parser.add_argument("--n_layers_enc", type=int, default=24)
    parser.add_argument("--n_layers_dec", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--residual_dropout", type=float, default=0.1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--kernel_size", type=int, default=33)
    parser.add_argument("--pe_maxlen", type=int, default=5000)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="exp/aed_librispeech")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load data items
    train_items = load_kaldi_manifest(args.train_wav_scp, args.train_text)
    valid_items = load_kaldi_manifest(args.valid_wav_scp, args.valid_text)
    print(f"#train={len(train_items)} #valid={len(valid_items)}")

    # Tokenizer
    tokenizer = ChineseCharEnglishSpmTokenizer(args.dict_path, args.spm_model, unk=args.unk_token)
    odim = len(tokenizer.dict)
    pad_id = tokenizer.dict.get(args.pad_token, 0)
    sos_id = tokenizer.dict.get(args.sos_token, 2)
    eos_id = tokenizer.dict.get(args.eos_token, 3)

    # Model
    model_args = build_model_args(odim=odim, idim=args.idim,
                                  pad_id=pad_id, sos_id=sos_id, eos_id=eos_id,
                                  n_layers_enc=args.n_layers_enc,
                                  n_layers_dec=args.n_layers_dec,
                                  n_head=args.n_head, d_model=args.d_model,
                                  residual_dropout=args.residual_dropout,
                                  dropout_rate=args.dropout_rate,
                                  kernel_size=args.kernel_size,
                                  pe_maxlen=args.pe_maxlen)
    model = FireRedAsrAed.from_args(model_args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    # Feature extractor
    feat_extractor = ASRFeatExtractor(args.cmvn)

    best_valid = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, optimizer, criterion, feat_extractor,
                            train_items, tokenizer, pad_id, sos_id, eos_id,
                            args.device, batch_size=args.batch_size, train=True)
        va_loss = run_epoch(model, optimizer, criterion, feat_extractor,
                            valid_items, tokenizer, pad_id, sos_id, eos_id,
                            args.device, batch_size=args.batch_size, train=False)
        dt = time.time() - t0
        print(f"Epoch {ep}: train_loss/token={tr_loss:.4f} valid_loss/token={va_loss:.4f} time={dt:.1f}s")

        # Save latest
        save_checkpoint(args.save_dir, model, model_args)
        # Track best
        if va_loss < best_valid:
            best_valid = va_loss
            save_checkpoint(os.path.join(args.save_dir, "best"), model, model_args)


if __name__ == "__main__":
    main()

