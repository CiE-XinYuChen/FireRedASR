import os
from typing import List, Tuple, Dict


def read_wav_scp(wav_scp: str) -> Dict[str, str]:
    """Read a Kaldi-style wav.scp: each line: <uttid> <path-to-wav>

    Returns:
        dict: uttid -> wav_path
    """
    utt2path: Dict[str, str] = {}
    with open(wav_scp, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            cols = line.strip().split(maxsplit=1)
            if len(cols) < 2:
                # Allow empty text (skip) or malformed line
                continue
            uttid, path = cols[0], cols[1]
            utt2path[uttid] = path
    return utt2path


def read_text(text_file: str) -> Dict[str, str]:
    """Read a Kaldi-style text: each line: <uttid> <transcript...>

    Returns:
        dict: uttid -> text (possibly empty string)
    """
    utt2text: Dict[str, str] = {}
    with open(text_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            cols = line.strip().split()
            if not cols:
                continue
            uttid = cols[0]
            txt = "" if len(cols) == 1 else " ".join(cols[1:])
            utt2text[uttid] = txt
    return utt2text


def load_kaldi_manifest(wav_scp: str, text_file: str) -> List[Tuple[str, str, str]]:
    """Create a list of (uttid, wav_path, text) by joining wav.scp and text.

    Only utterances present in both are returned.
    """
    utt2path = read_wav_scp(wav_scp)
    utt2text = read_text(text_file)
    items: List[Tuple[str, str, str]] = []
    for uttid, path in utt2path.items():
        if uttid in utt2text:
            items.append((uttid, path, utt2text[uttid]))
    return items

