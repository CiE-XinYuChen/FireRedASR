**FireRedASR AED Training (LibriSpeech)**

This guide shows how to train the AED model on LibriSpeech using the new training scripts in this repo. It focuses on the AED variant and a Kaldi-style manifest to keep dependencies minimal.

---

**1. Environment**
- Python 3.10
- Install deps:
  - `pip install -r requirements.txt`
- Ensure `ffmpeg` is available in PATH (audio conversion).

---

**2. Data Preparation (LibriSpeech)**
- Download LibriSpeech (train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other) from OpenSLR.
- Convert FLAC to 16kHz/16-bit PCM WAV:
  - Example bash (adjust paths):
    - `find /path/LibriSpeech -name "*.flac" | while read f; do \
         out=${f%.flac}.wav; \
         mkdir -p "$(dirname "$out")"; \
         ffmpeg -y -i "$f" -ar 16000 -ac 1 -acodec pcm_s16le -f wav "$out"; \
       done`

- Create Kaldi-style manifests for each split:
  - wav.scp: lines in the form `<uttid> <abs_path_to_wav>`
  - text: lines in the form `<uttid> <uppercase_transcript>`
  - Tip: `<uttid>` usually matches the LibriSpeech file stem, e.g. `84-121123-0001`.

Suggest layout:
- `data/libri/train/wav.scp`
- `data/libri/train/text`
- `data/libri/dev/wav.scp`
- `data/libri/dev/text`

You can generate these with your own scripting or external tools; ensure transcripts are uppercased (the tokenizer uppercases before SPM tokenization).

---

**3. Train SPM + Build Dict**
- Aggregate training transcripts into one file `data/libri/train_text.txt`:
  - `cut -d ' ' -f 2- data/libri/train/text > data/libri/train_text.txt`

- Train SPM (BPE-1000 as example):
  - `python fireredasr/tokenizer/train_spm.py \
      --input_text data/libri/train_text.txt \
      --model_prefix data/libri/spm_bpe1000 \
      --vocab_size 1000 --character_coverage 1.0 --model_type bpe`

- Build `dict.txt` from SPM model:
  - `python fireredasr/tokenizer/build_dict_from_spm.py \
      --spm_model data/libri/spm_bpe1000.model \
      --out_dict data/libri/dict.txt`

This produces:
- `data/libri/spm_bpe1000.model`
- `data/libri/dict.txt` (contains <pad>, <unk>, <sos>, <eos> + SPM pieces)

---

**4. Compute CMVN**
- Use the provided script to compute Kaldi-style CMVN stats on training audio:
  - `python fireredasr/data/compute_cmvn.py \
      --wav_scp data/libri/train/wav.scp \
      --out_cmvn data/libri/cmvn.ark`

---

**5. Train AED**
- Minimal single-GPU run (adjust batch/epochs/model size to your machine):
  - `python fireredasr/train/train_aed.py \
      --train_wav_scp data/libri/train/wav.scp \
      --train_text    data/libri/train/text \
      --valid_wav_scp data/libri/dev/wav.scp \
      --valid_text    data/libri/dev/text \
      --cmvn          data/libri/cmvn.ark \
      --dict          data/libri/dict.txt \
      --spm_model     data/libri/spm_bpe1000.model \
      --batch_size 8 --epochs 10 --lr 2e-4 \
      --save_dir exp/aed_librispeech`

Outputs:
- Checkpoints at `exp/aed_librispeech/model.pth.tar` and `exp/aed_librispeech/best/model.pth.tar`

---

**6. Package for Inference**
To use `speech2text.py`, gather these into a model dir:
- `pretrained_models/FireRedASR-AED-L/`
  - `model.pth.tar` (from training)
  - `cmvn.ark` (copy from `data/libri/cmvn.ark`)
  - `dict.txt` (copy from `data/libri/dict.txt`)
  - `train_bpe1000.model` (copy/rename from `data/libri/spm_bpe1000.model`)

Then run:
- `export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH`
- `export PYTHONPATH=$PWD/:$PYTHONPATH`
- `speech2text.py --asr_type "aed" \
    --model_dir pretrained_models/FireRedASR-AED-L \
    --wav_path examples/wav/BAC009S0764W0121.wav`

---

**7. Validation and WER**
- After decoding a set:
  - `wer.py --ref data/libri/dev/text --hyp your_decode.txt --print_sentence_wer 1`

---

**Notes & Tips**
- Start with a smaller model (`n_layers_enc=12`, `n_layers_dec=6`, `d_model=512`) if GPU memory is limited.
- The training loop uses teacher-forcing cross-entropy over `<sos> tokens -> tokens <eos>` pairs.
- Ensure transcripts used for SPM and training are uppercased to match `ChineseCharEnglishSpmTokenizer` behavior.
- LibriSpeech is English; Chinese-specific logic is harmless and unused here.

