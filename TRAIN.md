# FireRedASR AED 训练流程（LibriSpeech）

本文提供从零到可用的完整训练流程，覆盖数据准备、分词模型/词典构建、CMVN 统计、训练、推理打包与评测。训练目标为仓库自带的 AED 架构（ConformerEncoder + TransformerDecoder）。

提示：本仓库未提供 LLM 版训练脚本；本文聚焦 AED 训练。英文数据（LibriSpeech）可直接使用；中文相关逻辑不会产生副作用。

## 1. 环境准备
- Python 3.10（推荐）
- 安装依赖：`pip install -r requirements.txt`
- 准备 GPU/CUDA 环境（强烈建议）
- 可选：安装 ffmpeg，用于音频转换

设置路径（便于直接调用脚本）：
```
export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
```

## 2. 目录与变量
建议的数据与输出目录：
- `data/libri/{train,dev}/wav.scp` 和 `data/libri/{train,dev}/text`
- `data/libri/spm_bpe1000.model`、`data/libri/dict.txt`、`data/libri/cmvn.ark`
- 训练输出：`exp/aed_librispeech`

设置 LibriSpeech 根目录（替换为你的路径）：
```
LIBRISPEECH=/path/to/LibriSpeech
```

### 2.1 一键预处理（推荐）
使用预处理脚本可一次性完成 FLAC→WAV、清单生成、SPM/词典和 CMVN。

```
python fireredasr/data/progress.py \
  --librispeech_root $LIBRISPEECH \
  --out_dir data/libri 
```

常用可选参数：
- 已有 WAV，跳过转换：`--convert_to_wav 0`
- 自定义集合：`--train_splits "train-clean-100,train-clean-360,train-other-500"`，`--dev_splits "dev-clean,dev-other"`
- SPM：`--spm_vocab_size 1000 --spm_model_type bpe --character_coverage 1.0`
- 跳过步骤：`--no_spm` 或 `--no_cmvn`

脚本输出：
- `data/libri/train/{wav.scp,text}`，`data/libri/dev/{wav.scp,text}`
- `data/libri/train_text.txt`
- `data/libri/spm_bpe1000.model`（默认命名）
- `data/libri/dict.txt`
- `data/libri/cmvn.ark`

完成后可直接跳到第 6 步“启动训练（AED）”。若需自定义或了解细节，可按第 3–5 步手动执行。

## 3. 数据准备（LibriSpeech）
LibriSpeech 原始音频为 FLAC，需转为 16kHz/16-bit 单声道 WAV。以下示例涵盖 train 与 dev 两类 split。

1) 将所有 FLAC 转 WAV（覆盖写入同名 .wav）：
```
find $LIBRISPEECH -name "*.flac" | while read -r f; do \
  out="${f%.flac}.wav"; \
  mkdir -p "$(dirname "$out")"; \
  ffmpeg -loglevel error -y -i "$f" -ar 16000 -ac 1 -acodec pcm_s16le -f wav "$out"; \
done
```

2) 生成 wav.scp（uttid 为文件名，不含扩展名）：
```
mkdir -p data/libri/train data/libri/dev

# 训练集合并（示例：train-clean-100 + train-clean-360 + train-other-500）
for split in train-clean-100 train-clean-360 train-other-500; do \
  find $LIBRISPEECH/$split -name "*.wav" | while read -r f; do \
    id=$(basename "$f" .wav); echo "$id $f"; \
  done; \
done | LC_ALL=C sort > data/libri/train/wav.scp

# 验证集合并（dev-clean + dev-other）
for split in dev-clean dev-other; do \
  find $LIBRISPEECH/$split -name "*.wav" | while read -r f; do \
    id=$(basename "$f" .wav); echo "$id $f"; \
  done; \
done | LC_ALL=C sort > data/libri/dev/wav.scp
```

3) 生成 text（从官方 `*.trans.txt` 汇总）：
```
# 训练集 text
find $LIBRISPEECH -name "*.trans.txt" | grep -E "/(train-clean-100|train-clean-360|train-other-500)/" \
  | xargs cat \
  | awk '{utt=$1; $1=""; sub(/^ /, ""); txt=toupper($0); print utt, txt}' \
  | LC_ALL=C sort > data/libri/train/text

# 验证集 text
find $LIBRISPEECH -name "*.trans.txt" | grep -E "/(dev-clean|dev-other)/" \
  | xargs cat \
  | awk '{utt=$1; $1=""; sub(/^ /, ""); txt=toupper($0); print utt, txt}' \
  | LC_ALL=C sort > data/libri/dev/text
```

4) 检查对齐（uttid 一致性）：
```
diff <(cut -d ' ' -f1 data/libri/train/wav.scp) <(cut -d ' ' -f1 data/libri/train/text) | wc -l
diff <(cut -d ' ' -f1 data/libri/dev/wav.scp)   <(cut -d ' ' -f1 data/libri/dev/text)   | wc -l
```
输出应为 0（或尽量小）。

## 4. 训练 SPM 与词典
1) 汇总训练文本用于 SPM：
```
cut -d ' ' -f 2- data/libri/train/text > data/libri/train_text.txt
```

2) 训练 SentencePiece（示例 BPE-1000）：
```
python fireredasr/tokenizer/train_spm.py \
  --input_text data/libri/train_text.txt \
  --model_prefix data/libri/spm_bpe1000 \
  --vocab_size 1000 --character_coverage 1.0 --model_type bpe
```

3) 由 SPM 生成字典（包含 <pad>/<unk>/<sos>/<eos>）：
```
python fireredasr/tokenizer/build_dict_from_spm.py \
  --spm_model data/libri/spm_bpe1000.model \
  --out_dict  data/libri/dict.txt
```

## 5. 计算 CMVN（训练集）
```
python fireredasr/data/compute_cmvn.py \
  --wav_scp data/libri/train/wav.scp \
  --out_cmvn data/libri/cmvn.ark
```

## 6. 启动训练（AED）
单卡（或选择性设置 `CUDA_VISIBLE_DEVICES`）：
```
python fireredasr/train/train_aed.py \
  --train_wav_scp data/libri/train/wav.scp \
  --train_text    data/libri/train/text \
  --valid_wav_scp data/libri/dev/wav.scp \
  --valid_text    data/libri/dev/text \
  --cmvn          data/libri/cmvn.ark \
  --dict          data/libri/dict.txt \
  --spm_model     data/libri/spm_bpe1000.model \
  --batch_size 8 --epochs 10 --lr 2e-4 \
  --save_dir exp/aed_librispeech
```

或使用简化入口（仅需指定 `--asr_type aed`，其余参数与上相同，透传给 AED 训练器）：
```
python asr_train.py --asr_type aed \
  --train_wav_scp data/libri/train/wav.scp \
  --train_text    data/libri/train/text \
  --valid_wav_scp data/libri/dev/wav.scp \
  --valid_text    data/libri/dev/text \
  --cmvn          data/libri/cmvn.ark \
  --dict          data/libri/dict.txt \
  --spm_model     data/libri/spm_bpe1000.model \
  --batch_size 8 --epochs 10 --lr 2e-4 \
  --save_dir exp/aed_librispeech
```

说明：
- 日志中 `train_loss/token`、`valid_loss/token` 为按 token 聚合的交叉熵（已忽略 <pad>）。
- 如显存不足，可降低模型规模（示例）：`--n_layers_enc 12 --n_layers_dec 6 --d_model 512 --n_head 8`。
- 当前脚本未集成 AMP/多卡/DDP 与调度器；如需可后续扩展。

## 7. 推理打包
整理为推理目录（便于 `speech2text.py` 使用）：
```
mkdir -p pretrained_models/FireRedASR-AED-L
cp exp/aed_librispeech/model.pth.tar pretrained_models/FireRedASR-AED-L/
cp data/libri/cmvn.ark                   pretrained_models/FireRedASR-AED-L/
cp data/libri/dict.txt                   pretrained_models/FireRedASR-AED-L/
cp data/libri/spm_bpe1000.model          pretrained_models/FireRedASR-AED-L/train_bpe1000.model
```

快速推理（单条音频）：
```
speech2text.py --asr_type "aed" \
  --model_dir pretrained_models/FireRedASR-AED-L \
  --wav_path examples/wav/BAC009S0764W0121.wav
```

## 8. 评测（WER/CER）
将推理输出整理为 `uttid\ttext` 格式的文件后：
```
wer.py --ref data/libri/dev/text --hyp your_decode.txt --print_sentence_wer 1
```

## 9. 常见问题
- 显存不足：降低 batch 或模型规模；必要时切分更小 batch。
- 训练慢：可尝试更小 `d_model`，或后续加入 AMP/梯度累积。
- 文本大小写：本项目 tokenizer 在分词前会转大写；确保 SPM 训练数据与 text 一致（本文流程已统一大写）。
- wav.scp 与 text 不对齐：保证二者的 uttid 集合一致（第 3 步的 diff 检查应为 0）。
- 断点续训：当前脚本未提供 `--resume`；如需可补充加载已有 checkpoint 的逻辑。

## 10. 进一步优化（可选方向）
- 标签平滑（label smoothing）以提升泛化
- 学习率调度（Noam/余弦+warmup）
- 增广（SpecAugment、速度扰动）
- AMP/梯度累积/多卡 DDP

—— 若需要，我们可将以上策略集成到训练脚本，并在本文件追加对应使用说明。
