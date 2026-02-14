# SigExt Extension: BigBird & ModernBERT Encoders

This extension replaces the original Longformer keyphrase extractor with **BigBird** and **ModernBERT** encoders, and uses local LLMs (Mistral, Llama, Qwen) instead of the Claude API.

## Pipeline

The pipeline has four stages.

### 1. Prepare the dataset

```bash
python3 src/prepare_data.py \
  --dataset cnn \
  --output_dir experiments/cnn_dataset/
```

### 2. Train the keyphrase extractor

```bash
python3 src/updated_extractor.py \
  --model bigbird_large \
  --dataset_dir experiments/cnn_dataset/ \
  --checkpoint_dir experiments/cnn_extractor_model/
```

Available `--model` options:

| Key | Model | Notes |
|-----|-------|-------|
| `bigbird_base` | BigBird-RoBERTa-base | block_size=64, random_blocks=3 |
| `bigbird_large` | BigBird-RoBERTa-large | block_size=64, random_blocks=3 |
| `bigbird_large_r{2,3,5}b{32,64,128}` | BigBird-RoBERTa-large | various block_size / random_blocks combos |
| `modernbert_large` | ModernBERT-large | max_length=8192 |
| `longformer` / `longformer_large` | Longformer | original baseline |

### 3. Extract keyphrases (inference)

```bash
python3 src/inference_longformer_extractor.py \
  --model bigbird_large \
  --dataset_dir experiments/cnn_dataset/ \
  --checkpoint_dir experiments/cnn_extractor_model/ \
  --output_dir experiments/cnn_dataset_with_keyphrase/
```

This scores each candidate phrase and writes `input_kw_model` into the output JSONL files. The best checkpoint is selected automatically by `recall_20`.

### 4. Run LLM summarization

```bash
python3 src/zs_summarization.py \
  --model_name mistral \
  --kw_strategy sigext_topk \
  --kw_model_top_k 15 \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset_with_keyphrase/ \
  --output_dir experiments/cnn_extsig_predictions/
```

**LLM options (`--model_name`):** `mistral` (Mistral-7B-Instruct-v0.2), `llama` (Llama-3.1-8B-Instruct), `qwen` (Qwen2.5-7B-Instruct).