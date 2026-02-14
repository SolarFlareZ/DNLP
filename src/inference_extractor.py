import argparse
import copy
import glob
import logging
import pathlib

import jsonlines
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.train_extractor_context import KWDatasetContext, KeywordExtractorClf


MODELS = {
    "longformer": {
        "name": "allenai/longformer-base-4096",
        "max_length": 4096,
        "needs_block_padding": False,
    },
    "longformer_large": {
        "name": "allenai/longformer-large-4096",
        "max_length": 4096,
        "needs_block_padding": False,
    },
    "bigbird_base": {
        "name": "google/bigbird-roberta-base",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 3, 
    },
    "bigbird_large": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 3, 
    },
    # block_size=32
    "bigbird_large_r2b32": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 32,
        "use_explicit_mask": True,
        "num_random_blocks": 2
    },
    "bigbird_large_r3b32": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 32,
        "use_explicit_mask": True,
        "num_random_blocks": 3
    },
    "bigbird_large_r5b32": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 32,
        "use_explicit_mask": True,
        "num_random_blocks": 5
    },
    # block_size=64
    "bigbird_large_r2b64": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 2
    },
    "bigbird_large_r3b64": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 3
    },
    "bigbird_large_r5b64": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 64,
        "use_explicit_mask": True,
        "num_random_blocks": 5
    },
    # block_size=128
    "bigbird_large_r2b128": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 128,
        "use_explicit_mask": True,
        "num_random_blocks": 2
    },
    "bigbird_large_r3b128": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 128,
        "use_explicit_mask": True,
        "num_random_blocks": 3
    },
    "bigbird_large_r5b128": {
        "name": "google/bigbird-roberta-large",
        "max_length": 4096,
        "needs_block_padding": True,
        "block_size": 128,
        "use_explicit_mask": True,
        "num_random_blocks": 5
    },
    "modernbert_large": {
        "name": "answerdotai/ModernBERT-large",
        "max_length": 8192,
        "needs_block_padding": False,
        "use_explicit_mask": False,
    },
}


def find_best_checkpoint(checkpoint_dir):
    path = pathlib.Path(checkpoint_dir).expanduser()

    if path.is_file() and path.suffix == ".ckpt":
        return str(path)
    
    candidates = sorted(glob.glob(f"{checkpoint_dir}/lightning_logs/*/checkpoints/*.ckpt"))

    if not len(candidates):
        raise RuntimeError("Candidates not found.")

    best_score = 0
    best_checkpoint = None
    for item in candidates:
        tokens = pathlib.Path(item).stem.split("-")
        info = {}
        for tok in tokens:
            if "_" in tok:
                parts = tok.split("_", 1)
                if len(parts) >= 2:
                    info[parts[0]] = parts[1]

        if "recall20" in info and float(info["recall20"]) >= best_score:
            best_checkpoint = item
            best_score = float(info["recall20"])
    
    if best_checkpoint is None:
        best_checkpoint = candidates[-1]
        logging.warning(f"Using fallback checkpoint: {best_checkpoint}")
        
    return best_checkpoint


def parse_result(raw_dataset, predicts):
    raw_dataset = copy.deepcopy(raw_dataset)
    for pred_info in predicts:
        example_info = raw_dataset[pred_info["id"]]
        score = pred_info["score"]
        if len(score) > len(example_info["trunc_input_phrases"]):
            raise RuntimeError("model prediction does not match with number of phrases.")
        example_info["input_kw_model"] = score

    return raw_dataset


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Run inference with multiple model support")
    
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help=f"Model to use for inference. Options: {list(MODELS.keys())}"
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="Directory of train and validation data.")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Directory of checkpoints or path to .ckpt file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save predictions.")

    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    base_model = model_config["name"]
    base_model_max_length = model_config["max_length"]
    needs_block_padding = model_config.get("needs_block_padding", False)
    block_size = model_config.get("block_size", 64)

    best_checkpoint = find_best_checkpoint(args.checkpoint_dir)
    logging.info(f"Using checkpoint: {best_checkpoint}")
    logging.info(f"Using model: {base_model}")
    
    model = KeywordExtractorClf.load_from_checkpoint(best_checkpoint, model_config=model_config)

    trainer = pl.Trainer(devices=1, accelerator="auto")

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["validation", "test"]:
        split_file = dataset_dir.joinpath(f"{split}.jsonl")
        if not split_file.exists():
            logging.info(f"Skipping {split} split (file not found)")
            continue
            
        logging.info(f"Processing {split} split...")
        
        dataset = KWDatasetContext(
            dataset_filename=str(split_file),
            base_model=base_model,
            example_kw_hit_threshold=0,
            base_model_max_length=base_model_max_length,
            hide_gt=True,
            needs_block_padding=needs_block_padding,
            block_size=block_size,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        predicts = trainer.predict(model, dataloaders=dataloader)

        dataset_with_prediction = parse_result(dataset.raw_dataset, predicts)

        output_file = output_dir.joinpath(f"{split}.jsonl")
        with jsonlines.open(str(output_file), "w") as f:
            f.write_all(dataset_with_prediction)
        
        logging.info(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()