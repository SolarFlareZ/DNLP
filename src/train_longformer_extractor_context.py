import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import logging
import pathlib

import jsonlines
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
import transformers
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from transformers import BitsAndBytesConfig  # <<< ADDED


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

class KWDatasetContext(Dataset):
    def __init__(
            self,
            dataset_filename,
            base_model,
            example_kw_hit_threshold=1,
            base_model_max_length=4096,
            hide_gt=False,
            needs_block_padding=False,
            block_size=64
    ):
        super().__init__()
        self.data = []
        self.needs_block_padding = needs_block_padding
        self.block_size = block_size

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        if hide_gt:
            assert example_kw_hit_threshold == 0

        logging.info(f"Loading data from {dataset_filename}")
        with jsonlines.open(dataset_filename) as f:
            self.raw_dataset = list(f)

        pos_cc = 0
        neg_cc = 0

        for idx, item in tqdm.tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset),
                                   desc="process data"):
            x = [self.tokenizer.cls_token_id if self.tokenizer.cls_token_id else self.tokenizer.bos_token_id]
            y = [-100]
            example_kw_hit_cc = 0

            selected_keyword_strs = set()
            if not hide_gt:
                for kw in item["trunc_input_phrases"]:
                    best_rouge_f = 0
                    best_rouge_p = 0
                    best_rouge_r = 0
                    for ent in item["trunc_output_phrases"]:
                        score = scorer.score(target=ent["phrase"], prediction=kw["phrase"])
                        best_rouge_f = max(best_rouge_f, score["rouge1"].fmeasure)
                        best_rouge_p = max(best_rouge_p, score["rouge1"].precision)
                        best_rouge_r = max(best_rouge_r, score["rouge1"].recall)

                    if best_rouge_f >= 0.6 or best_rouge_p >= 0.8 or best_rouge_r >= 0.8:
                        selected_keyword_strs.add(kw["phrase"])

            current_text_index = 0
            for kw_info in item["trunc_input_phrases"]:
                if current_text_index < kw_info["index"]:
                    content = item["trunc_input"][current_text_index: kw_info["index"]]
                    content_tokens = self.tokenizer(content)["input_ids"][1:-1]
                    x.extend(content_tokens)
                    y.extend([-100] * len(content_tokens))
                else:
                    if current_text_index != 0:
                        content_tokens = self.tokenizer(" ")["input_ids"][1:-1]
                        x.extend(content_tokens)
                        y.extend([-100] * len(content_tokens))

                format_kw = f"{kw_info['phrase']}"
                input_ids = self.tokenizer(format_kw)["input_ids"][1:-1]

                if not hide_gt and kw_info["phrase"] in selected_keyword_strs:
                    labels = [1] * len(input_ids)
                else:
                    labels = [0] * len(input_ids)

                if len(labels) > 0 and labels[-1] == 1:
                    example_kw_hit_cc += 1

                x.extend(input_ids)
                y.extend(labels)
                current_text_index = kw_info["index"] + len(kw_info["phrase"])

            if current_text_index < len(item["trunc_input"]):
                content = item["trunc_input"][current_text_index:]
                content_tokens = self.tokenizer(content)["input_ids"][1:-1]
                x.extend(content_tokens)
                y.extend([-100] * len(content_tokens))

            x = x[: base_model_max_length - 1]
            y = y[: base_model_max_length - 1]

            sep = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id else 2
            x.append(sep)
            y.append(-100)

            if self.needs_block_padding:
                rem = len(x) % self.block_size
                if rem != 0:
                    pad_len = self.block_size - rem
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    x.extend([pad_id] * pad_len)
                    y.extend([-100] * pad_len)

            x = torch.tensor(x).long()
            y = torch.tensor(y).long()

            if example_kw_hit_cc >= example_kw_hit_threshold:
                self.data.append((x, y, idx))
                pos_cc += torch.sum(y == 1).numpy()
                neg_cc += torch.sum(y == 0).numpy()

        logging.info(f"keyword ratio {pos_cc / (pos_cc + neg_cc) if (pos_cc+neg_cc) > 0 else 0}")
        logging.info(f"Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KeywordExtractorClf(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config

        logging.info(f"Initializing Model: {model_config['name']}")

        config = AutoConfig.from_pretrained(model_config['name'], trust_remote_code=True)
        config.num_labels = 2

        if "bigbird" in model_config['name'].lower():
            if 'block_size' in model_config:
                config.block_size = model_config['block_size']
            if 'num_random_blocks' in model_config:
                config.num_random_blocks = model_config['num_random_blocks']
            config.attention_type = "block_sparse"
            logging.info(f"BigBird config: block_size={config.block_size}, "
                        f"num_random_blocks={config.num_random_blocks}")

        attn_impl = "sdpa" if "ModernBERT" in model_config['name'] else None

        self.clf = AutoModelForTokenClassification.from_pretrained(
            model_config['name'], 
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl 
        )

        self.clf.gradient_checkpointing_enable()

        if hasattr(self.clf.config, "use_cache"):
            self.clf.config.use_cache = False

        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        x, y, idx = batch
        assert x.size(0) == 1
        logits = F.log_softmax(self.clf(x)[0], dim=-1)

        kw_count = 0
        score_and_label = []

        current_logits = 0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y[0][i] != -100:
                if not current_in_phrase:
                    current_in_phrase = True
                current_logits += logits[0][i][1]
                current_len += 1
            else:
                if current_in_phrase:
                    current_in_phrase = False
                    score_and_label.append({"kw_index": kw_count, "score": float(current_logits / current_len)})
                    kw_count += 1
                    current_logits = 0
                    current_len = 0

        return {"id": int(idx[0]), "score": score_and_label}

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        assert x.size(0) == 1
        logits = self.clf(x)[0]

        loss_weight = torch.tensor([0.05, 1], device=self.device).type_as(logits)
        loss = F.cross_entropy(logits[0], y[0], reduction="sum", weight=loss_weight)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        assert x.size(0) == 1
        logits = self.clf(x)[0]

        loss_weight = torch.tensor([0.05, 1], device=self.device).type_as(logits)

        loss = (
            F.cross_entropy(logits[0], y[0], reduction="sum", weight=loss_weight)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        logits = F.log_softmax(logits, dim=-1)

        score_and_label = []

        logits = logits.float().cpu().numpy()
        y = y.cpu().numpy()

        current_logits = 0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y[0][i] != -100:
                if not current_in_phrase:
                    current_in_phrase = True
                current_logits += logits[0][i][1]
                current_len += 1
            else:
                if current_in_phrase:
                    current_in_phrase = False
                    score_and_label.append(
                        {"index": i, "logits": float(current_logits / current_len), "label": y[0][i - 1]}
                    )
                    current_logits = 0
                    current_len = 0

        score_and_label = sorted(score_and_label, key=lambda item: (item["logits"], -item["index"]), reverse=True)

        step_output = {
            "loss": loss,
            "logits_std": np.std([item["logits"] for item in score_and_label]) if score_and_label else 0.0,
        }

        for top_k in [5, 10, 15, 20, 35]:
            if len(score_and_label) > 0:
                step_output[f"precision_{top_k}"] = np.mean([item["label"] for item in score_and_label[:top_k]])
                total_positives = np.sum([item["label"] for item in score_and_label])
                step_output[f"recall_{top_k}"] = np.sum([item["label"] for item in score_and_label[:top_k]]) / max(total_positives, 1)
            else:
                step_output[f"precision_{top_k}"] = 0.0
                step_output[f"recall_{top_k}"] = 0.0

        self.validation_step_outputs.append(step_output)

    def on_validation_epoch_end(self):
        for top_k in [5, 10, 15, 20, 35]:
            self.log(
                f"val/precision_{top_k}",
                float(np.mean([item[f"precision_{top_k}"] for item in self.validation_step_outputs])),
                sync_dist=True,
            )
            self.log(
                f"val/recall_{top_k}",
                float(np.mean([item[f"recall_{top_k}"] for item in self.validation_step_outputs])),
                sync_dist=True,
            )
        self.log("val/loss", float(np.mean([item["loss"] for item in self.validation_step_outputs])), sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.estimated_stepping_batches, num_warmup_steps=100
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Train longformer keyword extractor.")
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="directory to save checkpoints.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for trainer.")

    parser.add_argument(
        "--model",
        default="longformer",
        choices=list(MODELS.keys()),
        help="Which model config to use."
    )

    parser.add_argument("--load_ckpt", default=None, type=str, help="Pretrained ckpt.")
    parser.add_argument("--accumulate_grad_batches", default=32, type=int)

    args = parser.parse_args()
    model_config = MODELS[args.model]

    pl.seed_everything(args.seed)

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()

    train_set = KWDatasetContext(
        dataset_filename=str(dataset_dir.joinpath("train.jsonl")),
        base_model=model_config["name"],
        base_model_max_length=model_config["max_length"],
        needs_block_padding=model_config.get("needs_block_padding", False),
        block_size=model_config.get("block_size", 64)
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = KWDatasetContext(
        dataset_filename=str(dataset_dir.joinpath("validation.jsonl")),
        base_model=model_config["name"],
        base_model_max_length=model_config["max_length"],
        needs_block_padding=model_config.get("needs_block_padding", False),
        block_size=model_config.get("block_size", 64)
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    if args.load_ckpt:
        model = KeywordExtractorClf.load_from_checkpoint(args.load_ckpt, model_config=model_config)
    else:
        model = KeywordExtractorClf(model_config=model_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        monitor="val/recall_20",
        mode="max",
        every_n_epochs=1,
        filename="epoch_{epoch:02d}-step_{step:06d}-recall20_{val/recall_20:.3f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        default_root_dir=args.checkpoint_dir,
        strategy="auto",
        accelerator="auto",
        devices=1,
        precision="bf16-true",
        log_every_n_steps=5,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
