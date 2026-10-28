"""Forget/retain corpora, evaluation sets and the benign relearning data."""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

LLM_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("PLEU_LLM_DATA", LLM_DIR / "data"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PLEU_LLM_OUTPUT", LLM_DIR / "outputs"))
FORGET_DATASETS = ("wmdp", "rwku")


def rwku_dir(data_root):
    """Expected layout (SEUF release): RWKU/wiki_forget_set/*.json ({"text"}),
    RWKU/forget_eval/*.json ({"question", "answer"})."""
    return Path(data_root) / "RWKU"


def _limit(items, max_samples):
    return items[:max_samples] if max_samples and max_samples > 0 else items


def load_wmdp_corpus(subset, max_samples):
    from datasets import load_dataset

    rows = load_dataset("cais/wmdp-corpora", subset, split="train")
    return _limit([row["text"] for row in rows if row.get("text")], max_samples)


def load_rwku_forget_corpus(data_root, max_samples):
    texts = []
    for path in sorted((rwku_dir(data_root) / "wiki_forget_set").glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            text = row.get("text", "").strip()
            if text:
                texts.append(text)
    if not texts:
        raise FileNotFoundError(f"No RWKU forget corpus under {rwku_dir(data_root) / 'wiki_forget_set'}")
    return _limit(texts, max_samples)


def load_wikitext(max_samples):
    from datasets import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return _limit([row["text"].strip() for row in rows if len(row["text"].strip()) > 100], max_samples)


def forget_and_retain_texts(args):
    """WMDP: cyber forget/retain corpora. RWKU: Wikipedia passages of the targets / WikiText.
    ``--forget-file`` / ``--retain-file`` (JSONL with "text") replace the downloaded corpora."""
    if getattr(args, "forget_file", None) and getattr(args, "retain_file", None):
        return (load_text_jsonl(args.forget_file, args.max_forget_samples),
                load_text_jsonl(args.retain_file, args.max_retain_samples))
    if args.forget_dataset == "wmdp":
        forget = load_wmdp_corpus(f"{args.wmdp_subset}-forget-corpus", args.max_forget_samples)
        retain = load_wmdp_corpus(f"{args.wmdp_subset}-retain-corpus", args.max_retain_samples)
    else:
        forget = load_rwku_forget_corpus(args.data_root, args.max_forget_samples)
        retain = load_wikitext(args.max_retain_samples)
    if getattr(args, "forget_file", None):
        forget = load_text_jsonl(args.forget_file, args.max_forget_samples)
    if getattr(args, "retain_file", None):
        retain = load_text_jsonl(args.retain_file, args.max_retain_samples)
    return forget, retain


def load_text_jsonl(path, max_samples=0):
    texts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                text = json.loads(line).get("text", "").strip()
                if text:
                    texts.append(text)
    return _limit(texts, max_samples)


def relearn_data_path(data_root, forget_dataset):
    return Path(data_root) / "relearn" / f"{forget_dataset}_benign.jsonl"


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, cutoff_len):
        self.items = [tokenizer(text, truncation=True, max_length=cutoff_len) for text in texts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_lm(tokenizer):
    def collate(features):
        features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
    return collate


def make_loader(dataset, tokenizer, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_lm(tokenizer))


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def lm_loss(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                 labels=batch["labels"], use_cache=False).loss


def split_holdout(texts, holdout, seed):
    """Deterministic (train, held-out) split used for curve-point selection."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=generator).tolist()
    holdout = min(holdout, max(len(texts) - 1, 0))
    return [texts[i] for i in order[holdout:]], [texts[i] for i in order[:holdout]]
