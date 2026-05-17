#!/usr/bin/env python3
"""DDP fine-tuning entrypoint for BAAI/bge-m3 on Kaggle T4 x2.

This script is intentionally self-contained so the notebook can write it into
/kaggle/working and launch it with torchrun.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path


os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import Dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--tag", default="bge-m3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--hardneg-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--cached-mini-batch-size", type=int, default=1)
    parser.add_argument("--n-hard", type=int, default=5)
    parser.add_argument("--bm25-top", type=int, default=30)
    parser.add_argument("--optim", default="adafactor")
    parser.add_argument("--hf-hub-repo", default="")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--is-e5-model", action="store_true")
    parser.add_argument("--no-cached-loss", dest="use_cached_loss", action="store_false")
    parser.add_argument(
        "--no-gather-across-devices",
        dest="gather_across_devices",
        action="store_false",
    )
    parser.set_defaults(use_cached_loss=True, gather_across_devices=True)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return rank() == 0


def print_main(*parts: object) -> None:
    if is_main_process():
        print(*parts, flush=True)


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def sync_json(path: Path, payload: dict | None = None) -> dict:
    if is_main_process():
        path.write_text(json.dumps(payload or {}, indent=2))
    barrier()
    data = json.loads(path.read_text())
    barrier()
    return data


def configure_runtime(args: argparse.Namespace) -> str:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        torch.cuda.manual_seed_all(args.seed)
        device = f"cuda:{local_rank()}"
    else:
        device = "cpu"

    if world_size() > 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, timeout=timedelta(hours=6))

    args.working_dir.mkdir(parents=True, exist_ok=True)
    print_main(
        f"DDP world_size={world_size()} | cuda={torch.cuda.is_available()} | "
        f"visible_gpus={torch.cuda.device_count()} | max_seq={args.max_seq_length}"
    )
    return device


def load_zalo_data(args: argparse.Namespace) -> tuple[list[dict], list[str], dict, list[dict], list[dict]]:
    corpus = []
    with args.corpus_file.open() as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    corpus_key_to_idx = {}
    corpus_texts = []
    for i, art in enumerate(corpus):
        key = (str(art["law_id"]), str(art["article_id"]))
        corpus_key_to_idx[key] = i
        corpus_texts.append(art["text"])

    qa_all = []
    with args.qa_file.open() as f:
        for line in f:
            if line.strip():
                qa_all.append(json.loads(line))

    def norm_key(record: dict) -> tuple[str, str]:
        law_id, article_id = str(record["law_id"]), str(record["article_id"])
        if "__" not in article_id:
            article_id = f"{law_id}__{article_id}"
        return law_id, article_id

    qa_valid = []
    dropped = 0
    for q in qa_all:
        rel = q.get("relevant_articles") or []
        keys = [norm_key(r) for r in rel]
        if rel and all(k in corpus_key_to_idx for k in keys):
            qa_valid.append({**q, "relevant_keys": keys})
        else:
            dropped += 1

    qa_train, qa_test = train_test_split(
        qa_valid, test_size=0.2, random_state=args.seed, shuffle=True
    )
    split_meta = {
        "seed": args.seed,
        "train_ids": [q["question_id"] for q in qa_train],
        "test_ids": [q["question_id"] for q in qa_test],
    }
    if is_main_process():
        (args.working_dir / "zalo_split.json").write_text(json.dumps(split_meta, indent=2))

    print_main(
        f"Corpus={len(corpus)} | QA valid={len(qa_valid)} | dropped={dropped} | "
        f"train={len(qa_train)} | test={len(qa_test)}"
    )
    return corpus, corpus_texts, corpus_key_to_idx, qa_train, qa_test


def recall_at_k(hits: list[tuple[str, str]], relevant: set[tuple[str, str]], k: int) -> float:
    if not relevant:
        return 0.0
    return float(any(h in relevant for h in hits[:k]))


def precision_at_k(hits: list[tuple[str, str]], relevant: set[tuple[str, str]], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = hits[:k]
    if not top_k:
        return 0.0
    return sum(1 for h in top_k if h in relevant) / k


def reciprocal_rank(hits: list[tuple[str, str]], relevant: set[tuple[str, str]]) -> float:
    if not relevant:
        return 0.0
    for item_rank, h in enumerate(hits, 1):
        if h in relevant:
            return 1.0 / item_rank
    return 0.0


def ndcg_at_k(hits: list[tuple[str, str]], relevant: set[tuple[str, str]], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, h in enumerate(hits[:k], 1) if h in relevant)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def f_beta_at_k(
    hits: list[tuple[str, str]], relevant: set[tuple[str, str]], k: int, beta: float = 2.0
) -> float:
    if not relevant or k <= 0:
        return 0.0
    matched = sum(1 for h in hits[:k] if h in relevant)
    precision = matched / k
    recall = matched / len(relevant)
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def compute_metrics(
    predictions: list[list[tuple[str, str]]],
    gold_lists: list[set[tuple[str, str]]],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    n = len(predictions)
    out = {f"recall@{k}": 0.0 for k in k_values}
    out.update({f"precision@{k}": 0.0 for k in k_values})
    out.update({f"ndcg@{k}": 0.0 for k in k_values})
    out.update({f"f2@{k}": 0.0 for k in k_values})
    out["mrr"] = 0.0
    for hits, gold in zip(predictions, gold_lists):
        out["mrr"] += reciprocal_rank(hits, gold)
        for k in k_values:
            out[f"recall@{k}"] += recall_at_k(hits, gold, k)
            out[f"precision@{k}"] += precision_at_k(hits, gold, k)
            out[f"ndcg@{k}"] += ndcg_at_k(hits, gold, k)
            out[f"f2@{k}"] += f_beta_at_k(hits, gold, k, beta=2.0)
    return {key: round(value / n, 4) for key, value in out.items()}


def safe_encode(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    current_batch = max(1, batch_size)
    while current_batch >= 1:
        try:
            return model.encode(
                texts,
                batch_size=current_batch,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current_batch == 1:
                raise
            print_main(f"CUDA OOM during encode; retrying with batch_size={current_batch // 2}")
            current_batch //= 2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("safe_encode exhausted all batch sizes")


def retrieve_top_k(query_vecs: np.ndarray, corpus_vecs: np.ndarray, k: int = 10) -> np.ndarray:
    sims = query_vecs @ corpus_vecs.T
    top_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    rows = np.arange(len(query_vecs))[:, None]
    sorted_order = np.argsort(-sims[rows, top_idx], axis=1)
    return top_idx[rows, sorted_order]


def eval_model(
    model: SentenceTransformer,
    corpus: list[dict],
    corpus_texts: list[str],
    queries: list[str],
    gold_lists: list[set[tuple[str, str]]],
    *,
    batch_size: int,
    query_prefix: str = "",
    passage_prefix: str = "",
    k_values: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict, list[list[tuple[str, str]]]]:
    t0 = time.time()
    passages = [passage_prefix + t for t in corpus_texts] if passage_prefix else corpus_texts
    corpus_vecs = safe_encode(
        model, passages, batch_size=batch_size, show_progress_bar=is_main_process()
    )
    t_corpus = time.time() - t0

    t1 = time.time()
    query_texts = [query_prefix + q for q in queries] if query_prefix else queries
    q_vecs = safe_encode(model, query_texts, batch_size=batch_size, show_progress_bar=False)
    t_query = time.time() - t1

    top_idx = retrieve_top_k(q_vecs, corpus_vecs, k=max(k_values))
    predictions = [
        [(str(corpus[int(idx)]["law_id"]), str(corpus[int(idx)]["article_id"])) for idx in row]
        for row in top_idx
    ]
    metrics = compute_metrics(predictions, gold_lists, k_values=k_values)
    metrics["_t_corpus_encode_s"] = round(t_corpus, 1)
    metrics["_t_query_encode_s"] = round(t_query, 1)
    return metrics, predictions


def load_model(args: argparse.Namespace, model_name_or_path: str | Path, device: str) -> SentenceTransformer:
    kwargs = {"device": device}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(str(model_name_or_path), **kwargs)
    model.max_seq_length = args.max_seq_length
    try:
        model[0].auto_model.config.use_cache = False
    except Exception:
        pass
    try:
        model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        try:
            model[0].auto_model.gradient_checkpointing_enable()
        except Exception as exc:
            print_main(f"Gradient checkpointing unavailable: {exc}")
    except Exception as exc:
        print_main(f"Gradient checkpointing unavailable: {exc}")
    return model


def make_loss(args: argparse.Namespace, model: SentenceTransformer):
    loss_cls = (
        losses.CachedMultipleNegativesRankingLoss
        if args.use_cached_loss
        else losses.MultipleNegativesRankingLoss
    )
    kwargs = {"gather_across_devices": args.gather_across_devices and world_size() > 1}
    if args.use_cached_loss:
        kwargs["mini_batch_size"] = args.cached_mini_batch_size
    try:
        return loss_cls(model, **kwargs)
    except TypeError:
        kwargs.pop("gather_across_devices", None)
        return loss_cls(model, **kwargs)


def build_dataset_pairs(
    qa_train: list[dict],
    corpus_texts: list[str],
    corpus_key_to_idx: dict,
    *,
    query_prefix: str,
    passage_prefix: str,
) -> Dataset:
    anchors, positives = [], []
    for q in qa_train:
        pos_key = q["relevant_keys"][0]
        pos_idx = corpus_key_to_idx[pos_key]
        anchors.append(query_prefix + q["question"])
        positives.append(passage_prefix + corpus_texts[pos_idx])
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def tokenize_vi(text: str) -> list[str]:
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize(text.lower()).split()


def build_or_load_hardneg_dataset(
    args: argparse.Namespace,
    qa_train: list[dict],
    corpus_texts: list[str],
    corpus_key_to_idx: dict,
    *,
    query_prefix: str,
    passage_prefix: str,
) -> Dataset:
    hardneg_path = args.working_dir / f"hardneg_{args.tag}.jsonl"
    if is_main_process():
        print_main("Building BM25 hard negatives...")
        t0 = time.time()
        tokenized_corpus = [tokenize_vi(t) for t in tqdm(corpus_texts, desc="Tokenize")]
        bm25 = BM25Okapi(tokenized_corpus)

        rows = []
        skipped = 0
        for q in tqdm(qa_train, desc="Mine hard negs"):
            pos_key = q["relevant_keys"][0]
            pos_idx = corpus_key_to_idx[pos_key]
            scores = bm25.get_scores(tokenize_vi(q["question"]))
            top_idx = np.argpartition(-scores, kth=args.bm25_top)[: args.bm25_top]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            hard_negs = []
            for idx in top_idx:
                if int(idx) == pos_idx:
                    continue
                hard_negs.append(passage_prefix + corpus_texts[int(idx)])
                if len(hard_negs) >= args.n_hard:
                    break

            if not hard_negs:
                skipped += 1
                continue
            if len(hard_negs) < args.n_hard:
                skipped += 1
                continue

            row = {
                "anchor": query_prefix + q["question"],
                "positive": passage_prefix + corpus_texts[pos_idx],
            }
            for i, neg_text in enumerate(hard_negs, 1):
                row[f"negative_{i}"] = neg_text
            rows.append(row)

        with hardneg_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print_main(
            f"Hard negatives={len(rows)} | skipped={skipped} | built in {time.time() - t0:.1f}s"
        )

    barrier()
    loaded_rows = [json.loads(line) for line in hardneg_path.read_text().splitlines() if line.strip()]
    columns: dict[str, list[str]] = {}
    for row in loaded_rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return Dataset.from_dict(columns)


def training_args_for_stage(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    batch_size: int,
    lr: float,
    epoch: int,
    steps_per_epoch: int,
) -> SentenceTransformerTrainingArguments:
    logging_steps = max(1, steps_per_epoch // 10)
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": batch_size,
        "learning_rate": lr,
        "warmup_ratio": 0.1 if epoch == 1 else 0.0,
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        # Gradient checkpointing is enabled manually in load_model() with
        # use_reentrant=False. Letting Trainer enable it again can interact
        # poorly with CachedMultipleNegativesRankingLoss under DDP.
        "gradient_checkpointing": False,
        "optim": args.optim,
        "save_strategy": "no",
        "logging_steps": logging_steps,
        "report_to": [],
        "disable_tqdm": not is_main_process(),
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        # BGE-M3 has extra retrieval heads/parameters that are not used by the
        # dense sentence_embedding loss. DDP must be told to tolerate them.
        "ddp_find_unused_parameters": True,
        "ddp_timeout": 21600,
    }
    parameters = inspect.signature(SentenceTransformerTrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return SentenceTransformerTrainingArguments(**filtered)


def train_stage(
    args: argparse.Namespace,
    *,
    stage_name: str,
    model_name_or_path: str | Path,
    train_dataset: Dataset,
    corpus: list[dict],
    corpus_texts: list[str],
    test_queries: list[str],
    test_gold: list[set[tuple[str, str]]],
    query_prefix: str,
    passage_prefix: str,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[Path, dict]:
    model = load_model(args, model_name_or_path, device)
    best_dir = args.working_dir / f"checkpoint_{stage_name}_best"
    run_dir = args.working_dir / f"trainer_{stage_name}"
    control_path = args.working_dir / f"control_{stage_name}.json"
    best_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(1, batch_size * world_size())))
    best_epoch = 0
    best_score = -1.0
    patience_left = args.early_stopping_patience
    per_epoch: list[dict] = []

    print_main(
        f"{stage_name}: rows={len(train_dataset)} | per_device_batch={batch_size} | "
        f"global_batch={batch_size * world_size()} | cached={args.use_cached_loss}"
    )

    for epoch in range(1, args.epochs + 1):
        print_main(f"\n--- {stage_name} epoch {epoch}/{args.epochs} ---")
        loss = make_loss(args, model)
        trainer_args = training_args_for_stage(
            args,
            output_dir=run_dir / f"epoch_{epoch}",
            batch_size=batch_size,
            lr=lr,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            loss=loss,
        )

        t_train = time.time()
        trainer.train()
        train_seconds = time.time() - t_train
        del trainer, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        barrier()

        stop = False
        if is_main_process():
            metrics, _ = eval_model(
                model,
                corpus,
                corpus_texts,
                test_queries,
                test_gold,
                batch_size=args.eval_batch_size,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
            )
            metrics["epoch"] = epoch
            metrics["_t_train_s"] = round(train_seconds, 1)
            per_epoch.append(metrics)
            score = metrics["f2@10"]
            print_main(
                f"{stage_name} epoch {epoch}: F2@10={score:.4f} | "
                f"Recall@10={metrics['recall@10']:.4f} | MRR={metrics['mrr']:.4f}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience_left = args.early_stopping_patience
                model.save(str(best_dir))
                print_main(f"New best {stage_name}: F2@10={score:.4f}")
            else:
                patience_left -= 1
                stop = patience_left <= 0

        state = sync_json(
            control_path,
            {
                "stop": stop,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "patience_left": patience_left,
                "per_epoch": per_epoch,
            }
            if is_main_process()
            else None,
        )
        if state.get("stop"):
            print_main(f"Early stopping {stage_name} at epoch {epoch}.")
            break

    summary = json.loads(control_path.read_text())
    summary["best_metrics"] = (
        summary["per_epoch"][summary["best_epoch"] - 1] if summary.get("best_epoch") else {}
    )
    if is_main_process():
        (args.working_dir / f"metrics_{stage_name}.json").write_text(json.dumps(summary, indent=2))
    barrier()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_dir, summary


def maybe_push_winner(
    args: argparse.Namespace,
    winner_strategy: str,
    winner_dir: Path | None,
    device: str,
) -> None:
    if not is_main_process():
        return
    if not args.push_to_hub or winner_dir is None:
        print_main(f"Skipped push. winner_strategy={winner_strategy}, push_to_hub={args.push_to_hub}")
        return
    if not args.hf_hub_repo:
        print_main("HF repo is empty. Skip push.")
        return

    try:
        from huggingface_hub import login
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        login(token=token)
    except Exception as exc:
        print_main(f"Cannot load Kaggle secret HF_TOKEN ({exc}). Skip push.")
        return

    model = SentenceTransformer(str(winner_dir), device=device, trust_remote_code=args.trust_remote_code)
    model.max_seq_length = args.max_seq_length
    model.push_to_hub(args.hf_hub_repo, private=True, exist_ok=True)
    print_main(f"Pushed: https://huggingface.co/{args.hf_hub_repo}")


def main() -> None:
    args = parse_args()
    device = configure_runtime(args)
    corpus, corpus_texts, corpus_key_to_idx, qa_train, qa_test = load_zalo_data(args)
    q_prefix = "query: " if args.is_e5_model else ""
    p_prefix = "passage: " if args.is_e5_model else ""
    test_queries = [q["question"] for q in qa_test]
    test_gold = [set(q["relevant_keys"]) for q in qa_test]

    baseline_metrics = {}
    baseline_path = args.working_dir / "metrics_baseline.json"
    if is_main_process():
        if baseline_path.exists():
            baseline_metrics = json.loads(baseline_path.read_text())
            print_main(f"Reusing baseline metrics: {baseline_path}")
        else:
            print_main(f"Loading baseline {args.model_name}...")
            baseline_model = load_model(args, args.model_name, device)
            baseline_metrics, _ = eval_model(
                baseline_model,
                corpus,
                corpus_texts,
                test_queries,
                test_gold,
                batch_size=args.eval_batch_size,
                query_prefix=q_prefix,
                passage_prefix=p_prefix,
            )
            baseline_path.write_text(json.dumps(baseline_metrics, indent=2))
            print_main("Baseline:", json.dumps(baseline_metrics, indent=2))
            del baseline_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    baseline_metrics = sync_json(
        args.working_dir / "control_baseline.json",
        {"metrics": baseline_metrics} if is_main_process() else None,
    )["metrics"]

    pair_dataset = build_dataset_pairs(
        qa_train,
        corpus_texts,
        corpus_key_to_idx,
        query_prefix=q_prefix,
        passage_prefix=p_prefix,
    )
    inbatch_dir, inbatch_summary = train_stage(
        args,
        stage_name="inbatch",
        model_name_or_path=args.model_name,
        train_dataset=pair_dataset,
        corpus=corpus,
        corpus_texts=corpus_texts,
        test_queries=test_queries,
        test_gold=test_gold,
        query_prefix=q_prefix,
        passage_prefix=p_prefix,
        batch_size=args.train_batch_size,
        lr=args.lr,
        device=device,
    )

    hardneg_dataset = build_or_load_hardneg_dataset(
        args,
        qa_train,
        corpus_texts,
        corpus_key_to_idx,
        query_prefix=q_prefix,
        passage_prefix=p_prefix,
    )
    hardneg_dir, hardneg_summary = train_stage(
        args,
        stage_name="hardneg",
        model_name_or_path=inbatch_dir,
        train_dataset=hardneg_dataset,
        corpus=corpus,
        corpus_texts=corpus_texts,
        test_queries=test_queries,
        test_gold=test_gold,
        query_prefix=q_prefix,
        passage_prefix=p_prefix,
        batch_size=args.hardneg_batch_size,
        lr=args.lr / 2,
        device=device,
    )

    if is_main_process():
        candidates = [
            ("baseline", baseline_metrics, None),
            ("inbatch", inbatch_summary["best_metrics"], inbatch_dir),
            ("hardneg", hardneg_summary["best_metrics"], hardneg_dir),
        ]
        winner_strategy, winner_metrics, winner_dir = max(
            candidates, key=lambda item: item[1].get("f2@10", 0.0)
        )
        final_summary = {
            "model_name": args.model_name,
            "tag": args.tag,
            "baseline": baseline_metrics,
            "inbatch": inbatch_summary,
            "hardneg": hardneg_summary,
            "winner_strategy": winner_strategy,
            "winner_epoch": None if winner_strategy == "baseline" else winner_metrics.get("epoch"),
            "winner_metrics": winner_metrics,
            "config": vars(args),
        }
        final_path = args.working_dir / f"metrics_final_{args.tag}.json"
        final_path.write_text(json.dumps(final_summary, indent=2, default=str))
        print_main("\n" + "=" * 60)
        print_main(f"WINNER: {winner_strategy}")
        print_main(f"F2@10:     {winner_metrics.get('f2@10', 0):.4f}")
        print_main(f"Recall@10: {winner_metrics.get('recall@10', 0):.4f}")
        print_main(f"MRR:       {winner_metrics.get('mrr', 0):.4f}")
        print_main(f"Saved:     {final_path}")
        print_main("=" * 60)

        if winner_dir is not None:
            zip_path = args.working_dir / f"checkpoint_{args.tag}_winner"
            shutil.make_archive(str(zip_path), "zip", winner_dir)
            print_main(f"Backup zip: {zip_path}.zip")

        maybe_push_winner(args, winner_strategy, winner_dir, device)

    barrier()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
