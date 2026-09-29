from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import (
    BEST_MODEL_DIR,
    CHECKPOINTS_DIR,
    CONTRADICTION_LABEL_ID,
    LOGS_DIR,
    MODEL_CONFIG,
    NLI_ID2LABEL,
)
from src.data.schema import Pair
from src.preprocess.text_processor import nli_text


def compute_classification_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_contradiction": float(
            f1_score(
                labels,
                preds,
                labels=[CONTRADICTION_LABEL_ID],
                average="macro",
                zero_division=0,
            )
        ),
    }


class ContradictionTrainer:
    def __init__(self, model_name: str = MODEL_CONFIG["base_model"], num_labels: int = MODEL_CONFIG["num_labels"]):
        self.model_name = model_name
        self.last_trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=NLI_ID2LABEL,
            label2id={"entailment": 0, "contradiction": 1, "neutral": 2},
        )

    def _encode_pairs(self, pairs: Sequence[Pair]) -> Dataset:
        records = {
            "text_a": [nli_text(p.text_a) for p in pairs],
            "text_b": [nli_text(p.text_b) for p in pairs],
            "labels": [int(p.label) for p in pairs],
        }
        dataset = Dataset.from_dict(records)

        def tokenize(batch):
            return self.tokenizer(
                batch["text_a"],
                batch["text_b"],
                truncation=True,
                padding="max_length",
                max_length=MODEL_CONFIG["max_length"],
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.remove_columns([c for c in ["text_a", "text_b"] if c in dataset.column_names])
        dataset.set_format("torch")
        return dataset

    def _training_args(self, output_dir: Path, do_eval: bool = True, **overrides) -> TrainingArguments:
        params = inspect.signature(TrainingArguments.__init__).parameters
        strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
        seed = MODEL_CONFIG.get("seed", 42)
        kwargs = {
            "output_dir": str(output_dir),
            "learning_rate": overrides.get("learning_rate", MODEL_CONFIG["learning_rate"]),
            "per_device_train_batch_size": overrides.get("batch_size", MODEL_CONFIG["batch_size"]),
            "per_device_eval_batch_size": overrides.get("batch_size", MODEL_CONFIG["batch_size"]),
            "num_train_epochs": overrides.get("num_epochs", MODEL_CONFIG["num_epochs"]),
            "weight_decay": MODEL_CONFIG["weight_decay"],
            "warmup_ratio": MODEL_CONFIG.get("warmup_ratio", 0.1),
            "logging_dir": str(LOGS_DIR),
            "load_best_model_at_end": do_eval,
            "metric_for_best_model": "accuracy",
            "greater_is_better": True,
            "save_total_limit": 2,
            "report_to": "none",
            strategy_key: "epoch" if do_eval else "no",
            "save_strategy": "epoch",
            "dataloader_pin_memory": False,
            "logging_steps": 50,
            "seed": seed,
            "data_seed": seed,
            "use_cpu": False,
        }
        if "fp16" in params:
            import torch

            kwargs["fp16"] = torch.cuda.is_available()
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        return TrainingArguments(**kwargs)

    def best_epoch(self) -> int:
        if self.last_trainer is None:
            return int(MODEL_CONFIG["num_epochs"])
        best_acc = -1.0
        epoch = 1
        for row in self.last_trainer.state.log_history:
            acc = row.get("eval_accuracy")
            if acc is None:
                continue
            if float(acc) > best_acc:
                best_acc = float(acc)
                epoch = max(1, int(round(float(row.get("epoch") or 1))))
        return epoch

    def train(
        self,
        train_pairs: Sequence[Pair],
        val_pairs: Optional[Sequence[Pair]] = None,
        output_dir: Path = BEST_MODEL_DIR,
        checkpoints_dir: Path = CHECKPOINTS_DIR,
        **overrides,
    ) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        do_eval = bool(val_pairs)
        train_ds = self._encode_pairs(train_pairs)
        val_ds = self._encode_pairs(val_pairs) if do_eval else None
        early_stopping = overrides.pop("early_stopping", do_eval)
        trainer_kwargs = {
            "model": self.model,
            "args": self._training_args(checkpoints_dir, do_eval=do_eval, **overrides),
            "train_dataset": train_ds,
            "eval_dataset": val_ds,
            "compute_metrics": compute_classification_metrics if do_eval else None,
            "callbacks": (
                [EarlyStoppingCallback(early_stopping_patience=MODEL_CONFIG["early_stopping_patience"])]
                if early_stopping
                else []
            ),
        }
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = self.tokenizer
        trainer = Trainer(**trainer_kwargs)
        trainer.train()
        self.last_trainer = trainer
        metrics = trainer.evaluate() if do_eval else {}
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        return metrics

    def evaluate(self, pairs: Sequence[Pair]) -> dict:
        dataset = self._encode_pairs(pairs)
        trainer_kwargs = {
            "model": self.model,
            "args": self._training_args(CHECKPOINTS_DIR / "eval"),
            "eval_dataset": dataset,
            "compute_metrics": compute_classification_metrics,
        }
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = self.tokenizer
        trainer = Trainer(**trainer_kwargs)
        return trainer.evaluate()

    def predict_logits(self, pairs: Sequence[Pair]) -> tuple[np.ndarray, np.ndarray]:
        dataset = self._encode_pairs(pairs)
        trainer_kwargs = {
            "model": self.model,
            "args": self._training_args(CHECKPOINTS_DIR / "predict"),
            "eval_dataset": dataset,
        }
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = self.tokenizer
        trainer = Trainer(**trainer_kwargs)
        output = trainer.predict(dataset)
        return np.asarray(output.predictions), np.asarray(output.label_ids)


def tune_hyperparameters(train_pairs: Sequence[Pair], val_pairs: Sequence[Pair], n_trials: int = 5) -> dict:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        trainer = ContradictionTrainer()
        metrics = trainer.train(
            train_pairs,
            val_pairs,
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            batch_size=trial.suggest_categorical("batch_size", [8, 16]),
            num_epochs=trial.suggest_int("num_epochs", 3, 7),
        )
        return float(metrics.get("eval_accuracy", metrics.get("eval_f1", 0.0)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=3600)
    return {"best_params": study.best_params, "best_value": study.best_value}
