import os
import json
import math
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
    BitsAndBytesConfig,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


def load_from_arrays(questions_path: str, answers_path: str, is_numpy: bool) -> Tuple[List[str], List[str], List[str]]:
    if is_numpy:
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not installed. Install numpy or provide JSON lists.")
        qs = np.load(questions_path, allow_pickle=True).tolist()
        ans = np.load(answers_path, allow_pickle=True).tolist()
        ids = [str(i) for i in range(len(qs))]
    else:
        with open(questions_path, "r", encoding="utf-8") as f:
            qs = json.load(f)
        with open(answers_path, "r", encoding="utf-8") as f:
            ans = json.load(f)
        ids = [str(i) for i in range(len(qs))]
    if len(qs) != len(ans):
        raise ValueError(f"Mismatched lengths: {len(qs)} questions vs {len(ans)} answers.")
    return qs, ans, ids


def load_from_object_list(qa_json_path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(qa_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("qa_json_path must contain a JSON list of objects with fields: id (optional), question, answer.")
    qs, ans, ids = [], [], []
    for i, row in enumerate(data):
        q = row.get("question")
        a = row.get("answer")
        if q is None or a is None:
            raise ValueError("Each object must have 'question' and 'answer' fields.")
        qs.append(str(q))
        ans.append(str(a))
        ids.append(str(row.get("id", i)))
    return qs, ans, ids


def train_val_split(qs: List[str], ans: List[str], ids: List[str], val_split: float, seed: int = 42):
    idx = list(range(len(qs)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(len(idx) * val_split)
    val_idx = set(idx[:n_val])
    train_qs, train_ans, train_ids = [], [], []
    val_qs, val_ans, val_ids = [], [], []
    for i in range(len(qs)):
        if i in val_idx:
            val_qs.append(qs[i]); val_ans.append(ans[i]); val_ids.append(ids[i])
        else:
            train_qs.append(qs[i]); train_ans.append(ans[i]); train_ids.append(ids[i])
    return (train_qs, train_ans, train_ids), (val_qs, val_ans, val_ids)


class QADataset(Dataset):
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        ids: Optional[List[str]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        add_eos_token: bool = True,
        use_chat_template: bool = True,
    ):
        self.questions = questions
        self.answers = answers
        self.ids = ids if ids is not None else [str(i) for i in range(len(questions))]
        self.tok = tokenizer
        self.max_len = max_seq_length
        self.add_eos = add_eos_token
        self.use_chat_template = use_chat_template and hasattr(tokenizer, "apply_chat_template")

        if self.add_eos and self.tok.eos_token_id is None:
            self.tok.add_special_tokens({"eos_token": "</s>"})

    def __len__(self):
        return len(self.questions)

    def _build_prompt_ids(self, question: str):
        if self.use_chat_template:
            messages = [{"role": "user", "content": question}]
            prompt_ids = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            return prompt_ids
        else:
            prompt = f"User: {question}\nAssistant:"
            return self.tok.encode(prompt, add_special_tokens=True)

    def _build_answer_ids(self, answer: str):
        # Normalize minor trailing punctuation like "8, " in your sample; model can still learn exact outputs.
        text = answer.strip()
        ids = self.tok.encode(text, add_special_tokens=False)
        if self.add_eos:
            ids += [self.tok.eos_token_id]
        return ids

    def __getitem__(self, idx):
        q = self.questions[idx]
        a = self.answers[idx]

        prompt_ids = self._build_prompt_ids(q)
        answer_ids = self._build_answer_ids(a)

        max_answer_len = max(0, self.max_len - len(prompt_ids))
        if len(answer_ids) > max_answer_len:
            answer_ids = answer_ids[:max_answer_len]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids), "id": self.ids[idx]}


class PackedQADataset(Dataset):
    """
    Packs many short QA samples into near-max length sequences to improve throughput.
    Precomputes packed samples for stable performance.
    """
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        ids: Optional[List[str]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        add_eos_token: bool = True,
        use_chat_template: bool = True,
    ):
        self.tok = tokenizer
        self.max_len = max_seq_length
        self.add_eos = add_eos_token
        self.use_chat_template = use_chat_template and hasattr(tokenizer, "apply_chat_template")
        self.samples: List[Dict[str, Any]] = []

        if self.add_eos and self.tok.eos_token_id is None:
            self.tok.add_special_tokens({"eos_token": "</s>"})

        # Pre-tokenize all pairs
        pair_tokens: List[Tuple[List[int], List[int], str]] = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            pid = self._build_prompt_ids(q)
            aid = self._build_answer_ids(a)
            # truncate answer if needed
            max_answer_len = max(0, self.max_len - len(pid))
            if len(aid) > max_answer_len:
                aid = aid[:max_answer_len]
            pair_tokens.append((pid, aid, str(ids[i]) if ids else str(i)))

        # Pack greedily
        cur_input, cur_labels, cur_ids = [], [], []
        for (pid, aid, _id) in pair_tokens:
            piece_len = len(pid) + len(aid)
            if piece_len > self.max_len:
                # skip pathological case (shouldn't happen due to trunc above)
                continue
            if len(cur_input) + piece_len <= self.max_len:
                cur_input.extend(pid + aid)
                cur_labels.extend([-100] * len(pid) + aid)
                cur_ids.append(_id)
            else:
                self.samples.append({
                    "input_ids": cur_input,
                    "labels": cur_labels,
                    "attention_mask": [1] * len(cur_input),
                    "ids": cur_ids,
                })
                cur_input = pid + aid
                cur_labels = [-100] * len(pid) + aid
                cur_ids = [_id]
        if cur_input:
            self.samples.append({
                "input_ids": cur_input,
                "labels": cur_labels,
                "attention_mask": [1] * len(cur_input),
                "ids": cur_ids,
            })

    def _build_prompt_ids(self, question: str):
        if self.use_chat_template:
            messages = [{"role": "user", "content": question}]
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
        else:
            prompt = f"User: {question}\nAssistant:"
            return self.tok.encode(prompt, add_special_tokens=True)

    def _build_answer_ids(self, answer: str):
        text = answer.strip()
        ids = self.tok.encode(text, add_special_tokens=False)
        if self.add_eos:
            ids += [self.tok.eos_token_id]
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "labels": s["labels"],
            "attention_mask": s["attention_mask"],
            "ids": s["ids"],  # packed ids list (not used by Trainer)
        }


def get_model_and_tokenizer(
    model_name: str,
    use_qlora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]],
    bf16: bool,
    flash_attn: bool,
):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.padding_side = "right"
    tok.truncation_side = "right"

    attn_impl = "flash_attention_2" if flash_attn else None

    quant_config = None
    if use_qlora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT is required for QLoRA. Install peft.")
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            raise RuntimeError("bitsandbytes is required for QLoRA. Install bitsandbytes.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    model_kwargs = {}
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        trust_remote_code=True,
        **model_kwargs,
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune: trainable params {trainable:,}/{total:,}")

    model.config.use_cache = False
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    # Data sources
    parser.add_argument("--qa_json_path", type=str, default=None, help="Path to a JSON list of {id?, question, answer}")
    parser.add_argument("--questions_path", type=str, default=None, help="Path to JSON/NPY array of questions")
    parser.add_argument("--answers_path", type=str, default=None, help="Path to JSON/NPY array of answers")
    parser.add_argument("--is_numpy", action="store_true", help="Set if questions/answers are .npy arrays.")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--use_qlora", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="")  # comma-separated

    parser.add_argument("--pack_samples", type=lambda x: str(x).lower() == "true", default=True, help="Pack short samples")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--val_split", type=float, default=0.02)

    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--flash_attention", type=lambda x: str(x).lower() == "true", default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON (optional).")

    args = parser.parse_args()
    set_seed(args.seed)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()] if args.target_modules else None

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bf16=args.bf16,
        flash_attn=args.flash_attention,
    )

    # Load data
    if args.qa_json_path:
        questions, answers, ids = load_from_object_list(args.qa_json_path)
    elif args.questions_path and args.answers_path:
        questions, answers, ids = load_from_arrays(args.questions_path, args.answers_path, args.is_numpy)
    else:
        raise ValueError("Provide either --qa_json_path OR both --questions_path and --answers_path")

    if args.val_split > 0.0:
        (train_q, train_a, train_ids), (val_q, val_a, val_ids) = train_val_split(questions, answers, ids, args.val_split, seed=args.seed)
        if args.pack_samples:
            train_ds = PackedQADataset(train_q, train_a, train_ids, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=True, use_chat_template=True)
            eval_ds = QADataset(val_q, val_a, val_ids, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=True, use_chat_template=True)
        else:
            train_ds = QADataset(train_q, train_a, train_ids, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=True, use_chat_template=True)
            eval_ds = QADataset(val_q, val_a, val_ids, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=True, use_chat_template=True)
    else:
        if args.pack_samples:
            train_ds = PackedQADataset(questions, answers, ids, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=True, use_chat_template=True)
        else:
            train_ds = QADataset(questions, answers, ids, tokenizer, max_seq_length=args.max_seq_length, add_eos_token=True, use_chat_template=True)
        eval_ds = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=min(8, os.cpu_count() or 4),
        dataloader_pin_memory=True,
        torch_compile=False,
        deepspeed=args.deepspeed,  # Activates DeepSpeed
        report_to="none",
        ddp_find_unused_parameters=False,
        group_by_length=False,  # packing already helps; set True if you disable packing
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()