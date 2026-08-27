import json
import os
import re
import time
from pathlib import Path

import wandb

from math_verify import parse, verify

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback

from trl import (
    GRPOTrainer,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with GRPO-specific options."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name) to log runs under."},
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "WandB project name to log runs under."},
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable Qwen3 thinking mode while generating training rollouts."},
    )
    selected_checkpoint_steps: str = field(
        default="",
        metadata={"help": "Comma-separated optimizer steps at which checkpoints are forced."},
    )
    stop_after_step: int = field(
        default=0,
        metadata={"help": "Stop cleanly after this optimizer step without changing the scheduler horizon."},
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Resume from the highest complete checkpoint in output_dir."},
    )
    save_final_model: bool = field(
        default=True,
        metadata={"help": "Save an additional final adapter directly in output_dir."},
    )


class SelectedCheckpointCallback(TrainerCallback):
    def __init__(self, steps: set[int]):
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = control.should_save or state.global_step in self.steps
        return control


class StopAfterStepCallback(TrainerCallback):
    def __init__(self, stop_after_step: int):
        self.stop_after_step = int(stop_after_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after_step > 0 and state.global_step >= self.stop_after_step:
            control.should_training_stop = True
        return control


class JsonlMetricsCallback(TrainerCallback):
    """Persist numeric Trainer logs so checkpoint cleanup does not erase curves."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.step_started_at = None
        self.step_seconds = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started_at = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started_at is not None:
            self.step_seconds = time.perf_counter() - self.step_started_at

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or state.global_step <= 0:
            return
        data = {
            str(key): value
            for key, value in (logs or {}).items()
            if isinstance(value, (bool, int, float))
        }
        if self.step_seconds is not None:
            data["timing_s/step"] = self.step_seconds
        record = {
            "step": int(state.global_step),
            "timestamp": time.time(),
            "data": data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)


def _parse_checkpoint_steps(raw_steps: str) -> set[int]:
    steps = {int(part.strip()) for part in raw_steps.split(",") if part.strip()}
    if any(step <= 0 for step in steps):
        raise ValueError(f"selected_checkpoint_steps must contain positive integers: {raw_steps}")
    return steps


def _checkpoint_is_complete(checkpoint: Path, step: int) -> bool:
    trainer_state = checkpoint / "trainer_state.json"
    if not trainer_state.is_file():
        return False
    try:
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        saved_step = int(state.get("global_step", -1))
    except (TypeError, ValueError):
        return False
    if saved_step != step:
        return False

    model_saved = any(
        (checkpoint / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        )
    )
    optimizer_saved = (checkpoint / "optimizer.pt").is_file() or any(checkpoint.glob("global_step*"))
    return model_saved and optimizer_saved


def _latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = []
    for checkpoint in Path(output_dir).glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if match is None:
            continue
        step = int(match.group(1))
        if _checkpoint_is_complete(checkpoint, step):
            checkpoints.append((step, checkpoint))
    return str(max(checkpoints)[1]) if checkpoints else None


def _uses_wandb(report_to) -> bool:
    if isinstance(report_to, str):
        targets = {part.strip().lower() for part in report_to.split(",")}
    else:
        targets = {str(part).lower() for part in (report_to or [])}
    return "wandb" in targets


def extract_boxed_answer(text):
    """
    Extract the answer from \\boxed{} format.
    For thinking models, only searches after </think> to avoid picking up
    intermediate answers from the thinking block.
    Handles nested braces correctly (e.g. \\boxed{\\frac{1}{2}}).
    """
    # For thinking models (e.g. Qwen3), only look after </think>
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    """Convert ratio notation a:b → \\frac{a}{b} so math_verify can parse it."""
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def reward_correctness(completions, Answer, **kwargs):
    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, Answer)):
        pred_answer = extract_boxed_answer(completion)

        reward = 0.0

        # Try math_verify for mathematical equivalence (handles fractions, algebra, etc.)
        # Only use it when both sides actually parse to something (avoids silent None returns
        # for MCQ answers like "E" which parse() returns None for)
        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass

        # Fallback: whitespace-stripped string match (handles MCQ like "E", "A", etc.)
        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)

    return rewards


def make_format_prompt(tokenizer, enable_thinking=False):
    """
    Returns a formatting function that applies the tokenizer's chat template.
    """

    def format_prompt(example):
        question = example.get("Question") or example.get("problem") or example.get("prompt")
        answer = example.get("Answer") or example.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Math record is missing a non-empty question: {sorted(example)}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Math record is missing a non-empty answer: {sorted(example)}")

        question = question.strip()
        if "put your final answer within \\boxed{}" not in question:
            question = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [
            {
                "role": "user",
                "content": question,
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return {"prompt": prompt, "Answer": answer.strip()}

    return format_prompt


def load_math_dataset(dataset_name: str):
    if not dataset_name:
        raise ValueError("--dataset_name must identify the exact local Math dataset or Hugging Face dataset")

    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.is_file():
        suffix = dataset_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(dataset_path), split="train")
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(dataset_path), split="train")
        raise ValueError(f"Unsupported local dataset format: {dataset_path}")

    dataset = load_dataset(dataset_name)
    if hasattr(dataset, "keys") and "train" in dataset:
        return dataset["train"]
    return dataset


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    checkpoint_steps = _parse_checkpoint_steps(script_args.selected_checkpoint_steps)

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-5 -> "2e-5")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_name = (
            f"GRPO_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Prompt Length: {training_args.max_prompt_length}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    use_wandb = _uses_wandb(training_args.report_to)
    if use_wandb and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_prompt_length": training_args.max_prompt_length,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "enable_thinking": script_args.enable_thinking,
                "selected_checkpoint_steps": sorted(checkpoint_steps),
                "stop_after_step": script_args.stop_after_step,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    train_dataset = load_math_dataset(script_args.dataset_name)

    # Apply the format_prompt function to create the expected structure
    format_prompt = make_format_prompt(tokenizer, enable_thinking=script_args.enable_thinking)
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    print(f"Training dataset: {script_args.dataset_name} ({len(train_dataset)} rows)")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if checkpoint_steps:
        trainer.add_callback(SelectedCheckpointCallback(checkpoint_steps))
    if script_args.stop_after_step > 0:
        trainer.add_callback(StopAfterStepCallback(script_args.stop_after_step))
    metrics_jsonl = os.environ.get("SDPO_METRICS_JSONL", "").strip()
    if metrics_jsonl:
        trainer.add_callback(JsonlMetricsCallback(metrics_jsonl))

    resume_from_checkpoint = _latest_checkpoint(training_args.output_dir) if script_args.auto_resume else None
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if script_args.save_final_model:
            trainer.save_model(training_args.output_dir)
    finally:
        if use_wandb and os.environ.get("LOCAL_RANK", "0") == "0" and wandb.run is not None:
            wandb.finish()
