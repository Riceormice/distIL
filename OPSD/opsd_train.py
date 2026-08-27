import os
import json as _json
import re
from pathlib import Path

import torch as _torch
_orig_default = _json.JSONEncoder.default
def _patched_default(self, obj):
    if isinstance(obj, _torch.dtype): return str(obj)
    return _orig_default(self, obj)
_json.JSONEncoder.default = _patched_default
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, TrainerCallback

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    loss_mode: str = field(
        default="jsd",
        metadata={
            "help": "Loss mode: 'distil', 'jsd' (OPSD), 'sdpo' (reverse KL), or "
            "'sr_opsd' (EMA/reference forward Renyi)."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    renyi_rho: float = field(
        default=0.95,
        metadata={"help": "Renyi order for loss_mode=sr_opsd; must be positive and different from 1."},
    )
    reference_teacher_weight: float = field(
        default=0.9,
        metadata={
            "help": "Weight of the EMA privileged teacher in the normalized geometric target. "
            "Use 1.0 for the no-reference ablation."
        },
    )
    renyi_token_clip: float = field(
        default=0.0,
        metadata={"help": "Optional per-token SR-OPSD divergence clip. Zero disables clipping."},
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )
    grouped_rollouts_per_prompt: int = field(
        default=1,
        metadata={
            "help": "Number of independent rollouts for each prompt inside one optimizer step. "
            "Values above one enable grouped-repeat sampling."
        },
    )
    grouped_unique_prompts_per_step: int = field(
        default=0,
        metadata={
            "help": "Unique prompts per optimizer step in grouped mode. Zero derives it from world size "
            "and per-device batch size."
        },
    )
    selected_checkpoint_steps: str = field(
        default="",
        metadata={"help": "Comma-separated optimizer steps at which checkpoints are forced."},
    )
    stop_after_step: int = field(
        default=0,
        metadata={
            "help": "Stop cleanly after this optimizer step without changing the learning-rate horizon."
        },
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
        control.should_save = state.global_step in self.steps
        return control


class StopAfterStepCallback(TrainerCallback):
    def __init__(self, stop_after_step: int):
        self.stop_after_step = int(stop_after_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after_step > 0 and state.global_step >= self.stop_after_step:
            control.should_training_stop = True
        return control


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
        state = _json.loads(trainer_state.read_text(encoding="utf-8"))
        saved_step = int(state.get("global_step", -1))
    except (OSError, TypeError, ValueError):
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
    optimizer_saved = (checkpoint / "optimizer.pt").is_file() or any(
        checkpoint.glob("global_step*")
    )
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


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # sdpo mode uses reverse KL (beta=1)
    if script_args.loss_mode == "sdpo":
        training_args.beta = 1.0

    if script_args.loss_mode == "sr_opsd":
        if not model_args.use_peft:
            raise ValueError("loss_mode=sr_opsd requires --use_peft")
        if not script_args.use_ema_teacher:
            raise ValueError("loss_mode=sr_opsd requires --use_ema_teacher")
        if script_args.fixed_teacher:
            raise ValueError("loss_mode=sr_opsd cannot be combined with --fixed_teacher")

    checkpoint_steps = _parse_checkpoint_steps(script_args.selected_checkpoint_steps)

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "renyi_rho": script_args.renyi_rho if script_args.loss_mode == "sr_opsd" else None,
                "reference_teacher_weight": (
                    script_args.reference_teacher_weight if script_args.loss_mode == "sr_opsd" else None
                ),
                "grouped_rollouts_per_prompt": script_args.grouped_rollouts_per_prompt,
                "grouped_unique_prompts_per_step": script_args.grouped_unique_prompts_per_step or None,
                "selected_checkpoint_steps": sorted(checkpoint_steps),
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
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
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

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
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    dataset_name = script_args.dataset_name or "siyanzhao/Openthoughts_math_30k_opsd"
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.is_file() and dataset_path.suffix in {".json", ".jsonl"}:
        train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    else:
        dataset = load_dataset(dataset_name)
        train_dataset = dataset["train"] if hasattr(dataset, "keys") else dataset

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        loss_mode=script_args.loss_mode,
        ema_decay=script_args.ema_decay,
        renyi_rho=script_args.renyi_rho,
        reference_teacher_weight=script_args.reference_teacher_weight,
        renyi_token_clip=script_args.renyi_token_clip if script_args.renyi_token_clip > 0 else None,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        grouped_rollouts_per_prompt=script_args.grouped_rollouts_per_prompt,
        grouped_unique_prompts_per_step=script_args.grouped_unique_prompts_per_step,
    )

    if checkpoint_steps:
        trainer.add_callback(SelectedCheckpointCallback(checkpoint_steps))
    if script_args.stop_after_step > 0:
        trainer.add_callback(StopAfterStepCallback(script_args.stop_after_step))

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    resume_from_checkpoint = _latest_checkpoint(training_args.output_dir) if script_args.auto_resume else None
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if script_args.save_final_model:
            trainer.save_model(training_args.output_dir)
    finally:
        if os.environ.get("LOCAL_RANK", "0") == "0" and wandb.run is not None:
            wandb.finish()
