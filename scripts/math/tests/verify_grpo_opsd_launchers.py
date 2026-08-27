#!/usr/bin/env python3
import json
import runpy
import sys
import tempfile
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def require(path: str, *snippets: str) -> str:
    source = (REPO / path).read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in source:
            raise AssertionError(f"{path}: missing protocol invariant: {snippet!r}")
    return source


def verify_launchers() -> None:
    wrappers = {
        "scripts/math/run_grpo_4b.sh": "grpo_opsd_trl_aligned/a800_grpo_4b.sh",
        "scripts/math/run_grpo_8b.sh": "grpo_opsd_trl_aligned/h200_grpo_8b.sh",
        "scripts/math/train_eval5_n16_a800_4b/a800_machine1_grpo.sh": (
            "grpo_opsd_trl_aligned/a800_grpo_4b.sh"
        ),
        "scripts/math/train_eval5_n16_h200/h200_machine1_grpo.sh": (
            "grpo_opsd_trl_aligned/h200_grpo_8b.sh"
        ),
    }
    for path, target in wrappers.items():
        source = require(path, target)
        if "run_verl_method" in source:
            raise AssertionError(f"{path}: native VERL GRPO is still reachable")

    require(
        "scripts/math/grpo_opsd_trl_aligned/a800_grpo_4b.sh",
        "MODEL_SIZE=4b",
        "HARDWARE=a800",
        "ENABLE_ADAPTIVE_GPU_KEEPALIVE",
        "math_grpo_4b_opsd_trl_aligned_eval5_n16_a800_20260827",
    )
    require(
        "scripts/math/grpo_opsd_trl_aligned/h200_grpo_8b.sh",
        "MODEL_SIZE=8b",
        "HARDWARE=h200",
        "math_grpo_8b_opsd_trl_aligned_eval5_n16_h200_20260827",
    )

    runner = require(
        "scripts/math/grpo_opsd_trl_aligned/run_grpo_opsd_trl_aligned.sh",
        'OPSD_CODE_ROOT="${OPSD_CODE_ROOT:-${REPO}/OPSD}"',
        'MAX_STEPS="${MAX_STEPS:-100}"',
        'SCHEDULER_HORIZON_STEPS="${SCHEDULER_HORIZON_STEPS:-420}"',
        'EVAL_FREQUENCY="${EVAL_FREQUENCY:-5}"',
        'VAL_N="${VAL_N:-16}"',
        "GRADIENT_ACCUMULATION_STEPS=8",
        "UNIQUE_PROMPTS_PER_STEP=8",
        "NUM_GENERATIONS=8",
        "TRAJECTORIES_PER_STEP=64",
        "NUM_ITERATIONS=1",
        "GRPO_EPSILON=0.2",
        "ENTROPY_COEFFICIENT=0",
        "LOSS_TYPE=dapo",
        "SCALE_REWARDS=group",
        "IMPORTANCE_SAMPLING_LEVEL=token",
        "VLLM_IS_MODE=token_mask",
        "VLLM_IS_CAP=3.0",
        "MAX_PROMPT_LENGTH=2048",
        "MAX_COMPLETION_LENGTH=16384",
        "ROLLOUT_TEMPERATURE=0.7",
        "ROLLOUT_TOP_P=0.95",
        "ROLLOUT_TOP_K=20",
        "LORA_DROPOUT=0.0",
        "EVAL_MAX_NEW_TOKENS=16384",
        'EVAL_SUBMISSION_MODE="${EVAL_SUBMISSION_MODE:-legacy_all_prompts}"',
        "--lr_scheduler_type linear",
        "--warmup_steps 0",
        "--weight_decay 0",
        "--max_grad_norm 0.1",
        "--num_generations",
        "--steps_per_generation",
        "--num_iterations",
        "--epsilon_high",
        "--vllm_importance_sampling_correction true",
        "--vllm_importance_sampling_mode",
        "--vllm_importance_sampling_cap",
        "--lora_r 64",
        "--lora_alpha 128",
        '--lora_dropout "${LORA_DROPOUT}"',
        "--selected_checkpoint_steps",
        "--stop_after_step",
        "--auto_resume true",
        "--save_final_model false",
        "--report_to none",
        "JsonlMetricsCallback",
        "training_metrics_jsonl=${METRICS_JSONL}",
        "evaluation_datasets=aime24,aime25,hmmt25,amc23,minerva",
        "entropy_coefficient=${ENTROPY_COEFFICIENT}",
        "entropy_coefficient_note=historical_OPSD_TRL_GRPO_has_no_entropy_bonus",
        "lora_dropout=${LORA_DROPOUT}",
        "remove_checkpoint \"${MAX_STEPS}\"",
        "lock_protocol_file",
    )
    for snippet in (
        "math_grpo_4b_native_verl",
        "math_grpo_8b_native_verl",
        "run_verl_method_h200.sh\" grpo",
        "run_verl_method_a800_4b.sh\" grpo",
    ):
        if snippet in runner:
            raise AssertionError(f"aligned runner contains stale native invariant: {snippet}")


def load_grpo_module() -> dict:
    old_modules = dict(sys.modules)
    try:
        wandb = types.ModuleType("wandb")
        wandb.run = None
        wandb.init = lambda **kwargs: None
        wandb.finish = lambda: None
        sys.modules["wandb"] = wandb

        math_verify = types.ModuleType("math_verify")
        math_verify.parse = lambda value: value
        math_verify.verify = lambda left, right: left == right
        sys.modules["math_verify"] = math_verify

        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: (args, kwargs)
        sys.modules["datasets"] = datasets

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = type("AutoTokenizer", (), {})
        transformers.TrainerCallback = type("TrainerCallback", (), {})
        sys.modules["transformers"] = transformers

        trl = types.ModuleType("trl")
        for name in ("GRPOTrainer", "GRPOConfig", "ModelConfig", "ScriptArguments", "TrlParser"):
            setattr(trl, name, type(name, (), {}))
        trl.get_kbit_device_map = lambda: None
        trl.get_peft_config = lambda args: None
        trl.get_quantization_config = lambda args: None
        sys.modules["trl"] = trl

        return runpy.run_path(str(REPO / "OPSD/grpo_train.py"), run_name="grpo_train_test")
    finally:
        sys.modules.clear()
        sys.modules.update(old_modules)


def verify_trainer_helpers() -> None:
    module = load_grpo_module()
    parse_steps = module["_parse_checkpoint_steps"]
    latest_checkpoint = module["_latest_checkpoint"]
    make_format_prompt = module["make_format_prompt"]
    metrics_callback = module["JsonlMetricsCallback"]

    assert parse_steps("5,10, 15") == {5, 10, 15}
    try:
        parse_steps("0,5")
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive checkpoint step was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for step in (5, 10):
            checkpoint = root / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": step}), encoding="utf-8"
            )
            (checkpoint / "adapter_model.safetensors").touch()
            (checkpoint / f"global_step{step}").mkdir()
        incomplete = root / "checkpoint-15"
        incomplete.mkdir()
        (incomplete / "trainer_state.json").write_text(
            json.dumps({"global_step": 15}), encoding="utf-8"
        )
        assert latest_checkpoint(str(root)) == str(root / "checkpoint-10")

    class Tokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return messages[0]["content"]

    tokenizer = Tokenizer()
    formatter = make_format_prompt(tokenizer, enable_thinking=False)
    formatted = formatter({"problem": "Compute 1+1.", "answer": "2"})
    assert formatted["Answer"] == "2"
    assert formatted["prompt"].endswith(
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }

    with tempfile.TemporaryDirectory() as directory:
        metrics_path = Path(directory) / "metrics.jsonl"
        callback = metrics_callback(str(metrics_path))
        callback.step_started_at = 1.0
        callback.on_step_end(None, None, None)
        state = types.SimpleNamespace(is_world_process_zero=True, global_step=5)
        callback.on_log(None, state, None, {"loss": 0.25, "message": "ignored"})
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["step"] == 5
        assert payload["data"]["loss"] == 0.25
        assert "message" not in payload["data"]
        assert payload["data"]["timing_s/step"] >= 0


def verify_training_data() -> None:
    opsd_path = REPO / "OPSD/data/math/train.jsonl"
    sdpo_path = REPO / "SDPO/datasets/math_probs/train.json"
    opsd = [json.loads(line) for line in opsd_path.read_text(encoding="utf-8").splitlines() if line]
    sdpo = [json.loads(line) for line in sdpo_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(opsd) == len(sdpo) == 758
    for index, (left, right) in enumerate(zip(opsd, sdpo)):
        if left["problem"] != right["description"] or left["answer"] != right["answer"]:
            raise AssertionError(f"Math representations diverge at row {index}")


def main() -> None:
    verify_launchers()
    verify_trainer_helpers()
    verify_training_data()
    print("Aligned OPSD/TRL GRPO protocol and resume invariants: PASS")


if __name__ == "__main__":
    main()
