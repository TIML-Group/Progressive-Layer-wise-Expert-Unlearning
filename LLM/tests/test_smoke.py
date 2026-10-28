"""CPU smoke tests on tiny randomly initialized Qwen2-MoE and DeepSeek-V2 models.

    cd LLM && DEEPSEEK_CODE=/path/with/modeling_deepseek.py python -m pytest tests -q

The DeepSeek test needs the remote-code files (``modeling_deepseek.py`` and
``configuration_deepseek.py``) of DeepSeek-V2-Lite in ``$DEEPSEEK_CODE``.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

LLM_DIR = Path(__file__).resolve().parent.parent
WORDS = [f"w{i}" for i in range(100)]


def tiny_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(["<pad>", "<eos>", "<unk>", "A", "B", "C", "D"] + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.save_pretrained(path)
    return len(vocab)


def build_qwen(path):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    vocab = tiny_tokenizer(path)
    config = Qwen2MoeConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                            shared_expert_intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                            max_position_embeddings=128, pad_token_id=0, eos_token_id=1)
    torch.manual_seed(0)
    Qwen2MoeForCausalLM(config).save_pretrained(path)


def build_deepseek(path, code_dir):
    from transformers import AutoConfig, AutoModelForCausalLM

    from pleu_llm.modeling import patch_transformers_compat

    vocab = tiny_tokenizer(path)
    for name in ("modeling_deepseek.py", "configuration_deepseek.py"):
        shutil.copy(code_dir / name, path / name)
    config = dict(model_type="deepseek_v2", vocab_size=vocab, hidden_size=32, intermediate_size=64,
                  moe_intermediate_size=16, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                  n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=1,
                  moe_layer_freq=1, q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8, v_head_dim=16,
                  qk_nope_head_dim=8, topk_method="greedy", n_group=1, topk_group=1, scoring_func="softmax",
                  norm_topk_prob=False, routed_scaling_factor=1.0, aux_loss_alpha=0.001, seq_aux=True,
                  max_position_embeddings=128, rope_scaling=None, pad_token_id=0, bos_token_id=1, eos_token_id=1,
                  auto_map={"AutoConfig": "configuration_deepseek.DeepseekV2Config",
                            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"})
    (path / "config.json").write_text(json.dumps(config))
    patch_transformers_compat()
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).save_pretrained(path)
    for name in ("modeling_deepseek.py", "configuration_deepseek.py"):
        shutil.copy(code_dir / name, path / name)


def write_texts(path, seed, count=12, length=24):
    import random

    rng = random.Random(seed)
    with open(path, "w") as handle:
        for _ in range(count):
            handle.write(json.dumps({"text": " ".join(rng.choice(WORDS) for _ in range(length))}) + "\n")


def run(script, model_dir, tmp, *extra):
    command = [sys.executable, str(LLM_DIR / script), "--model", str(model_dir), "--forget-dataset", "wmdp",
               "--forget-file", str(tmp / "forget.jsonl"), "--retain-file", str(tmp / "retain.jsonl"),
               "--output-root", str(tmp / "out"), "--device-map", "cpu", "--batch-size", "2",
               "--grad-accum-steps", "1", "--cutoff-len", "32", "--skip-eval", *extra]
    subprocess.run(command, cwd=LLM_DIR, check=True, env={**os.environ, "TQDM_DISABLE": "1"})


def pipeline(model_dir, tmp):
    write_texts(tmp / "forget.jsonl", 0)
    write_texts(tmp / "retain.jsonl", 1)
    write_texts(tmp / "relearn.jsonl", 2)
    run("unlearn_baseline.py", model_dir, tmp, "--method", "neggrad_plus", "--gradient-checkpointing")
    run("unlearn_baseline.py", model_dir, tmp, "--method", "ga")
    run("unlearn_baseline.py", model_dir, tmp, "--method", "ga", "--seuf")
    run("unlearn_baseline.py", model_dir, tmp, "--method", "neggrad_plus", "--seuf", "--seuf-layerwise")
    curve = ["--holdout", "2", "--t-grid", "0.5,1.0"]
    run("unlearn_pleu.py", model_dir, tmp, *curve, "--unassigned", "last")
    for ablation in ("no_lteu", "no_idff", "no_sepo"):
        run("unlearn_pleu.py", model_dir, tmp, *curve, "--ablation", ablation)
    run("unlearn_mcu.py", model_dir, tmp, *curve, "--mask-max-batches", "2")
    run("relearn.py", model_dir, tmp, "--method", "pleu", "--steps", "2", "--relearn-data", str(tmp / "relearn.jsonl"))
    run("relearn.py", model_dir, tmp, "--method", "mcu", "--steps", "2", "--lora", "--relearn-data",
        str(tmp / "relearn.jsonl"))

    runs = tmp / "out" / model_dir.name / "wmdp"
    pleu = json.loads((runs / "pleu_s123" / "results.json").read_text())
    assert pleu["layers"], "PLEU must report per-layer statistics"
    assert (runs / "pleu_s123" / "relearn_results.json").exists()
    from safetensors import safe_open

    with safe_open(str(runs / "pleu_s123" / "delta.safetensors"), "pt") as handle:
        keys = list(handle.keys())
    assert keys and all(".mlp.experts." in k or ".mlp.gate." in k for k in keys)
    with safe_open(str(runs / "neggrad_plus_s123" / "delta.safetensors"), "pt") as handle:
        full = set(handle.keys())
    assert any("self_attn" in k for k in full) and any("embed_tokens" in k for k in full), "full-model NegGrad+"


def test_scorer_and_curve(tmp_path):
    from pleu_llm.curve import ParamCurve
    from pleu_llm.evaluation import Scorer
    from pleu_llm.modeling import load_model_and_tokenizer, moe_layers

    build_qwen(tmp_path)
    model, tokenizer = load_model_and_tokenizer(str(tmp_path), device_map="cpu")
    scorer = Scorer(model, tokenizer)
    assert 0 <= scorer.choose("w1 w2 A B C D", 4) < 4
    assert isinstance(scorer.generate("w1 w2", 3), str)

    layer = moe_layers(model)[0]
    name = f"{layer.prefix}experts.0.up_proj.weight"
    weight = dict(model.named_parameters())[name]
    original = weight.detach().clone()
    end = original + 1.0
    mask = torch.zeros_like(original, dtype=torch.bool)
    mask[0] = True
    curve = ParamCurve(model, {name: end}, masks={name: mask})
    curve.bake(1.0)
    baked = dict(model.named_parameters())[name]
    assert torch.allclose(baked[0], end[0]) and torch.equal(baked[1:], original[1:])


@pytest.mark.slow
def test_qwen_pipeline(tmp_path):
    build_qwen(tmp_path / "qwen-tiny")
    pipeline(tmp_path / "qwen-tiny", tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("DEEPSEEK_CODE"), reason="set DEEPSEEK_CODE to the remote-code directory")
def test_deepseek_pipeline(tmp_path):
    build_deepseek(tmp_path / "deepseek-tiny", Path(os.environ["DEEPSEEK_CODE"]))
    pipeline(tmp_path / "deepseek-tiny", tmp_path)
