# PLEU — language-model experiments

**DeepSeek-V2-Lite** (64 routed experts, top-6, 2 shared experts, 26 MoE layers) and
**Qwen1.5-MoE-A2.7B** (60 routed experts, top-4, 1 shared expert, 24 MoE layers), unlearning
**WMDP-Cyber** hazardous knowledge or **RWKU** real-world entities. The chat checkpoints
(`DeepSeek-V2-Lite-Chat`, `Qwen1.5-MoE-A2.7B-Chat`) are used.

## Files

| File | Purpose |
|---|---|
| `download_model.py` | fetch `deepseek` or `qwen` into the model root |
| `evaluate.py` | FE / UT of the original model or of an unlearned checkpoint |
| `unlearn_baseline.py` | GA and NegGrad+ on all parameters; `--seuf` trains only the selected expert(s) and router(s) (`--seuf-experts N`, `--seuf-layerwise`) |
| `unlearn_mcu.py` | MCU: Bézier curve on every weight towards the NegGrad+ model, MCU tensor mask |
| `unlearn_pleu.py` | PLEU; `--ablation no_lteu \| no_idff \| no_sepo` |
| `relearn.py` | 10 full-parameter fine-tuning steps on benign text (`--lora` for adapters), then FE / UT |
| `generate_relearn_data.py` | benign relearning corpus from any OpenAI-compatible endpoint, filtered against the forget set |
| `summarize.py` | Table 3 rows as mean ± std over seeds |
| `scripts/run_all.sh`, `scripts/run.sbatch` | the whole pipeline, locally or on Slurm |
| `pleu_llm/` | `modeling` (loading, MoE introspection, router hooks) · `data` · `evaluation` · `curve` (parametrized Bézier) · `masks` · `routing` · `training` · `common` (checkpoints, results) |

## Metrics

All methods and both models share [pleu_llm/evaluation.py](pleu_llm/evaluation.py):

* **UT** — MMLU accuracy on 247 subject-balanced test questions with up to 10 dev-set shots.
* **FE (WMDP)** — accuracy on the first 247 WMDP-Cyber questions (chance level 0.25). The prompt keeps
  the original instruction prefix (`--wmdp-instruction original`); `none` removes it.
* **FE (RWKU)** — ROUGE-L recall of 64-token greedy answers on the RWKU forget probes.

Multiple-choice answers are scored by the likelihood of the option letter after the chat prompt.

## Data and outputs

```
$PLEU_LLM_MODELS  (default: LLM/models)    DeepSeek-V2-Lite-Chat/, Qwen1.5-MoE-A2.7B-Chat/
$PLEU_LLM_DATA    (default: LLM/data)
├── RWKU/wiki_forget_set/*.json            [{"text": ...}]                 (SEUF release layout)
├── RWKU/forget_eval/*.json                [{"question": ..., "answer": ...}]
└── relearn/{wmdp,rwku}_benign.jsonl       {"text": ...} per line
$PLEU_LLM_OUTPUT  (default: LLM/outputs)/<model>/<forget>/<method>_s<seed>/
├── pleu_checkpoint.json                   base model path and run metadata
├── delta.safetensors                      changed tensors: all of them for GA / NegGrad+ / MCU,
│                                          target experts + routers for SEUF / PLEU
└── results.json, relearn_results.json
```

Training corpora: WMDP uses `cais/wmdp-corpora` (cyber forget / retain), RWKU uses the target
Wikipedia passages with WikiText-2 as retain data. MMLU, WMDP and WikiText are loaded from the
Hugging Face hub; `--forget-file` / `--retain-file` (JSONL with `text`) replace the training corpora
on offline nodes. Checkpoints are loaded with `load_any` (base model + `delta.safetensors`).

## Running

```bash
pip install -r requirements.txt
python download_model.py --model deepseek
python generate_relearn_data.py --forget-dataset wmdp --num-texts 200 --api-model <model>   # OPENAI_API_KEY
bash scripts/run_all.sh deepseek wmdp "123 124 125"        # every method + relearning
ABLATIONS=1 bash scripts/run_all.sh deepseek wmdp 123       # + Table 1 (SEUF with more experts) and Table 4
sbatch scripts/run.sbatch qwen rwku "123 124 125"           # same on one 4-GPU Slurm node
```

Step by step:

```bash
python evaluate.py          --model deepseek --forget-dataset wmdp                       # original model
python unlearn_baseline.py  --model deepseek --forget-dataset wmdp --method neggrad_plus # also the endpoint
python unlearn_baseline.py  --model deepseek --forget-dataset wmdp --method ga --seuf
python unlearn_mcu.py       --model deepseek --forget-dataset wmdp
python unlearn_pleu.py      --model deepseek --forget-dataset wmdp
python relearn.py           --model deepseek --forget-dataset wmdp --method pleu
python summarize.py         --model deepseek --forget-dataset wmdp
```

## GPU memory

| Step | Held on GPU | DeepSeek-V2-Lite (bf16 ≈ 31 GB) |
|---|---|---|
| GA / NegGrad+ / relearning | weights, gradients, AdamW states | ≈ 125 GB + activations → 2–4 GPUs |
| MCU | θ_o, θ_p, fp32 control point, gradients + momentum of the masked half, recomputed weights | ≈ 200 GB → 4 GPUs |
| SEUF, PLEU | model + the curve of one target expert at a time (≈ 60 MB) | ≈ 35 GB + activations → 1 GPU |

`--device-map auto` spreads a model over all visible GPUs. `--gradient-checkpointing` (baselines,
relearning) and `--mid-dtype bfloat16` (MCU) reduce memory further. PLEU reads only the target-expert
tensors of the NegGrad+ endpoint from its `delta.safetensors`, so it never loads a second model.

## Implementation notes

**PLEU** ([unlearn_pleu.py](unlearn_pleu.py))

1. *Localization on the original model, once, over the whole forget corpus.* Routing weights are the
   router softmax scores averaged over non-padding tokens. LTEU takes the expert with the highest
   mean weight in every MoE layer; IDFF assigns each document to the first layer whose top-K experts
   (the model's own K) contain the target; SEPO protects the top k_r = 0.1 entries of each target
   expert by retain-gradient magnitude and activates the top k_f = 0.5 of the rest by forget-gradient
   magnitude. Shared experts are never updated.
2. *Sequential unlearning, shallow to deep.* The target expert weights are parametrized as a masked
   Bézier curve towards the NegGrad+ expert; the float32 control point and the router of the layer
   minimize E_t[L(D_r) − β L(D_match^(l))] + α L_dist (β = 0.5, α = 1, 50% of the retain texts).
3. *Choosing t^(l).* Evaluating MMLU for every grid point and layer is far too slow, so t^(l) is the
   grid point with the highest held-out forget loss whose held-out retain loss stays within
   `--retain-tol` (5%) of the current model. MCU uses the same rule.

**Baselines.** GA and NegGrad+ (β = 0.5) update all parameters (AdamW 1e-5, 1 epoch over 512 forget
documents). SEUF selects the expert(s) with the highest relevance over the whole network, trains those
experts and their routers (AdamW 3e-4) and adds the routing distance constraint (α = 1).

**Relearning** fine-tunes all parameters for 10 steps (AdamW 1e-5) on benign text that shares no
8-gram with the forget corpus (and, for RWKU, mentions no target), then evaluates FE / UT.

The default learning rates of the curve-based methods (`--learning-rate`, SGD) are starting points
and may need tuning per model.

## Tests

`python -m pytest tests -q` runs every script end to end on tiny random Qwen2-MoE models (CPU).
With `DEEPSEEK_CODE=<directory containing modeling_deepseek.py and configuration_deepseek.py>`
the same pipeline also runs on a tiny DeepSeek-V2 built from its remote code.
