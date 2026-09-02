# PLEU Architecture Design

## 1. Method Overview

Progressive Layer-wise Expert Unlearning (PLEU) is a selective unlearning framework for Mixture-of-Experts models. Its core idea is to avoid treating expert forgetting as a single global intervention. Instead, PLEU progressively identifies where forgetting knowledge concentrates across layers, then removes it with increasingly fine-grained control over data, experts, and parameters.

This design is model-agnostic. It applies to any architecture that satisfies three conditions:

- the model is organized in layers or stages;
- each layer contains routeable submodules such as experts, branches, adapters, or specialists;
- the model exposes routing probabilities or routing decisions for internal units.

The same abstraction works in both CV and LLM settings. In a visual model, routing units may correspond to spatial tokens or patch tokens. In a language model, routing units may correspond to sequence tokens or span-level positions.

## 2. Progressive Layer-wise Expert Localization

PLEU does not assume that forget knowledge belongs to one global expert. Instead, it localizes target experts layer by layer.

For each routed layer, the method:

1. collects routing probabilities on the forget set;
2. converts routing-unit predictions into a sample-level expert vote;
3. selects the expert that receives the strongest forget-side support in the current remaining samples;
4. records this expert as the target expert for that layer.

The key point is that expert selection is not repeated during training. Target experts are fixed once before unlearning starts. This makes the localization step stable and makes the subsequent optimization focus on a known forget path.

## 3. Progressive Deep Erasure

After layer-wise localization, PLEU performs erasure progressively rather than globally.

### Inter-layer data flow filtering

PLEU filters forget samples layer by layer. A sample participates in the current layer’s unlearning only when its patch-token vote toward the target expert is strong enough. Samples that do not satisfy this condition flow to the next layer. The last routed layer acts as a fallback so that leftover samples are still processed.

This mechanism prevents every layer from seeing the entire forget set. As a result, each layer only removes the portion of knowledge that it is most responsible for, which reduces redundant perturbation.

### Mode connectivity optimization

PLEU then uses an MCU-style Bezier curve to connect:

- the original model,
- the target unlearned model,
- and a trainable midpoint.

Only the midpoint is optimized. The curve keeps the endpoints fixed, so the system can search for an intermediate model that balances forgetting and utility preservation. This gives PLEU a controlled optimization path instead of direct full-parameter overwriting.

### Selective expert parameter optimization

PLEU does not update the whole network. For each target layer, the trainable scope is restricted to:

- the gate of the selected target expert path;
- the target expert parameters themselves;
- the parameters admitted by the mask in the MCU stage.

Other experts are frozen. This makes the unlearning action local, while the curve still preserves a global parameter path from original to target.

## 4. Comprehensive Evaluation

PLEU is evaluated with a relearning-oriented protocol instead of relying only on standard forgetting metrics.

The evaluation tracks four core indicators:

- ForgetAcc(train);
- RA(train);
- ForgetAcc(test);
- RA(test).

This setup checks two things at once:

1. whether the forget set is actually suppressed;
2. whether the resulting model can resist knowledge reactivation during relearning.

The relearning test is important because a model can look good under immediate forgetting metrics but still recover quickly once training resumes. PLEU is designed to be judged under that stronger condition.

## 5. System Workflow

The complete pipeline is:

1. load the original model and the target unlearned model;
2. compute layer-wise target experts from the forget set using patch-token voting;
3. build inter-layer flow masks and generate the MCU mask;
4. train the Bezier midpoint with retain loss, forget loss, and anchor loss;
5. search along the curve and save the best checkpoint under the split-specific criterion.

The split mode changes the selection rule and checkpoint naming:

- `class`: emphasize structured forgetting and utility preservation;
- `random`: align the saved checkpoint with the target forget accuracy reference.

## 6. Key Design Choices

PLEU is built around three implementation choices.

First, it localizes experts progressively instead of assuming one global target expert.

Second, it uses patch-token voting rather than a CLS-style shortcut, because routing is decided at the token level in MoE-style backbones.

Third, it separates erasure into two levels: expert-level flow filtering and parameter-level MCU masking. This keeps the method flexible enough for both CV and LLM backbones.

## 7. Practical Interpretation

In practice, PLEU behaves like a staged unlearning system.

Early layers remove the portion of forget knowledge they can confidently claim.
Later layers handle the leftovers.
The MCU stage then refines the parameter trajectory so that the final model does not overfit the forget objective and can still preserve retained utility.

That is the main idea of PLEU: progressive localization, progressive erasure, and progressive evaluation.
