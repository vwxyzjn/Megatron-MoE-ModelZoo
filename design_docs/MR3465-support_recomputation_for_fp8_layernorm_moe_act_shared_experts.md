# Modifications

Add fine-grained recomputation on layernorm/moe_act for FP8 training.
Add memory saving optimizations on attn linear_proj and shared experts.

This MR relies on the TE [PR1865](https://github.com/NVIDIA/TransformerEngine/pull/1865).

# MR Details

## Problems to Solve
In MR !2207 (merged), we added an output-discarding recomputation that trickily releases some saved tensors and restores them by recomputation. This feature relies on the fact that the saved tensor is the original output tensor (otherwise it cannot be released). However, in the current implementation of TE, the Linear/GroupedLinear always saves the quantized input tensors for backward computation, which prevents the output-discarding recomputation from working.

## Solution and Modifications
To solve this issue, we need modifications in both TE and MCore.

1. For the TE side, in PR1865, we added a `save_original_input` argument to Linear/GroupedLinear. If `save_original_input` is True, the module will save the original input rather than the quantized input, and it will quantize the input during the backward computation for `wgrad` calculation.
2. For the MCore side, in this MR, to make layernorm/moe_act work in FP8 training, we just need to set the `save_original_input` argument of the modules that accept the output of layernorm/moe_act as input.
  * For moe_act, we should set `linear_fc2` of the experts.
  * For input_layernorm, we should set `linear_q_down_proj` and `linear_kv_down_proj` of MLA, or `linear_qkv` of MHA/GQA.
  * For pre_mlp_layernorm, we should set linear_fc1 of the shared_experts.

Besides, utilizing this new feature in TE can also optimize some other parts. In the current implementation of the Transformer model, some input tensors are used by multiple modules and save in different dtypes, such as:
* The output tensor of fused attn is saved by its self as original and also saved by the following `proj_linear` as a quantized tensor.
* For MoE models with shared experts, the output of pre_mlp_layernorm is saved by the router-gating linear as original and by the shared experts linear as a quantized tensor.

These places have unnecessary extra memory overhead. So, in this MR, we also set `save_original_input` for the `proj_linear` of the attention and linear_fc1 of the shared expert to save memory usage.

## Results
Here are some memory and performance results of this feature (applied on a DeepSeek-V3 TP1PP8VPP4 setting).

1. With FP8 moe_act recompute, it can save about 4.4GB of memory.
2. With FP8 layernorm recompute, it can save about 9.6GB of memory.
3. For attn `proj_linear`, by enabling save_original_input, it can save about 4.4GB of memory, and the overhead is only about 10us.
4. Similar to 3, for shared experts, it can save about 2GB of memory with similar overhead.