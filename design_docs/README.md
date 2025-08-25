# Design Docs

This folder selectively collects the design docs of latest MoE features from [NVIDA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

# Table of Contents

| Date | Feature | Commit Link | Design Doc |
|------|---------|---------|------------|
|Aug 19, 2025|Support recomputation for FP8 layernorm/moe_act/shared_experts|[781e765](https://github.com/NVIDIA/Megatron-LM/commit/781e765818b86b8f2e03ac6bb6b09aaaa9d17074)|[MR-3465](./MR3465-support_recomputation_for_fp8_layernorm_moe_act_shared_experts.md)|
|Aug 17, 2025|Add MoE router fusion|[c08d89b](https://github.com/NVIDIA/Megatron-LM/commit/c08d89bea05b2071855733da684a5c15873e913f)|[MR-3809](./MR3809-add_moe_router_fusion.md)|
|Aug 15, 2025|Fixes and updates for external cudagraph|[2b6b46b](https://github.com/NVIDIA/Megatron-LM/commit/2b6b46b796bb8a6c5388e5abd95aad0c97eda391)|[MR-3631](./MR3631-fixes_and_updates_for_external_cudagraph.md)|
|Aug 11, 2025|Support CP and recompute for MTP|[08abeed](https://github.com/NVIDIA/Megatron-LM/commit/08abeedbfe8ac172a1243baf4e55504290d840f8)|[MR-3330](./MR3330-support_cp_and_recompute_for_mtp.md)|
|Aug 01, 2025|Support Expert Parallel A2A Overlapping - (02) Support EP A2A overlap at PP=1|[ae1c882](https://github.com/NVIDIA/Megatron-LM/commit/ae1c88296f465ab4ac9c503d75a57ba4044c47d1)|[MR-3470](./MR3470-support_ep_a2a_overlapping_2_at_pp=1.md)|
|June 16, 2025|Support Expert Parallel A2A Overlapping - (01) Add TransformerLayer Submodule Callables|[8333bd5](https://github.com/NVIDIA/Megatron-LM/commit/8333bd5bb6de2bdbdb3ebebf224b4a339a04ec90)|[MR-3217](./MR3217-support_ep_a2a_overlapping_1_transformer_layer_callables.md)|
|June 13, 2025|Flexible Asymmetric Virtual Pipeline Parallelism with Custom Pipeline Layout|[77732c3](https://github.com/NVIDIA/Megatron-LM/commit/77732c3628ea6843b64d0aa2b02017bd05dddcdb)|[MR-2795](./MR2795-flexible_asym_vpp_with_custom_pp_layout.md)|
|Mar 23, 2025|Multi-Token Prediction Support|[09ebca7](https://github.com/NVIDIA/Megatron-LM/commit/09ebca716da7651fce9e7c161184ea3cf11d6378)|[MR-2628](./MR2628-mtp_support.md)|

