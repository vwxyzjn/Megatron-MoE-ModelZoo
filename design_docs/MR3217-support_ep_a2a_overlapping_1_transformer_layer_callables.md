# What does this PR do ?
1. Splits the TransformerLayer forward into 5 wrapped submodules, including 2 computing submodules and 2 communicating submodules: attention， post_attention, dispatch, mlp/moe, combine. This supports more fine grained submodule scheduling strategy. 
   * callables: `submodule_attn_forward/ submodule_post_attn_forward/ submodule_dispatch_forward/ submodule_moe_forward/ submodule_combine_forward`.
   * wrapper: `build_layer_callables`.
      * Provides modular function wrappers​
      * Selects different implementations based on whether it's a MoE (Mixture of Experts) model​
      * Manages forward and backward_dw functions for each submodule​
   * Provide unit test to verify the correctness of submodules.
2. Refactor Token Dispatcher(All2All&amp;Flex) to PreProcess/AllToAll/PostProcess stages, this helps remove duplicated code in submodules and provides simpler interface.
3. Fix the DeepEP integration approach: The original sync version has potential bugs such as integer overflow and missing synchronizations, changing to async approach fixes these issues.
