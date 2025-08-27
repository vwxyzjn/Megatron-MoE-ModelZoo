# What does this PR do ?
This PR enables the support of A2A Overlapping for interleaved PP and lets the MTP layers be included in the A2A Overlapping strategy. Also, this PR supports `--overlap-grad-reduce` when enabling A2A overlap.

Major changes:
- Support `--overlap-grad-reduce`
  - [Merged][Changes in TE](https://github.com/NVIDIA/TransformerEngine/pull/1976):
    - For params whose wgrad is executed in `backward_dw()` instead of `backward()`, set attr `skip_backward_post_hook` to True
    - Expose an interface to manually register the `wgrad_accumulation_and_reduce_hook` for later call;
    - In backward_dw(), manually launch the `wgrad_accumulation_and_reduce_hook` after wgrad computation.
  - Changes in MCore
    - For params with attr `skip_backward_post_hook` being true, find their modules and register `_make_backward_post_hook` as `wgrad_accumulation_and_reduce_hook`
- Support interleaved PP
  - Separate the preprocessing and postprocessing part of `forward_step_helper` to `forward_step_helper_preprocess` and `forward_step_helper_postprocess`, so is `backward_step_helper`. 
  - Wrap the p2p send/recv into multiple methods, `pp_pre_forward`, `pp_post_forward`, `pp_pre_backward` and `pp_post_backward`.
  - Add `combined_1f1b_schedule_for_interleaved_pipelining` in combined_1f1b.py. This method merges the functionality of `forward_step_helper` and `backward_step_helper` and eventually calls `combined_forward_backward_step` function defined in `combined_1f1b.py`.
- Support MTP
  * Add pipeline parallel layout check for MTP: MTP can only be enabled in the last stage with LMLoss
  * Refactor MTP forward functions:
    * Move embedding generation, input rolling, MTP loss computation and loss tracker saving from `MTPBlock.forward` to `MTPLayer.forward`.
    * Add `self._get_embeddings`, `self._concat_embeddings` in MTPLayer as preprocessing functions.
    * Add `self._postprocess` in MTPLayer as postprocessing function.
  * Support MTP Overlap by adding function callables(reuse `_get_embeddings`, `_concat_embeddings`, `_postprocess`, etc.) in `fine_grained_callables.py`
  * Support `--mtp-num-layers 1` only. A2A overlap with multiple mtp layers supported is still not ready yet.
