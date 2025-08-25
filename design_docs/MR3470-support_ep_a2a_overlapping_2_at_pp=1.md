# What does this PR do ?
On top of [MR3217](./MR3217-support_ep_a2a_overlapping_1_transformer_layer_callables.md), this MR aims to add a fine-grained scheduler to organize the submodule callable proposed in MR3074 and support calling the scheduler when PP=1 for EP A2A hiding.

Major changes
- `fine_grained_callables.py`: add `PreProcessNode/PostProcessNode/TransformerLayerNode`, which serve as the interfaces of submodules and will be called in fine-grained scheduler
- `fine_grained_scheduler.py`
  - `class LayerSchedulePlan`：This class organizes the computation nodes for a transformer layer,
    including attention, post attention, MLP, dispatch, and combine nodes and put them into different stream: attn/mlp in computing stream and dispatch/combine in communicating stream
  - `schedule_layer_1f1b()`: Layer-level execute forward step of one micro batch and backward step of another micro batch
This function interleaves forward and backward operations to maximize
    parallelism and efficiency.
  - `schedule_chunk_1f1b()`: 
Schedules one-forward-one-backward operations for a model chunk.
This function interleaves forward and backward operations across multiple layers
    to maximize parallelism and efficiency.
- `combined_1f1b.py`: combine the logic of `forward_step()` and `backward_step()` with fine grained scheduler
```
    Descriptions:
        This method merges the forward_step() and backward_step() methods in the schedules.py file.
        Assuming that:
            def forward_step():
                # forward_preprocess()
                # forward_compute()
                # forward_postprocess()
            def backward_step():
                # backward_preprocess()
                # backward_compute()
                # backward_postprocess()
        Then the forward_backward_step() method will be:
            def forward_backward_step():
                # forward_preprocess() // the same as the forward_step()
                # GENERATE f_schedule_plan // schedule happens in schedule_chunk_1f1b()
                # backward_preprocess() // the same as the backward_step()
                # COMBINED_FORWARD_BACKWARD_COMPUTE() // by calling schedule_chunk_1f1b()
                # forward_postprocess() // the same as the forward_step()
                # backward_postprocess() // the same as the backward_step()
```
- `schedules.py`
  - wrap the loss calculation part of `forward_step()` into a separate function `forward_step_calc_loss()` so that it could be reused in fine-grained scheduler.
  - combine one forward and one backward and call the fine-grained scheduler to enable EP A2A overlap at PP=1
