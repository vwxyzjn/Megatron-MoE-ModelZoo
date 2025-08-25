# What does this PR do ?
Fixes and updates for external cudagraph:
* Update to TE `make_graphed_callables` new interface - `_num_layers_per_chunk` - for uneven PP.
* Update to TE `make_graphed_callables` new interface - `_reuse_graph_input_output_buffers` - for memory reduction.
* Fix cudagraph fp8 recipe issue by calling get_fp8_recipe from fp8_utils.
* Support MTP cudagraph.
* Share the same dtoh steam between token dispatchers.
* More assertions and conditions for safety.

Related TE PR: https://github.com/NVIDIA/TransformerEngine/pull/1234