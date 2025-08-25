# What does this PR do ?

The main motivation for introducing this additional flexibility is to effectively enable Virtual Pipeline Parallelism for heterogeneous models, such as Mixture of Experts (MoE), while ensuring balanced pipeline stages.

Recent MoE models, like DeepSeek, have become increasingly heterogeneous. Specifically, their total number of layers often isn't divisible evenly by common pipeline parallelism factors (such as 4 or 8). Additionally, these models frequently utilize different architectures at the beginning and end—DeepSeek, for instance, uses dense layers for the first few layers and a distinct MTP layer at the end. This heterogeneity presents a challenge in applying VPP effectively while maintaining balance across stages, which is crucial for minimizing pipeline parallelism bubbles.

The current implementation of VPP imposes significant restrictions that hinder its applicability in these scenarios. Two critical limitations are:

1. **Strict Divisibility Requirement:** The current VPP mandates that the number of layers in each pipeline stage must be divisible by the VPP size. When using `--decoder-{first,last}-pipeline-num-layers` for uneven pipeline partitioning, each stage's layer count must still be divisible by the VPP size.
    
    For example, with DeepSeek-v3 (61 layers plus 1 MTP layer structured as `[[embedding], [dense]*3, [moe]*58, [mtp, loss]]`), achieving balanced partitioning requires placing more lightweight dense layers in the first stage and fewer layers in the last stage due to loss computation. This arrangement results in layer counts that are not divisible by the required VPP size, leading to runtime errors:
    
    ```
    DeepSeek-v3: num_layer=61 layers + 1 mtp layer;
    we will get the error &quot;number of layers at last stage: 6 must be divisible by virtual pipeline parallel degree 4&quot; if we set:
    --decoder-last-pipeline-num-layers 6 --num-virtual-stages-per-pipeline-rank 4
    ```
    
2. **Restrictions with Standalone Embedding and Loss Layers:** Currently, specifying `account_for_{embedding,loss}_in_pipeline_split` prevents the use of the `--decoder-{first,last}-pipeline-num-layers`, further limiting VPP usability in diverse scenarios.

Given these complexities, minor adjustments to the existing arguments and implementation are insufficient. Therefore, this MR introduces a user-defined pipeline layout. This approach consolidates existing VPP and uneven PP arguments into a single, flexible argument, enabling precise control over the number and type of layers in each virtual pipeline stage. This not only addresses the heterogeneity challenges of MoE models but also broadens the applicability of VPP to potentially include multimodal models and other complex architectures. 

```
Case1: PP=3, VPP=2, num_layer=7
[[[embedding], [4,5]], 
[[0,1], [6]], 
[[2,3], [loss]]]
```

```
Case2: PP=3, VPP=3, num_layer=6
[[[embedding], [2], [5]],
[[0], [3], []],
[[1], [4], [loss]]]
```

```
DeepSeek-v3 PP=8 num_layer=61 + 1 mtp layer; vpp = 4; mtp costs 2x moe layer time
[
[[embedding, dense, dense], [moe, moe], [moe, moe], [moe, moe]],
[[dense, moe], [moe, moe], [moe, moe], [moe, moe]],
[[moe, moe], [moe, moe], [moe, moe], [moe, moe]] * 4,
[[moe, moe], [moe, moe], [moe, moe], [mtp]],
[[moe, moe], [moe, moe], [moe, moe], [loss]]
]

-&gt; [[embedding, dense, dense], [moe, moe] * ?, [mtp], [loss]]

```

```
Case4: 88 layer MoE from ant, PP=16
[
[[embedding], [moe, moe]],
[[moe, moe, moe], [moe, moe, moe]] * 14
[[moe, moe], [loss]]
]
```

```
Case5: Hybrid MoE, moe_freq=2, moe layers = 2x dense
[
[[embedding, dense, dense], [dense, dense]],
[[moe], [moe]],
[[dense, dense], [dense, dense]],
[[moe], [moe, loss]],
]
```

The above examples might not be the realistic configurations. But it shows that our existing approach cannot provide such flexibility where we obtain the layer mapping by calculation with some constraints. 

In this MR, we want to provide a new argument (let me first call it `pp_layout`) to allow users to specify the pipeline parallelism layout directly. In MCore, we can use regular expression to parse the input string and obtain the layer configurations for each rank individually. If there is `null` placeholder, we will add `IdentityOP` inside this layer to make the model trainable.
