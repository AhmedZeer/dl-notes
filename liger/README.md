# Liger
[Paper](https://arxiv.org/pdf/2410.10989) 
[Video](https://www.youtube.com/watch?v=gWble4FreV4) 

### Introduction
- The latency and performance degradation occurring when using
frameworks like PyTorch stems from extra computational overheads
like `call stack`, `dispatching` and `CUDA Kernel Launch` latencies.
- Furthermore, the true potential in optimizing models lies in
`Kernel Fusion`. The reason for that is accessing High Bandwidth
Memory (VRAM) is reduced and most computations are done in the
fly without requiring copying tensors back and forth all over the
place between kernels.
- Storing almost every activation tensor in memory for future use
in the backward pass also introduces significant GPU memory
usage.
- To address these issues `compilation` and `operation fusion`
have been used to address these issues.


### Model Compiler
- `torch.compile`: Has a Just-In-Time (JIT) frontend that captures
the computational graph and converts python-level operations
to intermediate representation (IR). Its backend performs
some optimization on this intermediate representation
and generate Triton code for GPUs and OpenMP C++ code
for CPUs.

### Operation Fusion
- The main reason for custom operation fusion is to mitigate the
bottleneck that arises between High Bandwidth Memory (HBM/VRAM)
and Shared Memory (SRAM). While the former is huge and slow,
the latter is small and fast. Each Stream Multiprocessor (SM)
wants to read the data and launch a bunch of threads for parallel
execution, however the aforementioned mismatch leads to delays.

- From an algorithmic perspective, techniques like operation
fusion depends on optimizing `computational patterns` that
enables tailored and more precise performance improvements.

- `FlashAttention`: Splits the attention computation into smaller
blocks that fits on the SRAM, avoiding the need for materializing
the whole attention matrix and redundant access to the VRAM/HBM.

- `xFormers`: A library from Meta that includes optimized
transformer building blocks kernels implemented in CUDA/Triton.

- `Unsloth`: Reimplements LLMs and LoRA adapters for optimized training.

- `EffecientCrossEntropy`: Fuses the linear projection with the
cross entropy loss, and computes the loss in a block-wise manner
to avoid materializing the entire logits tensor.
to avoid materializing the entire logits tensor. Furthermore,
we overwrite the logits tensor with their gradients to
minimize memory usage. Finally, these changes mean calculating
the gradients and the forward pass in a single kernel.
