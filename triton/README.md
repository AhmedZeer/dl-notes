# Triton
[Paper](www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) 

### Abstract
- Novel Deep Learning research computations are restricted
by the availability of reliable fast kernels.
- These robust kernels are a must for deploying novel DL 
models/approaches.
- This problem of fine-grained kernels is usually solved
at a high-expense of experts and platform dependence.
- The motivation of this project is the necessity of
new Deep Learning workloads abstraction for minimal cost
and highest efficiency.
- `tiles`, which are static shaped nd-arrays, are the core of Triton.
- This project includes `C-Based Language`, `LLVM Intermediate Representation`
for tensor expressions. IR level tile based optimizations for efficient
machine code execution.

### Introduction
- Performance issues for Deep Learning workloads are addressed through
libraries such as `cuBLAS` and `cuDNN`. However, they support limited
operations, leaving the implementation of novel approaches to experts
at these libraries.
- Current approaches to solve the aforementioned problems are restricted
to specific problems and not general enough to address most of Deep Learning
workload bottlenecks. (Halide, TVL, PlaidML)
- While high-level languages abstractions have been proposed, they lack
robust backbone for tile based optimizations.
- `Triton-C`: C-like language for expressing tensor programs in terms of tiles.
- `Triton-IR`: LLVM based intermediate representation that supports tile-level
abstractions.
- `Triton-JIT`: Just-In-Time compiler to generate efficient machine level LLVM-bitcode from
Triton-IR.

---

### Syntax
- `Tile Declarations`: Similar to that of NumPy's, they declare tiling declarations like so: `int a[16]`
to create `[x_1 x_2 ... x_16]`.
- `Tile Functions`: Vector based builtin functions for dot product, transpose, addition ... operations.
- `Broadcasting`: To broadcast a block in a certain axis `newaxis` keyword exists. For example
to stack columns you can `a[:, newaxis]`, and to stack rows you can `a[newaxis, :]`.

### Semantics
- `Tiling Semantics`: First, they hide the inner details for efficient kernel execution, like
memory coalescing, caching, special hardware utilizations etc. Second, they open the
door for compilers to optimize the code automatically.
- `Broadcasting Semantics`: Assuming two blocks with different shapes, the one with 
the least number of dimension is left padded with 1s until both of the blocks have the
same number of dimensions. Then, broadcasted to match shapes, otherwise an exception
is arised.
```
a: [3,4] ----------> [3,4] ---------------> [3,4]
b:   [4] --padding-> [1,4] --broadcasting-> [3,4]
```
### Programming Model
- Each block in a CUDA kernel consists of multiple threads, while in Triton 
each block contains only one thread. This approach simplifies the kernel.

---

### Triton-IR
- While being similar to the LLVM intermediate representation, Triton-IR also
includes extensions for tile-level data-flow and control-flow.

### Structure
- `Modules`: Triton programs consists of modules, which are basic independently
compiled building blocks that are eventually linked to form global definitions. Each
module consists of constants, global variables, functions etc.
- `Functions`: Has a return type, a name and possibly an empty list of arguments. The interdependency
of blocks forms the control-flow graph (CFG).
- `Blocks`: Triton-IR uses Static Single Assignment (SSA), which means
each block is assigned once and must be defined before usage. Each block
implicitly forms data-flow graph (DFG) whose different paths correspond to
chains in the SSA representation.
- `Types`: `int32<8, 8>` simply forms a block with shape of 8x8 with 32-Bit integers.
- `Instructions`: Block based instructions like `reshape`, `broadcast`, `trans` and `dot` are implemented.

---

### Triton-JIT
- The purpose of the JIT compiler is to transform Triton-IR code into
efficient machine code, via `Machine Dependent & Independent` passes backed
by an `autotuning` engine.

---
- `Prefetching`: Loops can be very inefficient for tile-level memory
operations. This can be mitigated by detecting loops at the IR
level and adding prefetching code where necessary.
- `Peephole`: Since we leverage tile-level computations, new peephole
optimizations can be done, like X = (X^T)^T
---
- `Hierarchical Tilling`: Decomposing relatively large tiles into macro/nano
tiles to ultimately fit memory and computing capabilities.
- `Memory Coalescing`: Since we access multiple adjacent memory addresses
with a single instruction, not utilizing these addresses directly leads
to inefficiencies. Fortunately, we can design a compiler to handle coalesced
memory accesses when possible.
- `Shared Memory Allocation`: Operations with high computation costs can 
save their operands in Shared Memory for faster calculations. The pass
is responsible of determining when and where to store a tile.
- `Shared Memory Synchronization`: Since SRAM operations are async, the risk
of race condition or other unwanted behaviours could arise. So, we add 
some code to form a kind of barrier ensuring thread-safety.
