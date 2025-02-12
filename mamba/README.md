# Mamba
[paper](https://arxiv.org/pdf/2312.00752) 

### Abstract
- Transformer's attention mechanism is inefficient for long sequences,
so we tried to use convolution, recurrent and state space models to 
mitigate this effect without decreasing the performance.
- Designing hardware-aware algorithms was needed for computational
efficiency.
- No `Attention` is used in this architecture.
- Was tested on genomics audio and language.

### Introduction
- Foundational Models are mostly using Transformers as `sequence models`. So, faster FMs 
better performance.
- The reason transformers are pretty good is their Attention mechanism that `selects` the 
right information inside a context window.
- `Structured space models` turns out to be good in continuous data modeling like audio and DNA,
but less successful for discrete dense modalities like text.
- So this paper introduces `Selective Space Models`, where they design an architecture to select
data in an input-dependent manner, like attention. Parametrizing regular SSMs turns out to be
good enough for this purpose. This modofied SSM architecture + MLPs = Mamba.

### State Space Models
- The computations can be done in either a `linear recurrence` manner or `global convultion`. Convolutional
mode is used for training while recurrent mode is used for inference.
- Discretizing the continuous input can be seen as the first step in the computation graph.
- Linear Time Invariance (LTI) is another important feature of these models. It means that the complexity
grows linearly with respect to number of input tokens.
![arch](./assets/mamba-1.jpeg) 

### Selective State Space Models
- Compressing the hidden state of a token is the thing that determines the tradeoff.  High compression -> Less Computation & Worse Performance. 
No compression -> Much more Computation & Best Performance. The earlier being sequence models like RNNs and the latter representing
attention mechanism.
- To measure the efficacy of the selective algorithm two tests can be conducted: 1) `Selective Copying`, 2) `Induction Heads`
- Showing the model a specific sequence then asking it for the same sequence after a blank long sequence is called selective copying.
- The circuit behind in-context learning is said to be Induction Heads.
- LTI models tend to fail in both of these tests because the lack of `selectivity`.
- Making the recurrent dynamics of an RNN (or convolutional kernel) input-dependent will increase the `selectivity`.
- The goal is to maximize the hidden state size without paying memory & computation cost.
- Without input-dependency SSMs can be interperted as convolutional operations, which leverages the fast Fourier transoform.
With selectivity SSMs are no longer equal to convolution so `kernel fusion`, `parallel scan` and `recomputation` techniques 
are leveraged to boost the performance on hardware level.
- Memory IO at the GPU level turns out the major bottleneck. Moving matrices from Global to Shared Memory is not a relatively fast
operation. So, fusing a bunch of functionalities in a single kernel was the solution.
- Using the gated attention unit (GAU) instead of regular Multi-Head Attnetion (MHA), and leveraging H3 SMM architecture at the same time led to the Mamba
hybrid approach.
- Selectivity is important for the following reasons: `Variable Spacing`, `Filtering Context` and `Boundary Resetting`.
Variable spacing and filtering context rely on the ability to remove noise and not pay much attention to it. Ignoring old
unrelated data and focusing on important relevant data is the goal.
- The variable `delta` controls the amount of focus on the current token `x_t`. Big delta makes the hidden state focus more on `x_t`
while small one ignores the current input.
- So, SSMs can be interperted as continuous systems discretized by `delta`.
- Selective variables `B & C` plays a role like gating: whether to let x -> h_t or h_t -> y.
