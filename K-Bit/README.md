# K-bit Methods for Efficient Deep Learning
[Video](https://youtu.be/2ETNONas068?si=Mv6MCe-It3Kd9ojU) 

### Qunatization By Definition.
- We have a set of indices {0, 1, 2, 3, ...} which correspond to all
  the possible values of a data type.

```
INT4:
  0    1    2    3    4  ...
 -7   -6   -5   -4   -3  ...

FP4:
  0    1    2    3    4  ...
-12   -8   -6   -4   -3  ...
```

- Then, we divide by the largest possible
  value among given values to obtain a range
  of numbers `[-1, 1]`.

```
INT4 (dividing by 7):

  0      1      2      3      4  ...
 -1.00  -0.86  -0.71  -0.57  -0.43  ...

FP4 (dividing by 12):

  0      1      2      3      4  ...
 -1.00  -0.67  -0.50  -0.33  -0.25  ...
```
- Now, suppose we have a tensor with the following 
inputs `[-0.8, -0.7, 0]` and we want to represent
it in `INT4` dtype. First, we will divide each element
by the absmax value.
Then, We would basically collect
the corresponding indices for each closest
value representable by our dtype.
```
[-0.8, -0.7, 0] --normalize--> [-1, -0.875, 0] --closest indices--> [0, 1, 9] ...
```
- These indices are the key for Quantization/De-quantization. They serve as intermediate 
representation to map to both quantized and non-quantized values..

---

- Other important consideration is that each datatype has its own tradeoff.
Think about FP8, where we have 8-bits to represent a floating point number.
The first bit being the sign bit, 7 bits remains to share between the exponent
and the fraction. 
The more bits we allocate for the exponent the better we 
represent large/small numbers, still, the worse the precision gets for sensitive
numbers.
The more bits we allocate for the fraction the more accurate
we represent numbers, still, the worse we represent large/small
numbers.

- So, we must understand the tradeoff for each datatype and choose the optimum
one for our use case.

- **Dynamic Exponent Quantization** comes into play to dynamically adjust the allocated
number of bits for either the exponent or the fraction. So, we are able to represent
either large/small numbers or high-precision for intermediate numbers.

---

### 8-Bit Optimizers
- During training, optimizer states occupies a large amount of GPU memory.
So, minimizing or decreasing the amount of their consumption would lead to
performance enhancement.

- The procedure is to quantize the optimizer states and de-quantize the gradients
when pushing them to the weights. Still, one major problem is outliers occurrence.
To mitigate their effect, we process the optimizer state in `blocks` so these outliers
only affect the block they are in.

---

### Scale
- Turns out that these methods for quantizing optimizer states works good enough for models
with number of parameters under certain threshold. Going back to the outliers, if we don't 
leverage block-wise methods the unstability dramatically increases. However, after a
certain model size, even block-wise methods can't mitigate the outliers effect.

- To mitigate this performance degradation caused by outliers, which exponentially increase
with model size, we detect them manually cast them back to half-precision do the operation then
cast them back to 8-Bit. This particular approach ensures almost all of the parameters to remain 8-bit
only casting back what is needed to be casted back. `LLM.int8()`.

- Imagine we have a specific amount of bits, which of the following would lead to better performance:
FP16 16M model or FP8 30M model. Turns out that going with the bigger model almost always is the best way.
