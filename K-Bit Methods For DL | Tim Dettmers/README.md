# K-bit Methods for Efficient Deep Learning
[Video](https://youtu.be/2ETNONas068?si=Mv6MCe-It3Kd9ojU) 

## Qunatization By Definition.
- We have a set of indices {0, 1, 2, 3, ...} which correspond to all
  the possible values of a data type.

### INT4:
| Index | Value |
|-------|-------|
| 0     | -7    |
| 1     | -6    |
| 2     | -5    |
| 3     | -4    |
| 4     | -3    |
| ...   | ...   |

### FP4:
| Index | Value |
|-------|-------|
| 0     | -12   |
| 1     | -8    |
| 2     | -6    |
| 3     | -4    |
| 4     | -3    |
| ...   | ...   |

---

- Then, we divide by the largest possible
  value among given values to obtain a range
  of numbers `[-1, 1]`.

### INT4:
Dividing by `7` (largest absolute value in `INT4`):

| Index | Value | Normalized Value |
|-------|-------|-----------------|
| 0     | -7    | -1.00           |
| 1     | -6    | -0.86           |
| 2     | -5    | -0.71           |
| 3     | -4    | -0.57           |
| 4     | -3    | -0.43           |
| ...   | ...   | ...             |

### FP4:
Dividing by `12` (largest absolute value in `FP4`):

| Index | Value | Normalized Value |
|-------|-------|-----------------|
| 0     | -12   | -1.00           |
| 1     | -8    | -0.67           |
| 2     | -6    | -0.50           |
| 3     | -4    | -0.33           |
| 4     | -3    | -0.25           |
| ...   | ...   | ...             |
