## SME/SVE Optimized 2D Convolution (3×3)

This project implements several SVE/SME-optimized versions of a 3×3 conv2d kernel and compares them with a scalar reference implementation.

## Performance Results

Test configuration:
Image: 256×256
Averaged over 2000 runs

| Version                                | Time (ms) | Speedup |
| :------------------------------------- | :-------- | :------ |
| Scalar (reference)                     | 0.018     | 1.00x   |
| SVE self-ext + predicate tail          | 0.045     | 0.40x   |
| SVE load9 (Algorithm 2 style)          | 0.040     | 0.45x   |
| SVE cyclic-kernel + pred-merge (Algo1) | 0.080     | 0.22x   |
| SVE load9 + register blocking (4 rows) | 0.038     | 0.47x   |
