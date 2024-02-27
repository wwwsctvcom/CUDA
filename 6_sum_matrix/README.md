### build
```
nvcc sum_matrix.cu -o sum_matrix
```

### output
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU execution time elapsed: 0.045000 sec.
CUDA configuration <<<(128, 128), (32, 32)>>>, Time elapsed: 0.002000 sec
PASS
CUDA configuration <<<(524288, 1), (32, 1)>>>, Time elapsed: 0.003000 sec
PASS
CUDA configuration <<<(128, 4096), (32, 1)>>>, Time elapsed: 0.002000 sec
PASS
```