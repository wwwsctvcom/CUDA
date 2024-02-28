### build
```
nvcc sum_matrix2D.cu -o sum_matrix2D
```

### output
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU Execution Time elapsed 0.045000 sec
GPU Execution configuration<<<(128, 128), (32, 32)>>> Time elapsed 0.002000 sec
PASS
```