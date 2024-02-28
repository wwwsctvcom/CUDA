### build
```
nvcc simple_sum_matrix2D.cu -o simple_sum_matrix2D
```

### output
```
D:\ClionsWorkspace\CUDA\11_simple_sum_matrix2D>simple_sum_matrix2D.exe 32 32
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU Execution Time elapsed 0.190000 sec
GPU Execution configuration<<<(256, 256), (32, 32)>>> Time elapsed 0.010000 sec

D:\ClionsWorkspace\CUDA\11_simple_sum_matrix2D>simple_sum_matrix2D.exe 32 16
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU Execution Time elapsed 0.161000 sec
GPU Execution configuration<<<(256, 512), (32, 16)>>> Time elapsed 0.010000 sec

D:\ClionsWorkspace\CUDA\11_simple_sum_matrix2D>simple_sum_matrix2D.exe 16 16
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU Execution Time elapsed 0.160000 sec
GPU Execution configuration<<<(512, 512), (16, 16)>>> Time elapsed 0.010000 sec

D:\ClionsWorkspace\CUDA\11_simple_sum_matrix2D>simple_sum_matrix2D.exe 16 8
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU Execution Time elapsed 0.170000 sec
GPU Execution configuration<<<(512, 1024), (16, 8)>>> Time elapsed 0.000000 sec

D:\ClionsWorkspace\CUDA\11_simple_sum_matrix2D>simple_sum_matrix2D.exe 8 8
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CPU Execution Time elapsed 0.160000 sec
GPU Execution configuration<<<(1024, 1024), (8, 8)>>> Time elapsed 0.010000 sec
```