### 线程束
个人理解为：尽量让thread的id在第一次if分支的时候就执行完，保证thread id的连续性；
参考地址：https://face2ai.com/CUDA-F-3-2-%E7%90%86%E8%A7%A3%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%89%A7%E8%A1%8C%E7%9A%84%E6%9C%AC%E8%B4%A8-P1/

### build
```
nvcc divergence.cu -o divergence
```

### output
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
Param size: 64, blocksize: 64
Execution Configure (block 64 grid 1)
warmup         <<<1, 64>>>, elapsed 0.002000 sec
kernel_no_warp <<<1, 64>>>, elapsed 0.000000 sec
kernel_use_warp<<<1, 64>>>, elapsed 0.000000 sec
```