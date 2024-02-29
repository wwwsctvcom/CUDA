### build
```
nvcc reduceInteger.cu -o reduceInteger
```

### output
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
with array size 16777216
grid 16384 block 1024
cpu sum:2139095040
cpu reduce                 elapsed 0.032000 ms cpu_sum: 2139095040
gpu warmup                 elapsed 0.008000 ms gpu_sum: 2139095040<<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.005000 ms gpu_sum: 2139095040<<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.002000 ms gpu_sum: 2139095040<<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.003000 ms gpu_sum: 2139095040<<<grid 16384 block 1024>>>
Test success!
```