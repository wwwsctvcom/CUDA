### reference
https://face2ai.com/CUDA-F-3-6-%E5%8A%A8%E6%80%81%E5%B9%B6%E8%A1%8C/

### build
```
nvcc -arch=sm_35 nested_hello_world.cu -o nested_hello_world -lcudadevrt --relocatable-device-code true
```

### output
```
depth: 0 blockIdx: 0, threadIdx: 0
depth: 0 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 1
depth: 1 blockIdx: 0, threadIdx: 0
depth: 1 blockIdx: 0, threadIdx: 1
depth: 1 blockIdx: 0, threadIdx: 2
depth: 1 blockIdx: 0, threadIdx: 3
depth: 1 blockIdx: 0, threadIdx: 4
depth: 1 blockIdx: 0, threadIdx: 5
depth: 1 blockIdx: 0, threadIdx: 6
depth: 1 blockIdx: 0, threadIdx: 7
depth: 1 blockIdx: 0, threadIdx: 8
depth: 1 blockIdx: 0, threadIdx: 9
depth: 1 blockIdx: 0, threadIdx: 10
depth: 1 blockIdx: 0, threadIdx: 11
depth: 1 blockIdx: 0, threadIdx: 12
depth: 1 blockIdx: 0, threadIdx: 13
depth: 1 blockIdx: 0, threadIdx: 14
depth: 1 blockIdx: 0, threadIdx: 15
depth: 1 blockIdx: 0, threadIdx: 16
depth: 1 blockIdx: 0, threadIdx: 17
depth: 1 blockIdx: 0, threadIdx: 18
depth: 1 blockIdx: 0, threadIdx: 19
depth: 1 blockIdx: 0, threadIdx: 20
depth: 1 blockIdx: 0, threadIdx: 21
depth: 1 blockIdx: 0, threadIdx: 22
depth: 1 blockIdx: 0, threadIdx: 23
depth: 1 blockIdx: 0, threadIdx: 24
depth: 1 blockIdx: 0, threadIdx: 25
depth: 1 blockIdx: 0, threadIdx: 26
depth: 1 blockIdx: 0, threadIdx: 27
depth: 1 blockIdx: 0, threadIdx: 28
depth: 1 blockIdx: 0, threadIdx: 29
depth: 1 blockIdx: 0, threadIdx: 30
depth: 1 blockIdx: 0, threadIdx: 31
-----------> nested execution depth: 2
depth: 2 blockIdx: 0, threadIdx: 0
depth: 2 blockIdx: 0, threadIdx: 1
depth: 2 blockIdx: 0, threadIdx: 2
depth: 2 blockIdx: 0, threadIdx: 3
depth: 2 blockIdx: 0, threadIdx: 4
depth: 2 blockIdx: 0, threadIdx: 5
depth: 2 blockIdx: 0, threadIdx: 6
depth: 2 blockIdx: 0, threadIdx: 7
depth: 2 blockIdx: 0, threadIdx: 8
depth: 2 blockIdx: 0, threadIdx: 9
depth: 2 blockIdx: 0, threadIdx: 10
depth: 2 blockIdx: 0, threadIdx: 11
depth: 2 blockIdx: 0, threadIdx: 12
depth: 2 blockIdx: 0, threadIdx: 13
depth: 2 blockIdx: 0, threadIdx: 14
depth: 2 blockIdx: 0, threadIdx: 15
depth: 3 blockIdx: 0, threadIdx: 0
depth: 3 blockIdx: 0, threadIdx: 1
depth: 3 blockIdx: 0, threadIdx: 2
depth: 3 blockIdx: 0, threadIdx: 3
depth: 3 blockIdx: 0, threadIdx: 4
depth: 3 blockIdx: 0, threadIdx: 5
depth: 3 blockIdx: 0, threadIdx: 6
depth: 3 blockIdx: 0, threadIdx: 7
-----------> nested execution depth: 3
depth: 4 blockIdx: 0, threadIdx: 0
depth: 4 blockIdx: 0, threadIdx: 1
depth: 4 blockIdx: 0, threadIdx: 2
depth: 4 blockIdx: 0, threadIdx: 3
-----------> nested execution depth: 4
depth: 5 blockIdx: 0, threadIdx: 0
depth: 5 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 5
depth: 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
```