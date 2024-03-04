### reference
```
https://face2ai.com/CUDA-F-4-1-%E5%86%85%E5%AD%98%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/
```

### build
```
nvcc global_variable.cu -o global_variable
```

### output
```
Host: copy 3.140000 to the global variable
Device: The value of the global variable is 3.140000
Host: the value changed by the kernel to 5.140000
```