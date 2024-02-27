### build
```
nvcc device_information.cu -o device_information
```

### output
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
CUDA Driver Version / Runtime Version         11.2  /  10.2
CUDA Capability Major/Minor version number:   7.5
GPU Clock rate:                               1560 MHz (1.56 GHz)
Memory Bus width:                             128-bits
  L2 Cache Size:                                1048576 bytes
Max Texture Dimension Size (x,y,z)            1D=(131072),2D=(131072,65536),3D=(16384,16384,16384)
Max Layered Texture Size (dim) x layers       1D=(32768) x 2048,2D=(32768,32768) x 2048
  Total amount of constant memory               65536 bytes
Total amount of shared memory per block:      49152 bytes
Total number of registers available per block:65536
Wrap size:                                    32
Maximun number of thread per multiprocesser:  1024
Maximun number of thread per block:           1024
Maximun size of each dimension of a block:    1024 x 1024 x 64
Maximun size of each dimension of a grid:     2147483647 x 65535 x 65535
Maximu memory pitch                           2147483647 bytes
----------------------------------------------------------
Number of multiprocessors:                      16
Total amount of constant memory:                64.00 KB
Total amount of shared memory per block:        48.00 KB
Total number of registers available per block:  65536
Warp size                                       32
Maximum number of threads per block:            1024
Maximum number of threads per multiprocessor:  1024
Maximum number of warps per multiprocessor:     32
```