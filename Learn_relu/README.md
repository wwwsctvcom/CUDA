### build
```
nvcc relu.cu -o relu
```

### output
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
src val: 4.000000, relu result: 4.000000
src val: -4.000000, relu result: 0.000000
src val: 2.000000, relu result: 2.000000
src val: -4.000000, relu result: 0.000000
src val: 0.000000, relu result: 0.000000
src val: -2.000000, relu result: 0.000000
src val: -4.000000, relu result: 0.000000
src val: -1.000000, relu result: 0.000000
src val: 3.000000, relu result: 3.000000
src val: -2.000000, relu result: 0.000000

```