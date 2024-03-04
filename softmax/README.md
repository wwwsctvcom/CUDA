### build
```
nvcc softmax.cu -o softmax
```

### output
执行如下指令进行测试，其中可以传入参数表示随机生成多少个数字来计算softmax
```
softmax.exe 5
```

结果如下
```
Set cuda device 0, device name: GeForce GTX 1650, device count: 1
probability: 11.853297
probability: 27.394417
probability: 15.085022
probability: 17.928768
probability: 27.738499
sum of all probability: 1.000000
```