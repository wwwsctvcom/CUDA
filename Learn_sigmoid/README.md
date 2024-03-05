### build
如果有不懂sigmoid的，请使用搜索引擎搜索答案
```
nvcc sigmoid.cu -o sigmoid
```

### output
执行如下指令进行测试
```
sigmoid.exe 5
```

结果如下
```
src val: 1.000000, corresponding sigmoid result: 0.731059
src val: 2.000000, corresponding sigmoid result: 0.880797
src val: 3.000000, corresponding sigmoid result: 0.952574
src val: 4.000000, corresponding sigmoid result: 0.982014
src val: 5.000000, corresponding sigmoid result: 0.993307

```