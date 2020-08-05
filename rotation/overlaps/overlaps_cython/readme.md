```python
rbox_overlaps(
     np.ndarray[DTYPE_t, ndim=2] boxes,
     np.ndarray[DTYPE_t, ndim=2] query_boxes,
     np.ndarray[DTYPE_t, ndim=2] indicator=None,
     np.float thresh=1e-4)
```

输入参数中：

* boxes和query_boxes：设置anchor和gt，anchor shape为(num_all_anchor, 5)， gt(num_gts, 5)为; 5=xyxyt, 其中t为角度制，定义同opencv方法；顺序无所谓，注意输出的overlaps和顺序对应就行；numpy数组
* indicator：筛选的flag，可以为None
* thresh：如果有indicator（例如采用直框iou或者一些soft-weight），根据这个阈值进行筛选计算，未达到阈值的overlap为0
* output：shape为（num_all_anchor, num_gts）或（num_gts, num_all_anchor），根据输入的哪个在前而定