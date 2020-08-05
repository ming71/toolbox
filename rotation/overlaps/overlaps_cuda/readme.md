### 安装

```
python setup.py build_ext --inplace
```

### 函数接口

* 调用

```
from overlaps_cuda.rbbox_overlaps  import rbbx_overlaps
```

* 参数

```
rbbx_overlaps(gt_rbboxes, query_boxes)
```

其中anchor和gt都为float的numpy数组，格式为xywha，其中a为角度制