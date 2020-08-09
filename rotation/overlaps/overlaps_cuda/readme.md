### install

```
python setup.py build_ext --inplace
```

### api

```
from overlaps_cuda.rbbox_overlaps  import rbbx_overlaps
rbbx_overlaps(gt_rbboxes, query_boxes)
```

其中anchor和gt都为float的numpy数组，格式为xywha，其中a为角度制

### bug report

当输入box的w小于一定范围（大概0.01以下）时，IoU计算不是0而成为inf，0.1以上正常为0，保险起见将输入box的wh中小于0.1的置为0.1，不影响结果。未找到cu文件中bug出处，暂时这么处理。

