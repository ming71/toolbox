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

当输入box的w小于一定范围（大概0.01以下）时，IoU结果有一定概率不是0而成为inf，结果上有可能引发CUDA error或者overlap为nan。

暂时未找到cu文件中bug出处。

安全起见，送进来的box预先进行filter，保留宽高有效的box，参考[transform_rbox](https://github.com/ming71/toolbox/blob/master/rotation/transform_rbox.py)。

