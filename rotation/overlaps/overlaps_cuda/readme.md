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

~~当输入box的w小于一定范围（大概0.01以下）时，IoU结果有一定概率不是0而成为inf，结果上有可能引发CUDA error或者overlap为nan。~~

~~暂时未找到cu文件中bug出处。~~

~~安全起见，送进来的box预先进行filter，保留宽高有效的box，参考[transform_rbox](https://github.com/ming71/toolbox/blob/master/rotation/transform_rbox.py)。~~

### bug fixed

根本错误出在overlap计算的cuda程序中多边形三角分割的有点问题，简单来说就是当输入box有area很小的时候inter_area不是0而是union，再加上他没有设置较小的分母，就nan了。

fix bug：在函数devRotateIoU中改改限定值就行

* 避免box重合而nan，分母加上小的数值

  ```
    float result = area_inter / (area1 + area2 - area_inter + 1e-6);
  ```

* 过小的box不参与分割直接给inter_area为0

  ```
    if (region1[2] < 0.1 | region1[3] < 0.1 | region2[2] < 0.1 | region2[3] < 0.1){
      area_inter = 0;
    } 
  ```

改完记得重新编译。