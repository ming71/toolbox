### mAP calculation (VOC 2012)
Refer to mAP implemention [Cartucho/mAP](https://github.com/Cartucho/mAP), and support rboxes mAP eval here.

---

Quick start next.
#### folder
Creat directory as follow:  
* **detection-results**:  detections in standard format.  
* **ground-truth**: gt in standard format.  
* **dets**(op): your output, not recommended, just generate results in proper format in `detection-results` folder.  
* **gts**: raw ground truth files.  
* **images-optional**(op): test imgs, not recommended, cause it's really time-consuming.  

#### format
* **converter** 
**for each gt**:  
`<class_name> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> [--<difficult>]`  
for each res(op):  
`<class_name> <conf> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`  
and I'll release some common convert code here.  

#### usage
the directory created above is root_dir, then you can start evaluation as follow"
```python
from map import eval_mAP
eval_mAP(root_dir)
```

#### options
* `difficult` string attached to gt will escape certain object from being included.(such as HRSC, ICDAR).
* `ignore` variabel make it possible to ignore certain class from eval.
* Horizontal bbox eval can be conducted via coordinate trans, or just use this repo [Cartucho/mAP](https://github.com/Cartucho/mAP).
* support 07/12 metric.

#### map_func
make convenient for  calling.
