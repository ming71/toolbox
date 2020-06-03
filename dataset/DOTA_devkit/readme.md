
### Installation
1. install swig
```
    sudo apt-get  update
    sudo apt-get install swig
```
2. create the c++ extension for python
```
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```

### Usage
Refer to `demo.ipynb` for more details .
1. Reading and visualizing data, you can use DOTA.py
2. Evaluating the result, you can refer to the "dota_evaluation_task1.py" and "dota_evaluation_task2.py" (or "dota-v1.5_evaluation_task1.py" and "dota-v1.5_evaluation_task2.py" for DOTA-v1.5)
3. Split the large image, you can refer to the "ImgSplit"
4. Merging the results detected on the patches, you can refer to the ResultMerge.py

### Structure
under root_dir:
```
.
├── images
└── labelTxt
```


### mark here
1. After ImgSplit, `diffcult = 2` means this instance is truncated and iou with whole gt < 0.7, you'd better abandon it.
2. Instances with diffcult > 1 will not be included during evaluation.
3. Official codes don't support remove bg files, you can achieve it via API I provided here.
4. If you want to save PR-curve result, check codes in dota_evaluation_task1(`eval_map`)

## modification
* DOTA.py
  some annotations.
* dota_utils.py
  add `detections2Task1` func which help to trans your det_res to required format.(line80 and last).
* ResultMerge_multi_proces.py
  repackage and attach some notes.
* dota_evaluation_task1.py
  repackage and attach some notes for better usage.
* ImgSplit_multi_proces.py