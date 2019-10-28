# toolbox
All kinds of utils for format convertion or something else (described in readme files)</br>
Each python file matches some samples inside.</br>
More details are marked in the python files as **annotation** or **`README.md` file** inside the folder.</br>


## converter 
Some conversion files included here,I'll update it if necessary.</br>
* **txt2voc**</br>
The tool inside helps you to conver txt in certain format to voc format.  
Provided txt labels ground truth in form of `NWPU VHR-10` , it's not a hard job to change related code for your owm labels.  
NWPU VHR-10's label is showed as follow:  
```
2,(270,296),(358,685)
2,(366,278),(453,661)
4,(541,633),(727,698)
```

* **voc2txt**  
User-friendly work for extracting imformation from xml files. And an example is attached for better understanding.  

* **labelme2COCO**  
It's a bit hard to finish this work , cause the widespread wrong version about it , I hate plagiarism :)  
This tool helps you to convert json file created by labelme to COCO style for training.     
**Attentions:**   
    1.If you want to label the segmentation mask , there is `no need` to label bbox!(Or bugs arised)   
    2.When using labelme , pls named mask according to COCO format,such as `vehicle_car_1`.(supercategory,category,instance id)    

* **voc2coco**    
Just use it.
  
* **voc2yolo**    
Yolo format: class_id + Normalized xywh (id markded from 0) for each row.


## dataset
* **data_partition**  
Helpful of partition for dataset.  

* **shuffle**  
Randomly generate small subdataset from the dataset.   

* **generate_imageset**  
Generate trainval setting files.  
(Two mode included:yolo and voc)  

* **operate_on_datasets**  
Operations on dataset , such as copy, label matching.   


## data augmentation
* **augmentation**   
Various kinds of data augmentation implementions as well as some demos are concluded inside .


## drawbox
* **drawbox**  
Useful tool for drawing bbox through providied points.The only customed part is your points obtaining function. 


## matplotlib
Provided a template for plotting 2D and 3D figure.


## log_show
Visualization for training log files.


## Spider
Easy implemention for crawling info from website.


## other tools
* **img_format_trans**    
A simple tool for conversion of image format.  

* **visdom-train-example**  
A example for training while monitoring on loss and ac curve.More details and attention have been attached to file inside. 

* **python-cmd**  
Linux command execution through python file , provided as a easy but sufficient demo.  
Beside,`exec` commond also helps for many cases ,take good use of it.(such as [drawing pictures](https://github.com/ming71/toolbox/blob/b473ea001c2498fe927115d0c4a66d1cd4e30a7f/matplotlib/HuMonent.py#L172))  


* **crop_bbox_and_save**   
Crop bbox area from raw image and save for other usage.


 * **K-means**   
K-means  implement for box clustering.

* **skewiou**  
For skewiou calc.
