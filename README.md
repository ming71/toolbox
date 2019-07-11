# toolbox
All kinds of utils for format convertion or something else (described in readme files)</br>
Each python file matches some samples inside.</br>
More details are marked in the python files as **annotation** or **`README.md` file** inside the folder.</br>


## converter 
Some conversion files included here,I'll update it if necessary.</br>
* **txt2voc**</br>
The tool inside helps you to conver txt in certain format to voc format.<br>
Provided txt labels ground truth in form of `NWPU VHR-10` , it's not a hard job to change related code for your owm labels.<br>
NWPU VHR-10's label is showed as follow:<br>
```
2,(270,296),(358,685)
2,(366,278),(453,661)
4,(541,633),(727,698)
```
* **labelme2COCO**<br>
It's a bit hard to finish this work , cause the widespread wrong version about it , I hate plagiarism :)<br>
This tool helps you to convert json file created by labelme to COCO style for training .<br> 
**Attentions:**<br> 
    1.If you want to label the segmentation mask , there is `no need` to label bbox!(Or bugs arised)<br> 
    2.When using labelme , pls named mask according to COCO format,such as `vehicle_car_1`.(supercategory,category,instance id)<br> 
* **voc2coco**<br>


## voc_extraction
* **extraction**  
User-friendly work for extracting imformation from xml files. And an example is attached for better understanding.<br>

## data augmentation
* **augmentation**
Various kinds of data augmentation implementions as well as some demos are concluded inside .

## other tools
* **data_partition**<br>
Helpful of partition of dataset.<br>
* **img_format_trans**<br>
A simple tool for conversion of image format.<br>
* **visdom-train-example**<br>
A example for training while monitoring on loss and ac curve.More details and attention have been attached to file inside.<br>
* **python-cmd**<br>
Linux command execution through python file , provided as a easy but sufficient demo.<br>
* **drawbox**  
Useful tool for drawing bbox through providied points.The only customed part is your points obtaining function.  
