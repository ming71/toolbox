# toolbox
Various tools for computer vision</br>

## converter 
Codes for label format conversion.</br>
* **toCOCO**</br>

  I have achieved the conversion to coco annotations including: `IC15`, `HRSC2016`, `UCAS-AOS`, `NWPU VHR 10`. and `VOC`. Besides, it also supports `labelme` annotations converted to COCO format.

*  **toDOTA**  

  The codes in this part support the conversion of data annotations from `IC15`, `HRSC2016`, `UCAS-AOS`, `NWPU VHR 10` into polygonal form of `DOTA` annotations. And it also supports the preprocessing code of the corresponding datasets and the `.json` files generation codes.

* **toTXT**  

  Convert to the label format required for mAP calculation, the calculation code refers to my [implementation](https://github.com/ming71/toolbox/tree/master/mAP) here (for rbox).    

* **toYOLO**  

  This part will no longer be maintained and it supports both `xml` and `ICDAR` formats.

## dataset  

* **DOTA_devkit**

  Toolkits for DOTA datasets, with some annotations and modification attached. 

* **dataset_partition**  

  Dataset partition for train, val, test part.  
**Note**: `x2` means train + val, `x3` means train + val + test.  
Remember to enlarge val & test set if your dataset is tiny. (such as 6:2:2)   

* **subdataset_generation**  
  
  Division of subset from total dataset, used for hyperparameter adjust.  
(you can regard it as `x1` dataset_partition)  

* **generate_imageset**    

  Generate image absolute path for easy training. Supported data sets include: `IC13`, `IC15`, `HRSC2016`, `DOTA`, `UCAS_AOD`, `NWPU_VHR`, and `VOC`. 

* **operate_on_datasets**    
  Operations on dataset , such as copy, label matching.     

## rotation

Codes for rotated object detection. Supports functions such as cuda rnms, cuda riou, python rnms, pytorch riou calculations.


## data augmentation  
I have implemented some data augmentations such as `Affine`, `HSV transform`, `Blur`, `Grayscale`, `Gamma`, `RandomNoise`, `Sharpen`, `Contrast`, `RandomFlip`. On this basis, imbalanced datasets can be automatically augmented via simple sampling strategy here.


## drawbox
Useful tool for drawing bbox through providied points. The only customed part is your points obtaining function. 

## mAP
Support rbox evaluation and mAP calculation for object detection.

## matplotlib
Provided a template for plotting 2D and 3D figure.

## excel 
Simple examples for excel files processing via pandas.

## log_show
Visualization for training process.


## spider
Easy implemention for crawling info from website.

## visualization

Feature visualization tools.

## Plug-and-play

* asff-fpn
* bam-attention

PRs are welcomed, if you have any questions, you can open an issue or contact me via mq_chaser@126.com. 