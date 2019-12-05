## img_aug
Quick implemention for data augmentation via `imgaug`,related usage refer to [rotate-yolo](https://github.com/ming71/rotate-yolo).  
(Or you can run the demo I leave behind the code,it's easy to replicate the effect.)

## augentation
This code is completed for `DOTA dataset` , thus specific modification is necessary for your own label.
1. Supported methods ：affine（sheer、translation、scale、rotation）、hsv augmentation、noise、flip、blur  
2. You're supposed to run **one** method everytime , and each aug_functioin wil output img files as well as corresponding xml_label files .
3. Customed: you need to rewrite `read_xml` and `rewrite_label` function for correct label process.  


## Result 
* **HSV**  
<p float="left">
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/raw.jpeg" width="350" />  
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/hsv.jpeg" width="350" /> 
</p>

* **Affine**
<p float="left">
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/raw.jpeg" width="350" />  
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/affine.jpeg" width="350" /> 
</p>

* **Flip**
<p float="left">
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/raw.jpeg" width="350" />  
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/flip.jpeg" width="350" /> 
</p>

* **Noise**
<p float="left">
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/raw.jpeg" width="350" />  
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/noise.jpeg" width="350" /> 
</p>

* **Blur**
<p float="left">
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/raw.jpeg" width="350" />  
  <img src="https://github.com/ming71/toolbox/blob/master/data_augmentation/blur.jpeg" width="350" /> 
</p>
