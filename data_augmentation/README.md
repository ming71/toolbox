## Notice
1. Supported methods ：affine（sheer、translation、scale、rotation）、hsv augmentation、noise、flip、blur  (to be continued...)
2. Augmentations are implemented randomly , you can adjust the prob .
3. You're supposed to run **one** method everytime , and each aug_functioin wil output img files as well as corresponding xml_label files .
4. Parameters for each transformation are manually set inside the aug_functioin.
5. This code is completed for `DOTA dataset` , thus specific modification is necessary for your own label.

## Customed Part
Customed code for label rewritting is all you need , thus you need to rewrite `read_xml` and `rewrite_label` function for correct label process.  
(Addtionally , DOTA labels objects with 4 points namely 8 coordinates.)


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
