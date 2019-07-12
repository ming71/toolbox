## Function 
The code help to draw bbox on the original pixel matrix based on the input points ,  and display or output the result.
Implemented in form of **polygon** by opencv-python.

## demo    
Draw the boxes on image by directly entering a picture path and point column.   

## DrawBox   
* get_points    
Coordinate information of points is read from annotation files such as XML    
**Customed part**；Obtain different information <u>according to the specific label</u>; **the output should be in the same format as the return to avoid subsequent drawbox changes**  

* drawbox (optional settings) 
1. Input a single picture or folder
2. Save results or just display
3. Related drawing properties


## Result
Applying the transformation of crop and affine on DOTA pictures, the transformation coordinates were mapped to get the following bbox results:
<div align=center><img width="600" height="480" src="https://github.com/ming71/toolbox/blob/master/drawbox/drawbox_screenshot_11.017.2019.png"/></div><br/>

<div align=center><img width="600" height="480" src="https://github.com/ming71/toolbox/blob/master/drawbox/drawbox_screenshot_11.07.2019.png"/></div><br/>

<div align=center><img width="600" height="480" src="https://github.com/ming71/toolbox/blob/master/drawbox/drawbox_screenshot_11.07.019.png"/></div><br/>
