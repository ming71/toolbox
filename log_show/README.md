## Function 
The code help visualize log `txt` file . Many detectors output its training info via json/txt  file , thus some tool for its utility is required .

## Demo 
* **log.py**  

  Code for log info visualization.（only `txt` format support at present）  
  1. Visualization for training log file .
  2. Support multifile compare.

* **result.txt**  

  A log file in `.txt` format.


## Result    
Parse the `result.txt`   and visualize the results of `mAP` , `Loss`  as follows:

<div align=center><img width="800" height="480" src="https://github.com/ming71/toolbox/blob/master/log_show/log.png"/></div><br/>  
Multifile supported:  
<div align=center><img width="800" height="480" src="https://github.com/ming71/toolbox/blob/master/log_show/multi-log.png"/></div><br/>
