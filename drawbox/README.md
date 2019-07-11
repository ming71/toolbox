## DrawBox功能  
根据输入的点在原图像素矩阵上上绘制**多边形**并显示或输出。  


## 简单demo    
直接输入一个图片路径和点列即可绘制图像。    
简单demo提供测试效果，可以直接参见最下面的效果图。


## 实际程序  
* get_points    
从xml等标注文件中读取得到点的坐标信息    
**自定义部分**；根据具体的标注自行改动获取不同的信息；输出形式尽量统一为return处的格式，避免后面drawbox改动
* drawbox  
直接绘图即可，可选是否**存储图片**并输入路径；默认不存储只显示


## 结果展示
用DOTA的图片crop和affine变换后读取变换的坐标绘图得到如下bbox结果：
![](https://github.com/ming71/toolbox/blob/master/drawbox/drawbox_screenshot_11.017.2019.png){:height="60%" width="60%"}

![](https://github.com/ming71/toolbox/blob/master/drawbox/drawbox_screenshot_11.07.019.png){:height="60%" width="60%"}

![](https://github.com/ming71/toolbox/blob/master/drawbox/drawbox_screenshot_11.07.2019.png){:height="60%" width="60%"}
