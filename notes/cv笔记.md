[TOC]

---



#### 一、opencv

---

##### imshow的图片闪退

需要加上这句：

> cv2.waitKey(0)

---

##### 手动固定显示窗口的位置

> cv2.moveWindow( windowname, x, y )

其中windowname是cv2.imshow窗口的名字字符串，如'pic'；x和y是窗口相对屏幕左上角的绝对位置

-----

##### 图像mask的方法

`cv2.fillConvexPoly()`或`cv2.fillPoly()`填充多边形实现图像的多边形mask。如果想对原图mask掉特定区域，先创建像素0蒙版，然后用上面函数指定好多边形区域，填充为1，与原图相乘即可mask掉。[使用方法实例](https://drivingc.com/p/5af8f4002392ec4f4727ccc8)。

---

##### putText加说明文字
> cv2.putText(img, str, point, cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细.

【注】需要说明的是，这个地方老会报错，为是否含有默认值的变量传入顺序以及格式问题导致的。

1. 参数顺序问题：解决方案是加上font参数（如上）；或者采用形参指定一下各参数的含义
2. point格式必须是二元tuple

---

##### 最小外接矩形以及xywha转四点坐标

程序实现：[地址](http://www.1zlab.com/wiki/python-opencv-tutorial/opencv-coutour-rect/)

```
cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 必须是array数组的形式
rect = cv2.minAreaRect(cnt) # （中心(x,y), (宽,高), 旋转角度）【注意是tuple】
box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标
【注意】上面的输入输出rect都是tuple，形如：cv2.boxPoints(((cx,cy),(w,h),a))
```

注意事项：

1. opencv中的角度定义：x轴的正半轴逆时针旋转碰到矩形第一条边是width，转过的角度是theta，其中顺时针为正
2. 输入直接转化为numpy数组往往会报错，因为cv2支持的`cv2.minAreaRect`输入是`int32/float32`，而直接转换得到的是`float64`，所以转换时：`cnt = np.array(cnt, dtype=np.float32)`



---



#### 二、matplotlib

---

##### plt.plot显示图像有色差

因为opencv的`cv2.imread()`读取的BGR, 而pyplot是RGB，注意用transpose换轴：

> img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB

---

##### 手动固定生成窗口位置

调整图片生成的位置（距左上xy分别480 110）:

>  mngr = plt.get_current_fig_manager()    
>  mngr.window.wm_geometry("+480+110")

---





#### 三、通用知识

---

#####  不同库的读取格式问题

opencv，matplotlib和PIL库的读取图像格式不同。

* plt.imread和PIL.Image.open读入的都是RGB顺序，而cv2.imread读入的是BGR顺序

* opencv和matplotlib读取图像尺寸都是是shape=（H，W，C）；PIL是shape=（W，H）

---

##### 数据类型的转换

读取出来的格式多种多样，如果不统一在对应框架的类型下，显示时会有bug。

* opencv：数组格式为np.unit8，查询方式为img.dtype，转换方式为:            

  > data = np.array(data,dtype='uint8')  # 第一种            
  >
  > data = np.uint8(data)  # 第二种

* matplotlib：数据格式为int32，转换一下：

  > img.astype(np.int32)

---

##### 图像仿射变换

* 变换矩阵M

首先将复杂变换分解成基本变换（平移、旋转、剪切等），再按照顺序将基本操作的变换矩阵相乘。原则：变换都是左乘矩阵，注意顺序。只有旋转等少数操作存在中心点问题。【注意】旋转操作逆时针为正

实际确定单个变换阵M时有现成的opencv函数，只需输入变换参数会输出变换矩阵。

* 变换公式

获得上述变换矩阵M：$\left[\begin{array}{lll}x_{1} & y_{1} & z_{1} \\ x_{2} & y_{2} & z_{2}\end{array}\right]$，对于任意输入点$(x, y)$，输出位置为$x^{\prime}=x \cdot x_{1}+y \cdot x_{1}+z_{1}, \quad y^{\prime}=x \cdot x_{2}+y \cdot x_{2}+z_{2}$。

具体做法：先写出单个的变换阵，然后组合（python中通过`@`字符实现矩阵乘法），得到综合变换矩阵M后，取前两行（第三行为001不管），进行上面的迭代计算即可。

参考：[程序实现](https://zhuanlan.zhihu.com/p/60659854)，[单个变换](https://blog.csdn.net/a396901990/article/details/44905791)

---

