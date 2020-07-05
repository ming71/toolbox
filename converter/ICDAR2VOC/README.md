Fork from this [repo](https://github.com/zazaliu/ICDAR2PASCAL_VOC).

# ICDAR2PASCAL_VOC

Convert scene text dataset ICDAR2013, ICDAR2015 to PASCAL_VOC dataset

## 使用

### 克隆代码到本地

```py
git clone https://github.com/zazaliu/ICDAR2PASCAL_VOC.git
```
### 安装依赖包
```py
pip install -r requirements.txt
```

### [ICDAR2013 dataset](https://pan.baidu.com/s/1YzU-FSiH8r7h5o3d4rj0cQ) 转化为 PASCAL_VOC dataset 格式
下载 ICDAR2013 dataset 解压放入 ICDAR2013 文件夹，包含：
- 训练图像集：Challenge2_Training_Task12_Images
- 训练标注集：Challenge2_Training_Task1_GT
- 测试图像集：Challenge2_Test_Task12_Images
- 测试标注集：Challenge2_Test_Task1_GT

标注格式：xmin, ymin, xmax, ymax, text

举例：38, 43, 920, 215, "Tiredness"

#### 执行
```py
python ICDAR2013/trans.py
```
生成的数据集保存在 VOC2007 文件夹中

### [ICDAR2015 dataset](https://pan.baidu.com/s/1YzU-FSiH8r7h5o3d4rj0cQ) 转化为 PASCAL_VOC dataset 格式
下载 ICDAR2015 dataset 解压放入 ICDAR2015 文件夹，包含：
- 训练图像集：ch4_training_images
- 训练标注集：ch4_training_localization_transcription_gt
- 测试图像集：ch4_test_images

> 注：ICDAR2015 未提供测试标注集

标注格式：x1,y1,x2,y2,x3,y3,x4,y4,text
其中，x1,y1为左上角坐标,x2,y2为右上角坐标,x3,y3为右下角坐标,x4,y4为左下角坐标。

举例：(### 表示文字无法辨认)
1. 377,117,463,117,465,130,378,130,Genaxis Theatre

2. 374,155,409,155,409,170,374,170,###

#### 执行
```py
python ICDAR2015/trans.py
```
生成的数据集保存在 VOC2007 文件夹中
