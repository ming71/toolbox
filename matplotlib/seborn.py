import os
import numpy as np 
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

valid_idx = (bf_ious>0.5)&(scores>0.5)
x = bf_ious[valid_idx]
y = scores[valid_idx]
# 设置风格
sns.set_context("paper")
sns.set(font_scale=1.7)	# 坐标轴字体
sns.set_style("white") 
# kind有三种方式reg--绘制拟合直线；hex--绘制蜂窝状散点图；kde--绘制核密度图
# ratio越大，边栏越窄
df = pd.DataFrame(np.stack((x,y),-1),columns=["iou", "conf"])
g = sns.jointplot(x="iou", y="conf", data=df, kind="kde",space=0, ratio=10, height=9)   # color=xxx
g.set_axis_labels("output IoU","Classification Confidence",fontsize=30,fontfamily='Times New Roman')
# g.plot_joint(plt.scatter, c="r", s=5, linewidth=1, marker=".")     # 把散点图叠加画到高斯图上去
# sns.despine()
g.savefig(base_partial_save_path, dpi = 600)
## 显示pearson相关系数
rsquare = lambda a, b: stats.pearsonr(a,b)[0]**2
g = g.annotate(rsquare, template='{stat}:{val:.2f}', stat='$R^2$', loc='upper left', fontsize=12)
## 画装逼的热图
# f, ax = plt.subplots(figsize=(12, 9))
# cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
# sns.kdeplot(df['iou'], df['conf'], cmap=cmap, n_levels=60, shade=True)
plt.show()
# 此外，如果想设置图幅大小偏移等，可以直接用plt.xxx属性，因为seaborn本质上是基于matplotlib的，向下兼容
