[TOC]

#### 一、常见问题和技巧

---

##### plt.plot显示图像有色差

因为opencv的`cv2.imread()`读取的BGR, 而pyplot是RGB，注意用transpose换轴：
```
img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB
```
---

##### 手动固定生成窗口位置

调整图片生成的位置（距左上xy分别480 110）:

```
mngr = plt.get_current_fig_manager()    
mngr.window.wm_geometry("+480+110")
```

---

##### 一个函数中画多张图分别保存

注意往往plt都是一张画布，后面plot的会覆盖上去，所以正确的做法是想存的时候savefig之后记得关掉画布，下次画图就不会叠加原来的了

    plt.savefig('anchor_clusters.png')
    plt.clf()