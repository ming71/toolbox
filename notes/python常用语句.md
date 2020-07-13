[TOC]
---




#### 1. 列表list操作

----

##### 排序
**注意**：文件夹读取相关的最好都先排序，否则zip排序出BUG不好找
文件读取list往往为乱序的，为了将img和xml对应，可以对两个list都采用排序函数如 img_files.sort()，然后再将两者zip后进行遍历。

> list.sort( )（返回值为None， 可传递参数reverse=True逆序排序）  
> sorted( list )

---



##### list转str

```
str = "".join(list)
str = " ".join(list)
str = ".".join(list)
str = "\n".join(list)
```



#### 2. 文件相关的操作

---

##### 获取当前路径

```
root_dir = os.getcwd()
```

---

##### 路径和文件名分离

方法：
> os.path.split( )  
> os.path.splitext( )  

例子：
> file_path = "D:/test/test.py"        
> (filepath,tempfilename) = os.path.split(file_path)                ('D:/test', 'test.py')   
> (filename,extension)      = os.path.splitext(tempfilename)   ('test', '.py')

---

##### 文件复制

> import shutil        
> shutil.copyfile('C:\\1.txt' ,  'D:\\1.txt')

---

##### glob获取文件绝对路径

> files = sorted( glob.glob( os.path.join( path, '**.*' ) ) )     

'.'是匹配项，可替换为.jpg .txt等制定特定类型，，从而简单获取特定类型的文件绝对路径。[参考链接](https://www.jianshu.com/p/542e55b29324)

---

##### 前缀后缀判断
> str.startswith( 'this' )

> str.endswith('.jpg')

---








####  3. 字符串str操作

---

#####  分割切片

在str中去掉字符‘a’：

> str.strip('a')

str以所有的 'a' 字符为界进行切割返回切片，注意'a'也没有了

> str.split('a')

---









####  4. 其他问题

---

##### 自定义的py文件import后找不到路径

多半是`__init__.py`文件文件没写或者不对，[入口](https://my.oschina.net/wangjiankui/blog/188698)

---







#### 5. 注意事项

---

##### 文件命名

<u>py文件起名不要和pip package一样</u>，不然运行的时候会报错，找都找不到（因为import的时候会优先从本地import而不是pip）

---





#### 6. 一些操作合集

---

#####  判断数据类型
>isinstance(feature_maps,list)

---

##### 集合set

利用好set的操作可以简化程序，并且加速运算，如交并补差等，参加：[set指令大全](http://c.biancheng.net/view/4400.html)

但是注意：**集合中的元素不会有重复**！！
```python
a=[1,1,2,3,4,5,5]
b=set(a)
b

{1, 2, 3, 4, 5}
```

---

##### 利用bool矩阵进行索引
```python
(1) i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1) pred = pred[i]     
(2) class_conf = class_conf[i]         
(3)dc = pred[pred[:, -1] == c]   # select class c
```

多说一句，pytorch的torch.where支持很多运算了，实在没有再用bool判断（因为这个可能导致梯度计算的矛盾）

---

##### 执行字符串表达式

> eval()和exec()

使用场景：

1.字符串计算式是无法别识别成我们常用的float型的，所以如配置文件等地读取出来是个字符串，使用eval()可以直接将字符串进行计算得到结果。    

2.需要在程序中调用执行命令

区别：可以看出，eval和exec相似度很高，都是执行字符串表达式，但是 eval() 是一个函数，需要变量来接收函数的执行结果；而exec()的字符串内部变量和外界相同，可以直接执行，返回None.

---

***号省略变量**        

单引号为多变量，`**`为多元素字典变量，例如：

> for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred)        

特殊：多返回值的忽略   

> img, *_ = letterbox(img0, new_shape=self.img_size)    

---

##### filter()过滤序列

> filter(function, iterable)
>

将可迭代对象传入这个函数，然后内部逐个将迭代器迭代到函数中，返回bool迭代器。俩字好用，参考[教程](https://www.runoob.com/python3/python3-func-filter.html)。给个例子：

> def is_odd(n):
> 	return n % 2 == 1
> tmplist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
> newlist = list(tmplist)
> print(newlist)

---

##### 保留固定位数的小数

```
from decimal import Decimal
a = 1 / 3         0.3333333333333333
a = Decimal(a).quantize(Decimal('0.00'))       0.33
a = Decimal(a).quantize(Decimal('0.0000'))   0.3300
```

---

##### tqdm设置描述语

```
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
	pbar.set_description("Processing %s" % char)
```

