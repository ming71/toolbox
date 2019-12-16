import os
import sys

'''
 读取图片的名称并写入一个文件里
 一个名字一行
'''

print(' processing......\n')

path = r"/media/xiaoming/CHASER/yolo/CAR/JPEGImages" 

files= os.listdir(path) #得到文件夹下的所有文件名称
for file in files:
    with open(path+'/'+file,'r') as f:
        f.write(file[:-4]+'\n')
#        lines=f.readlines() 
#        with open(path+'/'+file,"w") as fd:
#            for line in lines:
#                if  '),10'  in l:          
#                    fd.write(l)
#    #删除空文件          
#    if os.path.getsize(os.fspath(path+'/'+file)) == 0:  
#        os.remove(os.fspath(path+'/'+file)) 
#        #os.remove(os.fspath("/py/NWPU VHR-10 dataset/ground truth/394.txt")) 
'''
另一种操作
'''
with open(path,'r') as f:
        contents=f.read()
        objects=contents.split('\n')#分割出每个物体
        for i in range(objects.count('')):#去掉空格项
           objects.remove('')
        num=len(objects)#物体的数量
        for objecto in objects:
            xmin=objecto.split(' ')[7]  #以','进行分割，得到的片段存入列表，[n]代表列表的第n个分片


print('finished!')




