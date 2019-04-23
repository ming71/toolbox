#2018年12月17日 
#author: ming71

‘’‘
将含有全部文件名的name文件（不带路径）划分数据集得到训练验证测试的txt(不是最终的，最终还要运行voc_label)
只需要改变三个参数：
train_ratio=0.8
val_ratio=0.1
total=2055 
’‘’
import os 
import sys
import random



path = r"/media/xiaoming/ming/env/dataset/ship_dataset/label" 

##获取全部名称
#files= os.listdir(path) #得到文件夹下的所有文件名称
#for file in files:
#    with open(path+'/'+'name','a') as f:
#        f.write(file[:-4]+'\n')
        
#划分数据集
save_path=r"/media/xiaoming/ming/env/dataset/ship_dataset" 
random.seed(66666)
train_ratio=0.8
val_ratio=0.1
total=2055      #数据集总数
#生成打混的随机序号:0~(total-1）（打混后与索引不一一对应）
serials=random.sample([i for i in range(0,total)],len([i for i in range(0,total)]))  
files= os.listdir(path)     #得到文件夹下的所有文件名称(索引为0~(total-1))
with open(save_path+'/'+'train.txt','a') as f:
    for i in range(int(total*train_ratio)):
        f.write(files[serials[i]][:-4]+'\n')
with open(save_path+'/'+'val.txt','a') as f:
    for i in range(int(total*train_ratio),int(total*train_ratio)+int(total*val_ratio)):
        f.write(files[serials[i]][:-4]+'\n')
with open(save_path+'/'+'test.txt','a') as f:
    for i in range(int(total*train_ratio)+int(total*val_ratio),total):
        f.write(files[serials[i]][:-4]+'\n')        
    
#with open (path) as f:
#    print(f)
#for file in files:
#    with open(path+'/'+'name','a') as f:
#        f.write(file[:-4]+'\n')
#        line=f.readlines() 
#        with open(path+'/'+file,"w") as fd:
#            for l in line:
#                if  '),10'  in l:          
#                   fd.write(l)


#    #删除空文件          
#    if os.path.getsize(os.fspath(path+'/'+file)) == 0:  
#        os.remove(os.fspath(path+'/'+file)) 
#        #os.remove(os.fspath("/py/NWPU VHR-10 dataset/ground truth/394.txt")) 

print('finished!')

