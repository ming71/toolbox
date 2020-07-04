import os 

# 注意： os.system的命令不会保留，所以必须一次执行完全部命令。如第一句是cd xxx，第二句python xxx.py，必须写在一句话里，否则第一句执行完程序不会真的进入文件夹等待第二句
# 返回值正常是0，可用于检测
# 如下实例：
# status=os.system('cd  ~/ev/yolo-master/keras-yolo3 && python yolo.py')


# ---------简单例子-------------
cmd1 = 'ls'
cmd2 = 'cd /py/pic && ls'
cmd3 = 'cd /py && python '
os.system(cmd2)

# ---------带变量传入-------------
# 使用格式化方法format(想了好久，发现这样可行！)将一个文件目录下的文件移到另一个目录
xml_src = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_src'
xml_dst = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_dst'

os.system('cp -r {} {}'.format(xml_src+'/* ',xml_dst))  # 注意src需要加'/*'，如果只移动src的图片，则/*.jpg


