import os 

# ---------简单例子-------------
cmd1 = 'ls'
cmd2 = 'cd /py/pic && ls'
cmd3 = 'cd /py && python '
os.system(cmd2)

# ---------带变量传入-------------
# 使用格式化方法format(想了好久，发现这样可行！)将一个文件目录下的文件移到另一个目录
xml_src = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_src'
xml_dst = r'/py/R2CNN-tensorflow/data/VOCdevkit/VOCdevkit_train/clip/aug/xml_dst'

os.system('cp -r {} {}'.format(xml_src+'/* ',xml_dst))  # 注意src需要加'/*'
