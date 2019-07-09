import os 

# ---------简单例子-------------
cmd1 = 'ls'
cmd2 = 'cd /py/pic && ls'
cmd3 = 'cd /py && python '
os.system(cmd2)

# ---------带变量传入-------------
# 使用格式化方法format(想了好久，发现这样可行！)
path = '/py/pic'
os.system('cd {} && ls'.format(path) )
