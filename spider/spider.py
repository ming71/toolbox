'''
2019.10.24
功能： 自动根据提供内容进行反复搜索查询，省去手工多次输入和记录的麻烦
（花了十来分钟帮女票弄得）

Bug Report:
    没有解决windows的txt在linux下读取bug的程序问题。
    将文本的内容复制到网页上，然后粘贴到linux下新建的txt空白文本即可。
    不同笔记本的文档建立格式不同同理，可以更改提供的coding方式，再不行就复制粘贴，断点调试words输出可识别即可
'''
import requests
from urllib.parse import quote
import os
import glob
from tqdm import tqdm
import sys


if __name__ == "__main__":
    src_path = '/py/toolbox/spider/search'

    # 文件夹输入，将多输入和检索结果丢在一起比较方便
    if os.path.isdir(src_path):     
        files = sorted(glob.glob(os.path.join(src_path, '*.*')))
        for path in files:
            file_name = os.path.split(path)[1]
            save_path = os.path.join(src_path,os.path.splitext(file_name)[0]+'_result.txt')

            if os.path.exists(save_path):
                os.remove(save_path)
                files.pop(files.index(save_path))

            with open(path,'r') as f:   # encoding='utf-8'
                with open(save_path,'a') as fw:
                    contents = f.readlines()
                    words = [word.strip('\n').strip(' ') for word in contents]
                    assert len(contents)==len(words), '有空行，检查一下是不是输入有问题'

                    for word in tqdm(words):
                        urlencode = quote(word)
                        url='http://bcc.blcu.edu.cn/zh/search/2/' + urlencode
                        req = requests.get(url)  
                        import ipdb; ipdb.set_trace()
                        result=req.text[req.text.index('\t    共')+6:req.text.index('个结果')].strip(' ') # 搜索目标结果
                        fw.write(result+'\n')

    elif os.path.isfile(src_path):
        print('将文件放到对应的文件夹下即可')
    else:
        print('路径不对，检查一下')
