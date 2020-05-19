'''
2019.10.24
功能： 查询词频和笔画数
实现： 非常低级的爬虫和字符串匹配
问题： 
    1. 编码确认无误。windows下由于源文件不同，txt经常会出现乱码，需要选择合适的encode方式，先查看读取是否有误再运行
    2. 字符串匹配bug。往往是多次匹配或者提前匹配。
    3. 不同操作系统的txt显示乱码。同1，嫌麻烦可以直接复制到别的地方然后粘贴。
'''
import requests
from urllib.parse import quote
import os
import glob
from tqdm import tqdm
import sys

def frequence_query(src_path):
    # 文件夹输入，将多输入和检索结果丢在一起比较方便
    if os.path.isdir(src_path):     
        files = sorted(glob.glob(os.path.join(src_path, '*.*')))
        for path in files:
            file_name = os.path.split(path)[1]
            save_path = os.path.join(src_path,os.path.splitext(file_name)[0]+'_result.txt')

            if os.path.exists(save_path):
                os.remove(save_path)
                files.pop(files.index(save_path))

            # with open(path,'r', encoding='utf-8',errors='ignore') as f:
            with open(path,'r', encoding='gbk',errors='ignore') as f:
                with open(save_path,'a') as fw:
                    contents = f.readlines()
                    import ipdb; ipdb.set_trace()
                    words = [word.strip('\n').strip('') for word in contents]
                    assert len(contents)==len(words), '有空行，检查一下是不是输入有问题'

                    for word in tqdm(words):
                        urlencode = quote(word)
                        url='http://bcc.blcu.edu.cn/zh/search/2/' + urlencode
                        req = requests.get(url) 
                        result=req.text[req.text.index('\t    共')+6:req.text.index('个结果')].strip(' ') # 搜索目标结果
                        print(word)
                        # import ipdb; ipdb.set_trace()
                        fw.write(result+'\n')

    elif os.path.isfile(src_path):
        print('将文件放到对应的文件夹下即可')
    else:
        print('路径不对，检查一下')


## txt文件内所有汉字无视词语组合直接输出成单列
def strokes_query(src_path):
    # 文件夹输入，将多输入和检索结果丢在一起比较方便
    if os.path.isdir(src_path):     
        files = sorted(glob.glob(os.path.join(src_path, '*.*')))
        for path in files:
            file_name = os.path.split(path)[1]
            save_path = os.path.join(src_path,os.path.splitext(file_name)[0]+'_result.txt')

            if os.path.exists(save_path):
                os.remove(save_path)
                files.pop(files.index(save_path))

            # with open(path,'r', encoding='utf-8',errors='ignore') as f:
            with open(path,'r', encoding='gbk',errors='ignore') as f:
                with open(save_path,'a') as fw:
                    contents = f.read()
                    words = contents.replace('\n','').strip('')

                    for word in tqdm(words):
                        url='http://bishun.strokeorder.info/mandarin.php?q=' + word
                        req = requests.get(url) 
                        content = req.content.decode('utf-8')
                        result = content[content.index('的笔画数')+9 : content.index('<br><br>\n\n<b>相关汉字的笔顺')]
                        # print(word)
                        # import ipdb; ipdb.set_trace()
                        fw.write(result+'\n')

    elif os.path.isfile(src_path):
        print('将文件放到对应的文件夹下即可')
    else:
        print('路径不对，检查一下')



if __name__ == "__main__":
    src_path = 'D:\\application\Jupyter Notebook\spider\search'

    # frequence_query(src_path)
    strokes_query(src_path)



