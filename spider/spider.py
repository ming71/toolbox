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
import time

def frequence_BCC_query(src_path):
    # 文件夹输入，将多输入和检索结果丢在一起比较方便
    if os.path.isdir(src_path):     
        files = sorted(glob.glob(os.path.join(src_path, '*.*')))
        for path in files:
            file_name = os.path.split(path)[1]
            save_path = os.path.join(src_path,os.path.splitext(file_name)[0]+'_result.txt')

            if os.path.exists(save_path):
                os.remove(save_path)
                files.pop(files.index(save_path))

            with open(path,'r', encoding='utf-8',errors='ignore') as f:
            # with open(path,'r', encoding='gbk',errors='ignore') as f:
                with open(save_path,'a') as fw:
                    contents = f.readlines()
                    words = [word.strip('\n').strip('') for word in contents]
                    assert len(contents)==len(words), '有空行，检查一下是不是输入有问题'

                    for word in tqdm(words):
                        urlencode = quote(word)
                        url='http://bcc.blcu.edu.cn/zh/search/0/' + urlencode
                        req = requests.get(url) 
                        result=req.text[req.text.index('\t    共')+6:req.text.index('个结果')].strip(' ') # 搜索目标结果
                        print(word)
                        # import ipdb; ipdb.set_trace()
                        fw.write(result+'\n')

    elif os.path.isfile(src_path):
        print('将文件放到对应的文件夹下即可')
    else:
        print('路径不对，检查一下')


def frequence_CCL_query(src_path):
    # 文件夹输入，将多输入和检索结果丢在一起比较方便
    if os.path.isdir(src_path):     
        files = sorted(glob.glob(os.path.join(src_path, '*.*')))
        for path in files:
            file_name = os.path.split(path)[1]
            save_path = os.path.join(src_path,os.path.splitext(file_name)[0]+'_result.txt')

            if os.path.exists(save_path):
                os.remove(save_path)
                files.pop(files.index(save_path))

            with open(path,'r', encoding='utf-8',errors='ignore') as f:
            # with open(path,'r', encoding='gbk',errors='ignore') as f:
                with open(save_path,'a') as fw:
                    contents = f.readlines()
                    words = [word.strip('\n').strip('') for word in contents]
                    assert len(contents)==len(words), '有空行，检查一下是不是输入有问题'

                    for word in tqdm(words):
                        # import ipdb; ipdb.set_trace()
                        url='http://ccl.pku.edu.cn:8080/ccl_corpus/search?q=' + \
                            word + \
                            '&start=0&num=50&index=FullIndex&outputFormat=HTML&encoding=UTF-8&maxLeftLength=30&maxRightLength=30&orderStyle=score&LastQuery=&dir=xiandai&scopestr='
                        req = requests.get(url) 
                        # import ipdb; ipdb.set_trace()
                        if req.text.find('很抱歉，没有找到符合检索条件的实例') != -1:
                            result = '0'
                        else:
                            result=req.text[req.text.index('>共有 ')+7:req.text.index('条结果')-5].strip(' ') # 搜索目标结果
                        print(word + '  ' + result)
                        # import ipdb; ipdb.set_trace()
                        fw.write(result+'\n')

    elif os.path.isfile(src_path):
        print('将文件放到对应的文件夹下即可')
    else:
        print('路径不对，检查一下')


## txt文件内所有汉字无视词语组合直接输出成单列
## merge关键字是词的长度，将字的结果相加得到词的组合。例如merge=3，每个词都是3个字进行组合相加
# 文件夹输入，将多输入和检索结果丢在一起比较方便
def strokes_query(src_path, merge=0, Online=True):
    # 先parse数据库，为离线匹配做准备
    with open('database.txt', 'r',encoding='utf-8') as fb:
        database = fb.readlines()
        strokes = {}
        for charline in database:
            char = charline.split(':')[0]
            stroke = len(charline.split(','))
            strokes[char] = stroke
    if os.path.isdir(src_path):     
        files = sorted(glob.glob(os.path.join(src_path, '*.*')))
        for path in files:
            file_name = os.path.split(path)[1]
            save_path = os.path.join(src_path,os.path.splitext(file_name)[0]+'_result.txt')

            if os.path.exists(save_path):
                os.remove(save_path)
                files.pop(files.index(save_path))

            with open(path,'r', encoding='utf-8',errors='ignore') as f:
            # with open(path,'r', encoding='gbk',errors='ignore') as f:
                with open(save_path,'a') as fw:
                    contents = f.read()
                    words = contents.replace('\n','').strip('').replace(' ','')
                    for word in tqdm(words):
                        try:
                            if Online == True:
                                # Method 1: online
                                url='http://bishun.strokeorder.info/mandarin.php?q=' + word
                                req = requests.get(url) 
                                content = req.content.decode('utf-8')
                                # time.sleep(1)
                                result = content[content.index('的笔画数:</b>')+9 : content.index(' &nbsp;   <b>'+word+'的结构:</b>')]
                            else:
                                # Method 2: offline
                                result = str(strokes[word])
                        except:
                            if word in uncommon_character.keys():
                                result = str(uncommon_character[word])
                            else:
                                import ipdb; ipdb.set_trace()
                                print('生僻字未收录:  ' + word)
                                raise NotImplementedError                                   
                        print(word,result)
                        fw.write(result+'\n')
            
            ## 将输出结果按词语重组
            if merge > 0 :
                word_res = os.path.join(src_path,os.path.splitext(file_name)[0]+'_word_result.txt')
                if os.path.exists(word_res):
                    os.remove(word_res)
                    files.pop(files.index(word_res))    
                with open(save_path,'r') as fs:
                    strokes = [int(x.strip('\n')) for x in fs.readlines()]
                    if merge == 2 :
                        res = [i + j for i, j in zip(strokes[::2], strokes[1::2])]
                    elif merge == 3 :
                        res = [i + j + k for i, j, k in zip(strokes[::3] , strokes[1::3] , strokes[2::3])]
                    elif merge == 4 :
                        res = [i + j + k+l for i, j, k,l in zip(strokes[::4], strokes[1::4] ,strokes[2::4] ,strokes[3::4])]
                    else:
                        print('写一下多个词的合并规则，很简单')
                        raise NotImplementedError       
                with open(word_res,'w') as fm:
                    merged = [str(x)  for x in res]
                    fm.write('\n'.join(merged))

    elif os.path.isfile(src_path):
        print('将文件放到对应的文件夹下即可')
    else:
        print('路径不对，检查一下')



# 数据库未收录字
uncommon_character = {'尬':7, '矩':9, '伙':6, '辑':13}



if __name__ == "__main__":
    src_path = r'C:\Users\xiaoming\Desktop\spider\search'
    # frequence_CCL_query(src_path)
    # frequence_BCC_query(src_path)
    # strokes_query(src_path,merge=2)    
    strokes_query(src_path, merge=4, Online=False)    



