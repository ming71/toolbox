####################
# 功能：读写excel
# 任务：
#   将'words'文件的词语在目标文件'catalog'中索引，
#   得到该词语的汉语等级编号【读】；
#   并写入'word'文件的对应位置【写】。
#   支持输入文件有重复
####################

# pd.read_excel参数说明：参见解析：https://blog.csdn.net/brucewong0516/article/details/79096633
# header: default=0，也就是有title，在索引每个行时都会带上title名和列数；如果统计数据没有title名，将header=None（但是title的占行还是在）

# dataFrame转dict：
#   df.set_index('id')['value'].to_dict() 构成一个键值对，其中'id'就是key，可以对应列名，如果没有，护着在读取时header=None，'id'=0即可；
#   同理value也可指定哪一列作为value，最简单的方法直接为1，即下一列；
#   例如：     df.set_index(0)[1].to_dict()


import os
import pandas as pd
from copy import deepcopy

words_file = 'words.xlsx'
catalog_file = 'catalog.xlsx'
res_file = 'result.xlsx'

if os.path.exists(res_file):
    os.remove(res_file)

df_words = pd.read_excel(words_file, header=None)
df_catalog = pd.read_excel(catalog_file, header=None)
catalog = df_catalog.set_index(0)[1].to_dict()  # dict

### 输入文件自动去重索引
# words   = df_words.set_index(0).T.to_dict()  # dict
# if len(df_catalog)!= len(catalog) or len(words)!= len(df_words):
#     print('There exists duplicate in data! Check it !')
# for word in words.keys():
#     level = catalog[word] if word in catalog.keys() else 'NA'
#     words[word] = level
# res = pd.DataFrame.from_dict(words, orient='index')
# res.to_excel(res_file)

### 输入文件不去重，直接索引
words = df_words.to_dict(orient='index')    # dict
res = deepcopy(words)
for row_id in words:  # 行编号，索引得到的k-v为：{column_id,item}如{0: '休闲'}
    word = words[row_id]
    for column_id, item in word.items():
        level = catalog[item] if item in catalog.keys() else 'NA'
        res[row_id][1] = level
res = pd.DataFrame.from_dict(res, orient='index')
res.to_excel(res_file)
