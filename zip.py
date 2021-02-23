import os,os.path
import zipfile




## Method 1: python
def zip_dir(dirname,zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(os.path.join(root, name))
        
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        #print arcname
        zf.write(tar,arcname)
    zf.close()
 
 
def unzip_file(zipfilename, unziptodir):
    if not os.path.exists(unziptodir): 
        os.mkdir(unziptodir)
    zfobj = zipfile.ZipFile(zipfilename)
    for name in zfobj.namelist():
        name = name.replace('\\','/')
       
        if name.endswith('/'):
            os.mkdir(os.path.join(unziptodir, name))
        else:            
            ext_filename = os.path.join(unziptodir, name)
            ext_dir= os.path.dirname(ext_filename)
            if not os.path.exists(ext_dir) : 
                os.mkdir(ext_dir)
            outfile = open(ext_filename, 'wb')
            outfile.write(zfobj.read(name))
            outfile.close()
 

## Method 2 : linux cmd
# 有些比赛要求指定的压缩文件目录，method1无法方便地创建递归目录；
# zip命令-r可以递归创建，但是会带上父目录，为了避免这样，cd到指定目录进行创建。安装的话：
# sudo apt-get update && sudo apt-get install zip
# 下面创建的是名为submission.zip的压缩文件，解压后一级目录test下是所有的label
os.system('cd {} && zip -r -q  {} {} '.format(root_dir, 'submission.zip', 'test'))

# update: -j 一步到位不带父目录
os.system('cd {} && zip -j  {} {} '.format(root_dir, 'submission.zip', 'test'))



if __name__ == '__main__':

    ## zip files
    # dirname = 'ship'
    # zipfilename = 'ship.zip'
    # zip_dir(dirname, zipfilename)

    ## unzip files
    # zipfilename = 'ship.zip'
    # unziptodir = 'ship'
    # unzip_file(zipfilename, unziptodir)