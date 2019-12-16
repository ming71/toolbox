import os,os.path
import zipfile
 
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
 
if __name__ == '__main__':

    ## zip files
    # dirname = 'ship'
    # zipfilename = 'ship.zip'
    # zip_dir(dirname, zipfilename)

    ## unzip files
    # zipfilename = 'ship.zip'
    # unziptodir = 'ship'
    # unzip_file(zipfilename, unziptodir)