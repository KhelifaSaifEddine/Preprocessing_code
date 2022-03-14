import os
import pandas as pd
import pydicom
import glob
###########recursive function to retrieve dcm files ###############
def recur_f(path1,info):
    path2 = info
    for s in path1:
      for file in os.listdir(s):

        if("dcm" in file):
            if( len(os.listdir(s)) > 1):
              path2.append(s +"/" +file)
        else:
            new_path = glob.glob(s + "/" + file, recursive=True)
            path2 = recur_f(new_path,path2)
    return path2      
