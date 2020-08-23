## We ended up not using this file. It is submitted only as a testimony to our hard work.

import pandas as pd
from os import listdir
from os.path import isfile, join
cur_path = r"C:\Users\Nitzan\Documents\TOV\IML_Hackathon_2019\trainData"


def first_read(path_dir):
    onlyfiles = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
    data_frame = pd.read_csv(path_dir + "\\" + onlyfiles[0])
    for i in range(1,len(onlyfiles)):
        temp = pd.read_csv(path_dir + "\\" + onlyfiles[i])
        data_frame = data_frame.append(temp,ignore_index=True)
    data_frame["tweet"]=data_frame["tweet"].apply(lambda x: x[2:len(x)-2])

    return data_frame
