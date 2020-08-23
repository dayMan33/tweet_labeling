## We ended up not using this file. It is submitted only as a testimony to our hard work.

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
mypath = r'C:\Users\Nitzan\Documents\TOV\IML_Hackathon_2019\tweetData'
validation_destpath = r'C:\Users\Nitzan\Documents\TOV\IML_Hackathon_2019' \
                r'\validationData'
train_destpath = r'C:\Users\Nitzan\Documents\TOV\IML_Hackathon_2019\trainData'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def read_csv_files():
    '''
    read the csv_files save test data and save it in file name.(use only once)
    :return:
    '''
    for postfix_path in onlyfiles:
        cur_dtf = pd.read_csv(mypath+"\\"+postfix_path)
        validation_dtf = cur_dtf.sample(frac=0.15)  # sample test
        fit_dtf = cur_dtf.drop(validation_dtf.index)
        # fit = fit_dtf.as_matrix()
        # test = validation_dtf.as_matrix()
        # print(postfix_path+str(fit.shape)+":::::"+str(test.shape)) #for
        # validation
        postfix_path = postfix_path[:postfix_path.find(".")-1]
        validation_dtf.to_csv(path_or_buf=validation_destpath + "\\" + postfix_path
                                          +"_validation.csv",index=False)
        fit_dtf.to_csv(path_or_buf=train_destpath+"\\"+postfix_path
                                   +"_train.csv",index=False)
if __name__ == '__main__':
    read_csv_files()