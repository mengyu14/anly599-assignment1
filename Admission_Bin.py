

import pandas as pd

mydf = pd.read_csv('Admissions_id.csv',sep=',', encoding='latin1')
num_list = []
for i in range(len(mydf)):
    num_list.append(i)
mydf['id'] = num_list
##mydf.to_csv("Admissions_id.csv", index = False)

bin_list = []
for Exps in mydf['WorkExp']:
    if (Exps < 1.5):
        bin_list.append('Minimal Experienced')
    elif (Exps < 3):
        bin_list.append('Somewhat Experienced')
    else:
        bin_list.append('Experienced')
mydf['WorkExpStats'] = bin_list
mydf.to_csv("Admissions_Final.csv", index = False)