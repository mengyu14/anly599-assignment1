import pandas as pd 
import sys
import numpy as np
import matplotlib.pyplot as plt




### capitalize the un-capitalize States
def capstate(mydf):
    mydf.State = mydf.State.str.capitalize()
    return mydf
### remove any wrong value that may occur
def vaildateGPA(mydf):   
    for scores in mydf.GPA:
        if scores < 0 or scores > 4:
            mydf = mydf.drop(mydf.index[mydf['GPA']==scores])
    return mydf
def vaildateLabel(mydf):
    for labels in mydf.Decision:
        if labels != 'Admit' and labels != 'Decline' and labels != 'Waitlist':
            mydf = mydf.drop(mydf.index[mydf['Decision']==labels])
    return mydf
def vaildateGender(mydf):        
    for genders in mydf.Gender:
        if genders != 1 and genders != 0:
            mydf = mydf.drop(mydf.index[mydf['Gender']==genders])
    return mydf

def vaildateWorkExp(mydf):
    for workexps in mydf.WorkExp:
        if workexps < 0:
            mydf = mydf.drop(mydf.index[mydf['WorkExp']==workexps])
    return mydf

def vaildateVolunteer(mydf):
    for volunteerlvl in mydf.VolunteerLevel:
        if volunteerlvl > 5 or volunteerlvl < 0:
            mydf = mydf.drop(mydf.index[mydf['VolunteerLevel']==volunteerlvl])
    return mydf
##remove roles with empty value
def removena(mydf):      
    mydf = mydf.dropna(axis = 0, how ='any')
    return mydf

## Use statistical method to remove outliers
def stats_outlier(col,df):
    ##plot the histogram
    plt.hist(df[col])
    plt.show()
    ##plot the boxplot
    plt.boxplot(df[col])
    plt.show()
    ##get standard deviation of the data
    data_std = np.std(df[col]) 
    ##get the mean of the data
    data_mean = np.mean(df[col])
    ## we decided to usw 3-standard deviation away as the boundary to classfied outliers
    anomaly_cutoff = data_std*3
    ##upperbound and lower bound 
    upperbound = data_mean + anomaly_cutoff
    lowerbound = data_mean - anomaly_cutoff
    ##identified outlier and return it 
    outlier_upper_1 = df[df[col] > upperbound]
    outlier_lower_1 = df[df[col] < lowerbound]
    df_outlier = pd.concat([outlier_upper_1, outlier_lower_1], axis=0)
    return(df_outlier)

### delete outlier
def delete(df,outlier):
    for index, row in outlier.iterrows():
        df = df.drop(index)
    return(df)

def main(argv):
    mydf = pd.read_csv('SummerStudentAdmissions2.csv',sep=',', encoding='latin1')
    mydf = capstate(mydf)
    mydf = vaildateGPA(mydf)
    mydf = vaildateGender(mydf)
    mydf = vaildateWorkExp(mydf)
    mydf = vaildateVolunteer(mydf)
    mydf = removena(mydf)
    writting_outlier = stats_outlier('WritingScore',mydf)
    test_outlier = stats_outlier('TestScore',mydf)
    exp_outlier = stats_outlier('WorkExp',mydf)
    mydf = delete(mydf,exp_outlier)
    mydf = delete(mydf,writting_outlier)
    mydf = delete(mydf,test_outlier)
    mydf.to_csv("Admissions_Cleaned.csv", index = False)
    
if __name__ == "__main__":
    main(sys.argv)
    

