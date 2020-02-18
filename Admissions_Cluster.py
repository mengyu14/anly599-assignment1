# -*- coding: utf-8 -*-
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage


#### K-MEAN clustering Looking at the academic

def kmean_academic(mydf,k):
    ### select area about academic 
    df_numerical=pd.concat([mydf['WritingScore'], mydf['GPA'],mydf['TestScore']], 
                            axis=1, keys=['WritingScore', 'GPA','TestScore'])
    x = df_numerical.values
    ### normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    ### use k group
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    pca2D = decomposition.PCA(2)
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("2-dimensional scatter plot using PCA with theater data (k-mean)")
    plt.show()
    ### get the quality score using silhouette and cliniski_harabasz
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    ### update the data frame
    mydf['academic_label_kmean'] = cluster_labels




### Ward (hierachy cluster)


def ward_theater(k,mydf):
    df_numerical=pd.concat([mydf['WritingScore'], mydf['GPA'],mydf['TestScore']], 
                            axis=1, keys=['WritingScore', 'GPA','TestScore'])   
    ##get the normalizaed dataframe
    x = df_numerical.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled) 
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(normalizedDataFrame)
    ##print(cluster_labels)
    pca2D = decomposition.PCA(2)
    pca2D = pca2D.fit(normalizedDataFrame)
    plot_columns = pca2D.transform(normalizedDataFrame)
    ## plot dendogram
    linked = linkage(plot_columns, 'single')

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=cluster_labels,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()
    ##get the quality score using silhouette and cliniski_harabasz
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    mydf['academic_label_ward'] = cluster_labels

def main():
    mydf = pd.read_csv('Admissions_Cleaned.csv',sep=',', encoding='latin1')
    ### Since k =3 have the best score we will choose k =3.
    kmean_academic(mydf,3)
    ward_theater(3,mydf)
    mydf.to_csv("Admissions_Cleaned_clustered.csv", index = False)
    
main()

'''
corr, _ = pearsonr(mydf['GPA'], mydf['TestScore'])
corr2, _= pearsonr(mydf['WritingScore'], mydf['TestScore'])
corr3, _= pearsonr(mydf['WritingScore'], mydf['GPA'])
'''




