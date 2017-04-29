import csv
import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import normalize as norm
from sklearn.cluster import KMeans



GLOBAL_DATA_PATH='C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\'

filename = GLOBAL_DATA_PATH + 'acct_table2.txt'

data = {}
headerList = []
dataArray = []

# read in data
with open(filename) as fin:
    csvData = csv.reader(fin, delimiter=':')
    headerFlag = True
    for row in csvData:
        if headerFlag:
            for item in list(row):
                data[str(item).strip()] = []
                headerList.append(str(item).strip())
            headerFlag = False
        else:
            for i, item in enumerate(list(row)):
                data[headerList[i]].append(item)

for idx, item in enumerate(data['failed']):
    if int(item) == 0:
        dataArray.append([float(data['cpu'][idx]), float(data['mem'][idx])])