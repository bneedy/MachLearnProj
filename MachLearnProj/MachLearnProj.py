import DataUtil as util
import time
import matplotlib.pyplot as plt
import numpy as np
import getpass
import warnings
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from multiprocessing import Process, Queue
import sys

def blakesClustering(data, clustCount, num, beocatFlag, outputQ):
    ######### Blake's Model ##############
    linkageType = 'complete'
    newData = data[:]
    blakemodel = AgglomerativeClustering(linkage=linkageType, n_clusters=clustCount)

    npDataArray = np.array(newData)

    t = time.time()
    blakemodel.fit(npDataArray)
    blakeTimeTaken = time.time() - t

    blakelabels = blakemodel.labels_
    title = "Clustering " + str(len(npDataArray)) + " data points into " + str(clustCount) + " clusters using " + linkageType + " in " + str(round(blakeTimeTaken, 3)) + " sec"
    blakelabels, avgs = sortLabels(newData, blakelabels)
    plotData(npDataArray, title, blakelabels, avgs, num, beocatFlag)
    outputQ.put(blakelabels)

def tracysClustering(data, clustCount, num, beocatFlag, outputQ):
    ######### Tracy's Model ##############
    linkageType = 'average'
    newData = data[:]
    tracymodel = AgglomerativeClustering(linkage=linkageType, n_clusters=clustCount)
    
    npDataArray = np.array(newData)

    t = time.time()
    tracymodel.fit(npDataArray)
    tracyTimeTaken = time.time() - t

    tracylabels = tracymodel.labels_
    title = "Clustering " + str(len(npDataArray)) + " data points into " + str(clustCount) + " clusters using " + linkageType + " in " + str(round(tracyTimeTaken, 3)) + " sec"
    tracylabels, avgs = sortLabels(newData, tracylabels)
    plotData(npDataArray, title, tracylabels, avgs, num, beocatFlag)
    outputQ.put(tracylabels)


def plotData(plotData, title, labels, avgLabels, figureNum, beocatFlag):
    plt.figure(num=figureNum, figsize=(9,6), dpi=150)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, len(set(labels)))
    plt.scatter(plotData[:,0], plotData[:,1], c=labels, cmap=cmap)
    plt.title(title)
    plt.xlabel('Normalized CPU Usage')
    plt.ylabel('Normalized Memory Usage')
    cbar = plt.colorbar(cmap=cmap)
    
    counts = [0] * len(set(labels))
    for item in labels:
        counts[item] += 1
    
    offsetVal = (1/len(counts))/2
    ticks = np.linspace(0.5 - offsetVal, len(counts) - offsetVal - 0.5, len(counts) + 1)
    cbar.set_ticks(ticks)
    tickLabels = []
    for i, val in enumerate(counts):
        tickLabels.append(str(val) + " (" + str(round(float(avgLabels[i]),3)) + ")")
    cbar.set_ticklabels(tickLabels)

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# in each cluster with average cpu/mem', rotation=270)

    if beocatFlag:
        plt.savefig('/homes/knedler/figures1/Figure_' + str(figureNum))
    elif getpass.getuser() == 'blake':
        plt.savefig('C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\tmp4\\Figure_' + str(figureNum))
    else:
        plt.savefig(str(figureNum)+'.png')


def sortLabels(data, labels):
    memCpuByLabel = {}
    for idx, item in enumerate(data):
        if labels[idx] not in memCpuByLabel:
            memCpuByLabel[labels[idx]] = []

        memCpuByLabel[labels[idx]].append(data[idx,0] + data[idx,1])

    labelAverages = {}
    for label in memCpuByLabel:
        labelAverages[label] = np.mean(memCpuByLabel[label])

    # Sort labels from min to max
    minToMaxList = [0] * len(labelAverages.keys())
    swapDict = {}

    for key in labelAverages.keys():
        minToMaxList[int(key)] = labelAverages[key]
    
    minToMaxList.sort()

    for key in labelAverages.keys():
        swapDict[key] = minToMaxList.index(labelAverages[key])

    newLabels = []
    for item in labels:
        newLabels.append(swapDict[item])

    newLabelAverages = {}
    for key, item in labelAverages.items():
        newLabelAverages[swapDict[key]] = item

    return newLabels, newLabelAverages

def performSupervisedLearning(X_train, X_test, y_train, y_test):
    #clf = OneVsRestClassifier(svm.SVC(probability=True))
    clf = MLPClassifier()
    
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    my_metrics = metrics.classification_report( y_test, predictions)
    print(my_metrics)

if __name__ == '__main__':
    
    beocatFlag = False
    dataPoints = 1500

    if not beocatFlag:
        if getpass.getuser() == 'blake':
            GLOBAL_DATA_PATH='C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\'
        else:
            GLOBAL_DATA_PATH='K:\\Tracy Marshall\\Transporter\\KSU Masters\\2017\\CIS 732\\Projects\\'
        filename = GLOBAL_DATA_PATH + 'acct_table_tiny.txt'
        filename = GLOBAL_DATA_PATH + 'acctg_small'
        filename = GLOBAL_DATA_PATH + 'accounting'
    
    else:
        GLOBAL_DATA_PATH = '/homes/knedler/'
        filename = GLOBAL_DATA_PATH + 'acctg'

    acctData = util.stripAcctFileHeader(filename,dataPoints)

    # Output from processes
    out_blake = Queue()
    out_tracy = Queue()
    blakeOutput = []
    tracyOutput = []

    # Data with feature selection
    subSetData = []
    subSetDataHeader = ['cpu', 'mem', 'project', 'ru_wallclock', 'io']

    # List of symbolic columns
    symCols = [2]

    # Cluster count
    clustCounts = [5] #[3, 4, 5, 6, 7, 8]
    
    # read in data
    #if not beocatFlag:
    #    data, head = util.readData(filename) # for personal...
    #else:
    data, head = util.readData(acctData, False) # for beocat...

    for idx, item in enumerate(util.column(data, head.index('failed'))):
        if int(item) == 0:
            cpuIndex = head.index('cpu')
            memIndex = head.index('mem')
            projIndex = head.index('project')
            clockIndex = head.index('ru_wallclock')
            ioIndex = head.index('io')

            subSetData.append([ \
                str(data[idx][cpuIndex]),   \
                str(data[idx][memIndex]),   \
                str(data[idx][projIndex]),  \
                str(data[idx][clockIndex]), \
                str(data[idx][ioIndex])])

    newData, keyDict = util.convertSymbolic(subSetData, symCols, True)

    blakeProcs = []
    tracyProcs = []

    for i, clustCount in enumerate(clustCounts):
        blakeProcs.append(Process(target=blakesClustering, args=(newData, clustCount, (i*2)+1, beocatFlag, out_blake, )))
        tracyProcs.append(Process(target=tracysClustering, args=(newData, clustCount, (i*2)+2, beocatFlag, out_tracy, )))
        
    print("Starting multi processor cluster...")
    procTime = time.time()

    if beocatFlag:
        for i in range(min(len(blakeProcs), len(tracyProcs))):
            blakeProcs[i].start()
            tracyProcs[i].start()

        for i in range(min(len(blakeProcs), len(tracyProcs))):
            blakeOutput.append(out_blake.get())
            blakeProcs[i].join()
            tracyOutput.append(out_tracy.get())
            tracyProcs[i].join()

    else:
        for i in range(min(len(blakeProcs), len(tracyProcs))):
            blakeProcs[i].start()
            blakeOutput.append(out_blake.get())
            blakeProcs[i].join()

            tracyProcs[i].start()
            tracyOutput.append(out_tracy.get())
            tracyProcs[i].join()

    # Dependent on time to close plots...
    endTime = time.time() - procTime
    print("Time to process was " + str(endTime) + " seconds.")

    dataWithoutCluster = []
    dataClustered = []
    answers = []

    # Build up lists with data with and without cluster information
    for i, item in enumerate(newData):
        withoutClust = []
        withClust = []
        ans = []

        for j, feat in enumerate(item):
            if j == subSetDataHeader.index('mem') or j == subSetDataHeader.index('cpu'):
                ans.append(feat)
            else:
                withoutClust.append(feat)
                withClust.append(feat)

        withClust.append(blakeOutput[0][i])

        dataWithoutCluster.append(withoutClust)
        dataClustered.append(withClust)
        answers.append(ans)

    # Put data into numpy arrays to process
    dataWithoutCluster = np.array(dataWithoutCluster)
    dataClustered = np.array(dataClustered)

    mlb = MultiLabelBinarizer()
    answers_enc = mlb.fit_transform(np.array(answers))

    warnings.simplefilter("ignore", UserWarning)

    # Perform on data without clusters
    kf = KFold(n_splits=5)
    for k, (train, test) in enumerate(kf.split(dataWithoutCluster, answers_enc)):
        #print("K:%s - %s %s" % (k, train, test))

        X_train, X_test, y_train, y_test = dataWithoutCluster[train], dataWithoutCluster[test], answers_enc[train], answers_enc[test]
        
        performSupervisedLearning(X_train, X_test, y_train, y_test)

    # Perform on data with clusters
    kf = KFold(n_splits=5)
    for k, (train, test) in enumerate(kf.split(dataClustered, answers_enc)):
        #print("K:%s - %s %s" % (k, train, test))

        X_train, X_test, y_train, y_test = dataClustered[train], dataClustered[test], answers_enc[train], answers_enc[test]

        performSupervisedLearning(X_train, X_test, y_train, y_test)