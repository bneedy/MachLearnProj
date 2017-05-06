import DataUtil as util
import time
import matplotlib.pyplot as plt
import numpy as np
import getpass
import argparse
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputClassifier
from multiprocessing import Process, Queue

def blakesClustering(data, clustCount, num, path, outputQ):
    ######### Blake's Model ##############
    linkageType = 'complete'
    newData = data[:]
    blakemodel = AgglomerativeClustering(linkage=linkageType, n_clusters=clustCount)

    npDataArray = np.array(newData)

    t = time.time()
    blakemodel.fit(npDataArray)
    blakeTimeTaken = time.time() - t

    blakelabels = blakemodel.labels_
    title = "Clustering " + str(len(npDataArray)) + " data points into " + str(clustCount) + \
        " clusters using " + linkageType + " in " + str(round(blakeTimeTaken, 3)) + " sec"
    blakelabels, avgs = sortLabels(newData, blakelabels)
    plotData(npDataArray, title, blakelabels, avgs, num, path)
    outputQ.put(blakelabels)

def tracysClustering(data, clustCount, num, path, outputQ):
    ######### Tracy's Model ##############
    linkageType = 'average'
    newData = data[:]
    tracymodel = AgglomerativeClustering(linkage=linkageType, n_clusters=clustCount)
    
    npDataArray = np.array(newData)

    t = time.time()
    tracymodel.fit(npDataArray)
    tracyTimeTaken = time.time() - t

    tracylabels = tracymodel.labels_
    title = "Clustering " + str(len(npDataArray)) + " data points into " + str(clustCount) + \
        " clusters using " + linkageType + " in " + str(round(tracyTimeTaken, 3)) + " sec"
    tracylabels, avgs = sortLabels(newData, tracylabels)
    plotData(npDataArray, title, tracylabels, avgs, num, path)
    outputQ.put(tracylabels)


def plotData(plotData, title, labels, avgLabels, figureNum, path):
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

    plt.savefig(path + 'Figure_' + str(figureNum))

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

def performSupervisedLearning(type, data, answers, memLoc, cpuLoc, split=5):
    kf = KFold(n_splits=split)
    for k, (train, test) in enumerate(kf.split(data, answers)):
        #print("K:%s - %s %s" % (k, train, test))

        X_train, X_test, y_train, y_test = data[train], data[test], answers[train], answers[test]

        clf = MultiOutputClassifier(SVR(cache_size=1000))
    
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
    
        r2_metrics = metrics.r2_score(y_test, predictions, multioutput='uniform_average')
        meanSquareError = metrics.mean_squared_error(y_test, predictions, multioutput='uniform_average')
        explainedVariance = metrics.explained_variance_score(y_test, predictions, multioutput='uniform_average')

        print('-------------')
        print('Metrics for:                ' + str(type))
        print('R2 Metrics:                 ' + str(r2_metrics))
        print('Mean Squared Error Metrics: ' + str(meanSquareError))
        print('Explained Variance Metrics: ' + str(explainedVariance))
        print('-------------')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beocat', type=bool, default=False, nargs=1, help='flag to run multiprocessing more intensively')
    parser.add_argument('-dp', '--dataPoints', type=int, default=1800, nargs=1, help='number of datapoints to use')
    parser.add_argument('-p', '--dataPath', type=str, default='', nargs=1, help='path to the data')
    parser.add_argument('-fp', '--figurePath', type=str, default='', nargs=1, help='path to save the figures')
    args = parser.parse_args()

    beocatFlag = args.beocat
    dataPoints = args.dataPoints
    figurePath = args.figurePath
    GLOBAL_DATA_PATH = args.dataPath

    if not beocatFlag:
        if getpass.getuser() == 'blake':
            GLOBAL_DATA_PATH='C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\'
            figurePath = GLOBAL_DATA_PATH + 'FIGURES\\Blake2\\'
        else:
            GLOBAL_DATA_PATH='K:\\Tracy Marshall\\Transporter\\KSU Masters\\2017\\CIS 732\\Projects\\'
            figurePath = GLOBAL_DATA_PATH + 'FIGURES\\Tracy\\'
        
        filename = GLOBAL_DATA_PATH + 'accounting'
    
    else:
        GLOBAL_DATA_PATH = '/homes/knedler/'
        figurePath = GLOBAL_DATA_PATH + 'figures5/'
        filename = GLOBAL_DATA_PATH + 'acctg'

    if figurePath == '':
        figurePath = GLOBAL_DATA_PATH

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
    clustCounts = [5, 6] #[3, 4, 5, 6, 7, 8]

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

    del subSetData

    blakeProcs = []
    tracyProcs = []

    for i, clustCount in enumerate(clustCounts):
        blakeProcs.append(Process(target=blakesClustering, args=(newData, clustCount, (i*2)+1, figurePath, out_blake, )))
        tracyProcs.append(Process(target=tracysClustering, args=(newData, clustCount, (i*2)+2, figurePath, out_tracy, )))
        
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
    dataClustered_blake = []
    dataClustered_tracy = []
    answers = []
    memLoc = 0
    cpuLoc = 0

    for curr, clusterCount in enumerate(clustCounts):
        dataClustered_blake.append([])
        dataClustered_tracy.append([])
        # Build up lists with data with and without cluster information
        for i, item in enumerate(newData):

            withClust = []

            if curr == 0:
                withoutClust = []
                ans = []

            for j, feat in enumerate(item):
                if curr == 0 and (j == subSetDataHeader.index('mem') or j == subSetDataHeader.index('cpu')):
                    if memLoc == 0 and cpuLoc == 0:
                        if j == subSetDataHeader.index('mem'):
                            cpuLoc = 1
                        else:
                            memLoc = 1
                    ans.append(feat)
                else:
                    if curr == 0:
                        withoutClust.append(feat)
                    withClust.append(feat)
                    
            if curr == 0:
                dataWithoutCluster.append(withoutClust)
                answers.append(ans)

            dataClustered_blake[curr].append(withClust)
            dataClustered_blake[curr][len(dataClustered_blake[curr]) - 1].append(blakeOutput[curr][i])

            dataClustered_tracy[curr].append(withClust)
            dataClustered_tracy[curr][len(dataClustered_tracy[curr]) - 1].append(tracyOutput[curr][i])


    del newData

    # Create list of processes to run supervised learning...
    performSupervisedLearning("Non-Clustered.", np.array(dataWithoutCluster), np.array(answers), memLoc, cpuLoc)
    for i, item in enumerate(clustCounts):
        performSupervisedLearning("Clustered with Blakes and " + str(item) +" clusters.", \
            np.array(dataClustered_blake[i]), np.array(answers), \
            memLoc, cpuLoc)
        performSupervisedLearning("Clustered with Tracys and " + str(item) +" clusters.", \
            np.array(dataClustered_tracy[i]), np.array(answers), \
            memLoc, cpuLoc)