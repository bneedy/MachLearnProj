import DataUtil as util
import time
import matplotlib.pyplot as plt
import numpy as np
import getpass
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Process
from sklearn.model_selection import KFold


def supervisedLearning(data, xStart, xStop, yStart, yStop):
    
    print data

    Ndata = np.array(data)

    kf = KFold(n_splits=5)

    for trainIndex, testIndex in kf.split(data):
        #print("Train x: %s Train y: %s Test x: %s Test y: %s" % (train[xStart:xStop][:], train[:][yStart:yStop], test[:][xStart:xStop], test[:][yStart:yStop]))
        #print("Train x: %s Train y: %s Test x: %s Test y: %s" % (train, train, test, test))
        #xTrain, xTest = data[trainIndex][xStart:xStop], data[testIndex][xStart:xStop]
        #yTrain, yTest = data[trainIndex][yStart:yStop], data[testIndex][yStart:yStop]

        ntri = np.array(trainIndex)
        ntei = np.array(testIndex)

        xTrain = Ndata[ntri]

        print('X Train')
        print xTrain
        #print('X Test')
        #print xTest
        #print('Y Train')
        #print yTrain
        #print('Y Test')
        #print yTest



def blakesClustering(data, clustCount, num):
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
    plotData(npDataArray, title, blakelabels, avgs, num)

def tracysClustering(data, clustCount, num):
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
    plotData(npDataArray, title, tracylabels, avgs, num)


def plotData(plotData, title, labels, avgLabels, figureNum):
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

if __name__ == '__main__':
    if getpass.getuser() == 'blake':
        GLOBAL_DATA_PATH='C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\'
    else:
        GLOBAL_DATA_PATH='K:\\Tracy Marshall\\Transporter\\KSU Masters\\2017\\CIS 732\\Projects\\'

    filename = GLOBAL_DATA_PATH + 'acct_table_tiny.txt'

    # Data with feature selection
    subSetData = []

    # List of symbolic columns
    symCols = [2]

    # Cluster count
    clustCounts = [3, 4, 5, 6, 7, 8]

    # Target columsn for supervised learning
    xStart = 2
    xStop = 4
    yStart = 0
    yStop = 1

    # read in data
    data, head = util.readData(filename)

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

    newData, keyDict = util.convertSymbolic(subSetData, symCols, False)

    supervisedLearning(newData,xStart, xStop, yStart, yStop)

    newData, keyDict = util.convertSymbolic(subSetData, symCols, True)

    blakeProcs = []
    tracyProcs = []

    for i, clustCount in enumerate(clustCounts):
        blakeProcs.append(Process(target=blakesClustering, args=(newData,clustCount, (i*2)+1)))
        tracyProcs.append(Process(target=tracysClustering, args=(newData,clustCount, (i*2)+2)))
        
    print("Starting multi processor cluster...")
    procTime = time.time()
    for i in range(min(len(blakeProcs), len(tracyProcs))):
        blakeProcs[i].start()
        tracyProcs[i].start()

    for i in range(min(len(blakeProcs), len(tracyProcs))):
        blakeProcs[i].join()
        tracyProcs[i].join()

    # Dependent on time to close plots...
    endTime = time.time() - procTime
    print("Time to process was " + str(endTime) + " seconds.")