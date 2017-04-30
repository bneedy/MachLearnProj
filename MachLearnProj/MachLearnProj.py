import DataUtil as util
import time
import matplotlib.pyplot as plt
import numpy as np
import getpass
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Process

def blakesClustering(data):
    ######### Blake's Model ##############
    linkageType = 'complete'
    newData = data[:]
    blakemodel = AgglomerativeClustering(linkage=linkageType, n_clusters=3)

    npDataArray = np.array(newData)

    t = time.time()
    blakemodel.fit(npDataArray)
    blakeTimeTaken = time.time() - t

    blakelabels = blakemodel.labels_
    title = "Clustering " + str(len(npDataArray)) + " data points using " + linkageType + " in " + str(round(blakeTimeTaken, 3)) + " sec"
    blakelabels, avgs = sortLabels(newData, blakelabels, 2)
    plotData(npDataArray, title, blakelabels, avgs, 1)

def tracysClustering(data):
    ######### Tracy's Model ##############
    linkageType = 'average'
    newData = data[:]
    tracymodel = AgglomerativeClustering(linkage=linkageType, n_clusters=3)
    
    npDataArray = np.array(newData)

    t = time.time()
    tracymodel.fit(npDataArray)
    tracyTimeTaken = time.time() - t

    tracylabels = tracymodel.labels_
    title = "Clustering " + str(len(npDataArray)) + " data points using " + linkageType + " in " + str(round(tracyTimeTaken, 3)) + " sec"
    tracylabels, avgs = sortLabels(newData, tracylabels, 2)
    plotData(npDataArray, title, tracylabels, avgs, 2)


def plotData(plotData, title, labels, avgLabels, figureNum):
    plt.figure(num=figureNum, figsize=(9,6), dpi=150)
    plt.scatter(plotData[:,0], plotData[:,1], c=labels, cmap='Accent')
    plt.title(title)
    plt.xlabel('Normalized CPU Usage')
    plt.ylabel('Normalized Memory Usage')
    cbar = plt.colorbar()
    
    counts = [0] * len(set(labels))
    for item in labels:
        counts[item] += 1
    
    cbar.set_ticks(np.linspace(0.5, len(counts) - 0.5, len(counts) + 1))
    tickLabels = []
    for i, val in enumerate(counts):
        tickLabels.append(str(val) + " (" + str(round(float(avgLabels[i]),3)) + ")")
    cbar.set_ticklabels(tickLabels)

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# in each cluster with average cpu/mem', rotation=270)

    plt.savefig(str(figureNum)+'.png')

def sortLabels(data, labels, num):
    memCpuByLabel = {}
    for idx, item in enumerate(data):
        if labels[idx] not in memCpuByLabel:
            memCpuByLabel[labels[idx]] = []

        memCpuByLabel[labels[idx]].append(data[idx,0] + data[idx,1])

    labelAverages = {}
    for label in memCpuByLabel:
        labelAverages[label] = np.mean(memCpuByLabel[label])

    #print('CPU/Mem Averages by label for figure' + str(num) + ': ')
    #print(labelAverages)

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

    filename = GLOBAL_DATA_PATH + 'acct_table_small.txt'

    # Data with feature selection
    subSetData = []

    # List of symbolic columns
    symCols = [2]

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

    newData, keyDict = util.convertSymbolic(subSetData, symCols, True)

    print("Starting multi processor cluster...")
    procTime = time.time()

    blakeProc = Process(target=blakesClustering, args=(newData,))
    tracyProc = Process(target=tracysClustering, args=(newData,))

    blakeProc.start()
    tracyProc.start()

    blakeProc.join()
    tracyProc.join()

    # Dependent on time to close plots...
    endTime = time.time() - procTime
    print("Time to process was " + str(endTime) + " seconds.")