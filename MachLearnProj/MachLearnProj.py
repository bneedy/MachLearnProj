import DataUtil as util
import time
import matplotlib.pyplot as plt
import numpy as np
import getpass
from sklearn.cluster import AgglomerativeClustering
  

user = getpass.getuser()

if user == 'blake':
    GLOBAL_DATA_PATH='C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\'
else:
    GLOBAL_DATA_PATH='K:\\Tracy Marshall\\Transporter\\KSU Masters\\2017\\CIS 732\\Projects\\'

filename = GLOBAL_DATA_PATH + 'acct_table_tiny.txt'

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


######### Blake's Model ##############
blakemodel = AgglomerativeClustering(linkage='complete', n_clusters=5)

npDataArray = np.array(newData)

t = time.time()
blakedta = blakemodel.fit(npDataArray)
blakeTimeTaken = time.time() - t

blakelabels = blakemodel.labels_

#print(str(blakelabels))

plt.figure(1)
plt.scatter(npDataArray[:,0], npDataArray[:,1], c=blakelabels, cmap='Accent')
plt.title("Ward clustering in " + str(blakeTimeTaken) + " s")
plt.xlabel('CPU')
plt.ylabel('Memory')
plt.colorbar()

memCpuByLabel = {}
for idx, item in enumerate(newData):
    if blakelabels[idx] not in memCpuByLabel:
        memCpuByLabel[blakelabels[idx]] = []

    memCpuByLabel[blakelabels[idx]].append(newData[idx,0] + newData[idx,1])

labelAverages = {}
for label in memCpuByLabel:
    labelAverages[label] = np.mean(memCpuByLabel[label])

print('CPU/Mem Averages by label for Blake Model: ')
print(labelAverages)


######### Tracy's Model ##############

tracymodel = AgglomerativeClustering(linkage='average', n_clusters=5)

t = time.time()
tracydta = tracymodel.fit(npDataArray)
tracyTimeTaken = time.time() - t

tracylabels = tracymodel.labels_

#print(str(tracylabels))

plt.figure(2)
plt.scatter(npDataArray[:,0], npDataArray[:,1], c=tracylabels, cmap='Accent')
plt.title("Average clustering in " + str(tracyTimeTaken) + " s")
plt.xlabel('CPU')
plt.ylabel('Memory')
plt.colorbar()


memCpuByLabel = {}
for idx, item in enumerate(newData):
    if tracylabels[idx] not in memCpuByLabel:
        memCpuByLabel[tracylabels[idx]] = []

    memCpuByLabel[tracylabels[idx]].append(newData[idx,0] + newData[idx,1])

labelAverages = {}
for label in memCpuByLabel:
    labelAverages[label] = np.mean(memCpuByLabel[label])

print('CPU/Mem Averages by label for Tracy Model: ')
print(labelAverages)







plt.show()