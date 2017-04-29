import DataUtil as util

GLOBAL_DATA_PATH='C:\\MyDrive\\Transporter\\KSU Shared\\2017\\CIS 732\\Projects\\'

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

        subSetData.append([str(data[idx][cpuIndex]), str(data[idx][memIndex]), str(data[idx][projIndex])])


        
newData, keyDict = util.convertSymbolic(subSetData, symCols)

for idx, item in enumerate(subSetData):
    print(str(item))
    print(str(newData[idx]))
    print(' ')