import csv
from sklearn.preprocessing import normalize

def column(matrix, i):
    return [row[i] for row in matrix]

def normalizeData(origData, normalizeFlag=True):
    if normalizeFlag:
        return normalize(origData[:], axis=1, norm='l1')
    else:
        return origData

def convertSymbolic(data, symCol, normalizeFlag=False):

    # New Holder of Data
    tmpData = []

    # Dict of Enumerations
    enumerations = {}

    for rowNum, row in enumerate(data):
        newRow = []

        for colNum, item in enumerate(row):
            # Check if we have a list of known symbolic columns
            if colNum in symCol:

                if colNum in enumerations:
                    if item in enumerations[colNum]:
                        newItem = enumerations[colNum].index(item)
                    else:
                        newItem = len(enumerations[colNum])
                        enumerations[colNum].append(item)
                else:
                    # Create list of enumerations per column
                    enumerations[colNum] = [item]
                    newItem = 0

                newRow.append(float(newItem))

            else:
                newRow.append(float(item))

        tmpData.append(newRow)

    return normalizeData(tmpData,normalizeFlag), enumerations

def stripAcctFileHeader(filename):
    header = "qname:hostname:group:owner:job_name:job_number:account:priority:submission_time:\
    start_time:end_time:failed:exit_status:ru_wallclock:ru_utime:ru_stime:ru_maxrss:ru_ixrss:ru_ismrss:\
    ru_idrss:ru_isrss:ru_minflt:ru_majflt:ru_nswap:ru_inblock:ru_oublock:ru_msgsnd:ru_msgrcv:ru_nsignals:\
    ru_nvcsw:ru_nivcsw:project:department:granted_pe:slots:task_number:cpu:mem:io:category:iow:pe_taskid:\
    maxvmem:arid:ar_submission_time\n"

    with open(filename) as fin:
        output = fin.readlines()

    retVal = []
    retVal.append(header.replace(":",": "))
    for item in output:
        retVal.append(item.replace(":",": "))

    return retVal

def readData(dataStream, isFile=True, dataInDict=False):
    header = []
    data = []
    dataDict = {}
    
    if isFile:
        with open(dataStream) as fin:
            csvData = csv.reader(fin, delimiter=':')
            headerFlag = True
            for rowNum, row in enumerate(csvData):
                if headerFlag:
                    for item in list(row):
                        header.append(str(item).strip())
                        if dataInDict:
                            dataDict[str(item).strip()] = []
                    headerFlag = False
                else:
                    if dataInDict:
                        for i, item in enumerate(list(row)):
                            dataDict[headerList[i]].append(item)
                    else:
                        data.append(row)
    else:
        csvData = csv.reader(dataStream, delimiter=':')
        headerFlag = True
        for rowNum, row in enumerate(csvData):
            if headerFlag:
                for item in list(row):
                    header.append(str(item).strip())
                    if dataInDict:
                        dataDict[str(item).strip()] = []
                headerFlag = False
            else:
                if dataInDict:
                    for i, item in enumerate(list(row)):
                        dataDict[headerList[i]].append(item)
                else:
                    data.append(row)

    if dataInDict:
        return dataDict, header
    else:
        return data, header