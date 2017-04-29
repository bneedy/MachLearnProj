class DataUtil(object):
    """Utilities for normalizing data and converting symbolic data"""

    #def __init__(self):
    #    pass

    def column(matrix, i):
        return [row[i] for row in matrix]

    def normalizeData(self, origData, normalizeFlag=True):
        if normalizeFlag:
            return normalize(origData, axis=1, norm='l1')
        else:
            return origData

    def convertSymbolic(self, data, normalizeFlag=False):
        tmpData = data
        enumeratedDict = {}
        columns = len(tmpData[0])

        # Loop through all columns
        for col in range(columns):
            enumeratedDict[col] = {}
            currColumn = column(tmpData, col)

            # Determine if they are all floats
            if not all(item.isdigit() for item in currColumn):

                for row, item in enumerate(currColumn):
                    itemVal = 0

                    if item in enumeratedDict[col]:
                        itemVal = int(enumeratedDict[col][item])
                    else:
                        itemVal = len(enumeratedDict[col].keys())
                        enumeratedDict[col][item] = itemVal

                    tmpData[row][col] = itemVal

        return self.normalizeData(self,tmpData,normalizeFlag)