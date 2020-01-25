import numpy as np
import pandas as pd
import port.py as prt
import copy
import muliprocessing as mp

def NpSummIndex(index, prepNp):
    ivalues = IntersectionNp(prepNp[0], index)
    arr = prepNp[1]
    agg = 0
    for i, ind in enumerate(arr):
        if i in ivalues:
            for c, col in enumerate(ind):
                agg += col

    return agg


def IntersectionNp(main, sub):
    return np.array([np.where(main == x) for x in sub]).flatten()


def NpSummIndexColumn(index, column, prepNp):
    ivalues = IntersectionNp(prepNp[0], index)
    cvalues = IntersectionNp(prepNp[2], column)
    arr = prepNp[1]
    agg = 0
    for i, ind in enumerate(arr):
        if i in ivalues:
            for c, col in enumerate(ind):
                if c in cvalues:
                    agg += col

    return agg


class Grades:
    """returns summ of positive differences between each zone and average
    divided by total cases"""
    @staticmethod
    def NcalcUniformity(zones, prepNp):
        def prepUniform():  # returns array of cases by each zone
            calculatedZones = []
            for zone in zones:
                lablist = [x for x in zone]
                calculatedZones.append(NpSummIndex(lablist, prepNp))
            return calculatedZones
        cZones = prepUniform()
        avgLoad = sum(cZones) / len(cZones)
        positiveDelta = [x - avgLoad for x in cZones if x > avgLoad]
        return sum(positiveDelta)/np.sum(prepNp[1])

    # returns percentage of cases that are not alterated between hands
    @staticmethod
    def NcalcAlteration(parts, prepNp):
        case = 0
        for i, part in enumerate(parts):
            lablist = [x for x in part]
            labOplist = [x for x in parts[i - len(parts)]]
            case += NpSummIndexColumn(lablist, labOplist, prepNp)
        return 1 - case/np.sum(prepNp[1])

    @staticmethod
    def NcalcProximity(labels, prepNp):
        cases = NpSummIndex(np.append(labels[1][0:4], labels[1][6:-2]), prepNp)
        return 1 - (cases/np.sum(prepNp[1]))

    # returns percentage of cases that lead to repetition of zone
    @staticmethod
    def NcalcRepeats(zones, prepNp): 
        depth = []
        for zone in zones:
            lablist = [x for x in zone]
            depth.append(NpSummIndexColumn(lablist, lablist, prepNp))
        return sum(depth)/np.sum(prepNp[1])


def buffSorted(x):
    return x[1]


def NiterateAlgo(labels_, prepNp, quanity):
    innerLabels = [(labels_,4)]
    historyLabels = []
    with mp.Pool(processes=4) as pool:
        for i in range(0,quanity):
            buffLabels = []
            for k in range(0,len(innerLabels)):
                label = innerLabels[k][0]
                computed = pool.apply(NcomPute(label, prepNp, labels_))
                buffLabels = buffLabels + computed
            historyLabels = historyLabels+innerLabels
            sortLb= sorted(buffLabels, key=buffSorted)
            innerLabels = sortLb[:10]
    historyLabels += innerLabels
    return historyLabels 


def NcomPute(label, prepNp, labels_):

    def modifyLabel(cc,labels):
        x,y,z,c,summ = cc[0], cc[1], cc[2],cc[3], cc[4]
        newLabels = copy.deepcopy(labels)
        newLabels[x][y], newLabels[z][c] = newLabels[z][c], newLabels[x][y]
        return (newLabels, summ)

    cols = [('Uniformity',float), ('Alteration',float), ('Repeats',float),('Proximity',float), ('Summ',float), ('x',int), ('y',int), ('z',int), ('c',np.int), ('from',np.unicode_,1), ('to',np.unicode_,1)]
    result = NfullIteration(label, prepNp, labels_)
    tupylize = []
    for res in result:
        tupylize.append(tuple(res))
    sortedNp = np.array(tupylize, dtype=cols)
    sortedNp.sort(order='Summ')

    def getCoordinates(self):
        return sortedNp[:5][['x','y','z','c', 'Summ']]
    cordList = getCoordinates()

    def transformCoord():
        newLb = []
        for cc in cordList:
            newLb.append(modifyLabel(cc, label))
        return newLb

    buffLabels = transformCoord()
    return buffLabels


def exchange(array, x,y,z,c):
    newArr = copy.deepcopy(array)
    newArr[x][y], newArr[z][c] = newArr[z][c], newArr[x][y]
    return newArr


def NfullIteration(labels_, prepNp, labelsInit):
    results = []
    initResult = NcalcResults(labelsInit,prepNp)
    multipliers = [1/x for x in initResult]
    sumInitResult = sum(initResult)
    for x, row in enumerate(labels_):
        for y, label in enumerate(row):
            for z, rowSec in enumerate(labels_):
                for c, labelSec in enumerate(rowSec):
                    if x != z and y != c and x >= z:

                        newLabels = exchange(labels_, x,y,z,c)
                        newResult = NcalcResults(newLabels, prepNp)
                        weightedResult = [x*multipliers[i] for i,x in enumerate(newResult)]
                        sumNewResult =  sum(weightedResult)#newResult)
                        if sumNewResult < len(initResult): #sumInitResult:
                            weightedResult += [sumNewResult,x,y,z,c,label,labelSec]
                            results.append(weightedResult)
    return results


def NcalcResults(labels, prepNp):
    zones = prt.convoluteZone(labels)
    parts = prt.convoluteParts(labels)
    result = []
    result.append(Grades.NcalcUniformity(zones,prepNp))
    result.append(Grades.NcalcAlteration(parts,prepNp))
    result.append(Grades.NcalcRepeats(zones,prepNp))
    result.append(Grades.NcalcProximity(labels,prepNp))
    return result
