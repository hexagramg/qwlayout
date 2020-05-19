import numpy as np
import pandas as pd
import port as prt
import copy
import multiprocessing as mp
from functools import partial
import random
import  statistics

REACH_WEIGHTS = [[16, 1,   0,   1,   8,  8,  1,   0,   1,  16, 16, 16],
                 [4,  0,   0,   0,   0,  0,  0,   0,   0,   4,  8,  16],
                 [8,  2,   4,   2,   8,  8,  2,   4,   2,   8,  16, 16]]

PINKY = 4

SORT_END = 6
RAND_MAX = 6


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

# aggregates values 
def NpSummIndexColumn(index, column, prepNp):
    """aggregates values that according to indexes and columns of 
    transition matrix """
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


def _debug_uniformity(labels, prepNp):
    zones = prt.convoluteZone(labels)
    cZones = Grades.prepUniform(zones, prepNp)
    avgLoad = statistics.median(cZones)
    positiveDelta = [pow(x - avgLoad, 2) for x in cZones if x > avgLoad]
    return cZones, avgLoad, positiveDelta


class Grades:
    @staticmethod
    def prepUniform(zones, prepNp):
        """ returns array of cases by each zone"""
        calculatedZones = []
        for i, zone in enumerate(zones):
            lablist = [x for x in zone]
            summ = NpSummIndex(lablist, prepNp)
            if i == 0 or i == len(zones) - 1:
                # pinky finger is marginally weaker
                # than other fingers, this weight is to compensate that
                summ = summ * PINKY
            calculatedZones.append(summ)
        return calculatedZones

    @staticmethod
    def NcalcUniformity(zones, prepNp):
        """returns summ of positive differences between each zone and average
        divided by total cases"""

        cZones = Grades.prepUniform(zones, prepNp)
        #avgLoad = sum(cZones) / len(cZones)
        avgLoad = statistics.median(cZones)
        positiveDelta = [x - avgLoad for x in cZones if x > avgLoad]
        return sum(positiveDelta)/np.sum(prepNp[1])


    @staticmethod
    def NcalcUnreachability(labels, prepNp):
        """grade that punishes model for far reachable keys
        """
        entries = [[NpSummIndex(char, prepNp) for char in row] for row in labels]
        cases = np.sum(np.multiply(REACH_WEIGHTS, entries))
        return cases/np.sum(prepNp[1])

    @staticmethod
    def NcalcAlteration(parts, prepNp):
        """returns percentage of cases that are not alterated between hands
        """
        case = 0
        for i, part in enumerate(parts):
            lablist = [x for x in part]
            # probably wont need it
            # labOplist = [x for x in parts[i - len(parts)]]
            case += NpSummIndexColumn(lablist, lablist, prepNp)
        return case/np.sum(prepNp[1])

    @staticmethod
    def NcalcProximity(labels, prepNp):
        cases = NpSummIndex(np.append(labels[1][0:5], labels[1][5:-2]), prepNp)
        return 1 - (cases/np.sum(prepNp[1]))

    @staticmethod
    def NcalcVerticalAlteration(labels, parts, prepNp):
        """ returns percentage of cases that are vertical alteration
        inside the part"""

        def getRowForPart(index, parts_, labels_):
            """ 
            returns array of arrays of chars that
            are intersection of label row and parts
            Args:
                index (int): variable is row_index in labels.
            """
            return [[label for label in labels_[index] if label in x] for x in parts_]
        case = 0
        topParts = getRowForPart(0, parts, labels) 
        bottomParts = getRowForPart(2, parts, labels)

        for i, part in enumerate(topParts):
            case += NpSummIndexColumn(part, bottomParts[i], prepNp)
            case += NpSummIndexColumn(bottomParts[i], part, prepNp)

        return case/np.sum(prepNp[1])

    @staticmethod
    def NcalcRepeats(zones, prepNp): 
        """ returns percentage of cases that lead to repetition of zone"""

        depth = []
        for zone in zones:
            lablist = [x for x in zone]
            depth.append(NpSummIndexColumn(lablist, lablist, prepNp))
        return sum(depth)/np.sum(prepNp[1])


def buffSorted(x):
    return x[1]


def NiterateAlgo(labels_, prepNp, quanity, weights):
    innerLabels = [(labels_,sum(weights))]
    historyLabels = []
    with mp.Pool(processes=15) as pool:
        for i in range(0,quanity):
            func = partial(NcomPute, prepNp=prepNp, labels_=labels_, weights=weights)
            historyLabels = historyLabels+innerLabels
            computed = pool.map(func, innerLabels)
            bRshape = np.array(computed)
            reshapen = bRshape.reshape(-1, *bRshape.shape[2:])
            sortLb = sorted(reshapen.tolist(), key=buffSorted)
            sortEnd = SORT_END
            randMax = RAND_MAX
            if len(sortLb) + sortEnd < randMax: #>
                randEnd = 0
            else:
                randEnd = randMax
            addRand = [random.randint(sortEnd, len(sortLb) - 1) for i in range(0, randEnd)]
            innerLabels = [sortLb[i] for i in addRand]
            innerLabels += sortLb[:sortEnd][::-1]
    historyLabels += innerLabels
    return historyLabels 


def NcomPute(innerLabels, prepNp, labels_, weights):
    label = innerLabels[0] 
    def modifyLabel(cc,labels):
        x,y,z,c,summ = cc[0], cc[1], cc[2],cc[3], cc[4]
        newLabels = copy.deepcopy(labels)
        newLabels[x][y], newLabels[z][c] = newLabels[z][c], newLabels[x][y]
        return (newLabels, summ)

    cols = [('Uniformity',float), ('Alteration',float), ('Repeats',float),('Proximity',float),
        ('VerticalAlt', float), 
        ('Summ',float), ('x',int), ('y',int), ('z',int), ('c',np.int), ('from',np.unicode_,1), 
        ('to',np.unicode_,1)]
    results = NfullIteration(label, prepNp, labels_, weights)
    #tupleResults = []
    #for res in results:
    #    tupleResults.append(tuple(res))
    tupleResults = [tuple(res) for res in results]
    sortedNp = np.array(tupleResults, dtype=cols)
    sortedNp.sort(order='Summ')

    def getCoordinates():
        return sortedNp[:9][['x','y','z','c', 'Summ']]

    cordList = getCoordinates()

    def transformCoord():
        newLb = []
        for cc in cordList:
            newLb.append(modifyLabel(cc, label))
        return newLb

    buffLabels = transformCoord()
    return buffLabels


def exchange(array, x, y, z, c):
    newArr = copy.deepcopy(array)
    newArr[x][y], newArr[z][c] = newArr[z][c], newArr[x][y]
    return newArr


def NfullIteration(labels_, prepNp, labelsInit, weights):
    results = []
    initResult = NcalcResults(labelsInit, prepNp)
    multipliers = [i/x for i, x in zip(weights, initResult)]
    sumInitResult = sum(weights)
    for x, row in enumerate(labels_):
        for y, label in enumerate(row):
            for z, rowSec in enumerate(labels_):
                for c, labelSec in enumerate(rowSec):
                    if x != z and y != c and x >= z:

                        newLabels = exchange(labels_, x, y, z, c)
                        newResult = NcalcResults(newLabels, prepNp)
                        weightedResult = [x*multiplier for x, multiplier in zip(newResult, multipliers)]
                        sumNewResult =  sum(weightedResult)#newResult)
                        if sumNewResult < sumInitResult: #if newResult is less than summOfWeights of initial label:
                            weightedResult += [sumNewResult,x,y,z,c,label,labelSec]
                            results.append(weightedResult)
    return results


def NcalcResults(labels, prepNp):
    zones = prt.convoluteZone(labels)
    parts = prt.convoluteParts(labels)
    result = (Grades.NcalcUniformity(zones, prepNp),
              Grades.NcalcAlteration(parts, prepNp),
              Grades.NcalcRepeats(zones, prepNp),
              Grades.NcalcUnreachability(labels, prepNp),
              Grades.NcalcVerticalAlteration(labels, parts, prepNp))
    return result
