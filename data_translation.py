import numpy as np
import pandas as pd
import grader as gd
import seaborn as sns
import matplotlib.pyplot as plt


def SIScaledToDf(SIScaledArr):
    columns = ['Uniformity', 'Alteration', 'Repeats','Proximity', 'Summ', 'Index' ]
    return pd.DataFrame(np.array(SIScaledArr), columns=columns)


def NscaleLabelsArray(initial, derivatives, prepNp):
    initialResult = gd.NcalcResults(initial, prepNp)
    multipliers = [1/x for x in initialResult]
    scaled = [ [y*multipliers[i] for i,y in enumerate(gd.NcalcResults(x,prepNp))] for x in np.array(derivatives)[:,0]]
    return scaled
    

def addSummIndexToScaled(scaledArr):
    return [x+[sum(x),i] for i,x in enumerate(scaledArr)]


'''matcher function takes label and returns depth'''
def drawLayout(labels, matcher, collapse, transDf): 
    depth = []
    for row in labels: 
        if collapse == 1:
            if row != '':
                depth.append(matcher(row, transDf))
            else: 
                depth.append(0)
        else:
            drow = []
            for label in row:
                if label != '':
                    drow.append(matcher(label, transDf))
                else: 
                    drow.append(0)
            depth.append(drow)
    fig, ax = plt.subplots()
    if collapse == 1:   
        ax = sns.heatmap(np.array([[x] for x in depth]),annot = np.array([[x] for x in labels]),fmt = '')
    else:
        ax = sns.heatmap(np.array(depth),annot = np.array(labels),fmt = '')


def translationMatcher(label, transDf):
    return transDf[transDf.index == label].sum().aggregate('sum')

def translationMatcherZone(label, transDf):
    lablist = [x for x in label]
    return transDf[transDf.index.isin(lablist) ].sum().aggregate('sum')

def translationMatcherZoneFull(label, transDf):
    lablist = [x for x in label]
    return transDf[transDf.index.isin(lablist) ].sum().aggregate('sum')

def translationMatcherFull(label, transDf):
    lablist = [x for x in label]
    return transDf[transDf.index.isin(lablist) ].sum().aggregate('sum')