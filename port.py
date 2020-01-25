import pandas as pd
import numpy as np
import re
from html.parser import HTMLParser
import matplotlib.pyplot as plt
import matplotlib as mtp


class DataParser(HTMLParser):
    def handle_data(self, data):

        splitted = data.split(':')
        filtered =  re.sub("[^\u0430-\u044F ]", "", ''.join(splitted[1:]).lower())
        self.phrases = filtered


"""Класс для всех операций с данными"""
class DataHandler:

    def __init__(self, path):
        self.phrases = []
        self._offset = 1072
        self._delta = ord('я') - ord('а') + 1
        self.loadData(path)
        self.saveCalculatedEntries()
        self.saveTransitions()
        self.saveDfToNp()
    """Загрузка данных"""
    def loadData(self, path):
        self.phrases = []
        data = pd.read_csv(path, sep='\t', header=0)
        parser = DataParser()
        for episode in data["dialogue"].to_numpy().tolist():
            parser.feed(episode)
            self.phrases.append(parser.phrases)
    """Добавить в словарь ключ или инкрементировать"""
    def incOrAdd(self, key, dictionary):
        if key not in dictionary:
            dictionary[key] = 1
        else:
            dictionary[key] += 1
    """Посчитать количство вхождений для каждого символа"""
    def saveCalculatedEntries(self):
        firstDimenstion = dict()
        for phrase in self.phrases:
            for symb in phrase:
                if symb != ' ':
                    self.incOrAdd(symb, firstDimenstion)
        self.entries = firstDimenstion
    """Генерация разреженной матрицы переходов нужной глубины(шага)"""
    def _generateSparse(self, depth):
        sparseDimension = dict()
        for phrase in self.phrases:
            words = phrase.split(' ')
            for word in words:
                if len(word) > depth:
                    length = len(word)
                    for i, letter in enumerate(word):
                        if i+depth < length:
                            self.incOrAdd((letter, word[i+depth]), sparseDimension)
        return sparseDimension
    """Сохранить матрицу переходов в объекте"""
    def saveTransitions(self):
        if len(self.phrases)>0 :
            self.transitions = self._generateSparse(1)
    """Генерация лейблов и матрицы переходов с сортировкой по символам"""
    def generateArrayLabels(self):
        sparseArray = np.zeros((self._delta, self._delta))
        for key in self.transitions:
            sparseArray[ord(key[0])-self._offset][ord(key[1])-self._offset] = self.transitions[key]
        x_axis_labels = [chr(x) for x in range(self._offset, self._offset+self._delta)]
        return sparseArray, x_axis_labels
    """Генерация самопереходов"""
    def generateSparseSame(self, labels, depth):
        sparseDimension = dict()
        everyL = 0
        partL = 0
        for phrase in self.phrases:
            words = phrase.split(' ')
            for word in words:
                length = len(word)
                if length > depth:
                    for i, letter in enumerate(word):
                        everyL += 1
                        if i+depth < length:
                            if word[i+depth] in [s for s in convoluteZone(labels) if letter in s][0]:
                                self.incOrAdd((letter, word[i+depth]), sparseDimension)
                                partL += 1
        return (sparseDimension, everyL, partL)

    def generateDframe(self):
        array, labels = self.generateArrayLabels()
        sparDf = pd.DataFrame(array)
        sparDf.columns = labels
        sparDf.index = labels
        return sparDf

    def prepDfToNp(self, Dframe):
        return (Dframe.index.to_numpy().astype(np.unicode_), Dframe.to_numpy().astype(np.float), Dframe.columns.to_numpy().astype(np.unicode_))

    def saveDfToNp(self):
        self.DfToNp = self.prepDfToNp(self.generateDframe())


def convoluteZone(labels):  # returns array of str(letters) by finger zones
    newZone = []
    for x, row in enumerate(labels):
        i = 0
        for y, label in enumerate(row):
            if y in [4,6] or y > 9:
                newZone[i-1] += label
            else:
                if x==0:
                    newZone.append(label)
                else:
                    newZone[i] += label
                    i += 1
    return newZone


def convoluteParts(labels):
    newPart = []
    for x, row in enumerate(labels):
        i = -1
        for y, label in enumerate(row):
            if x==0 and (y==0 or y==5):
                newPart.append(label)
            else:
                if y==0 or y==5:
                    i+=1
                newPart[i] += label
    return newPart


