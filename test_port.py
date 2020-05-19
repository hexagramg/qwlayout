import port as pt
import grader as gd
import unittest as ut 

class TestDataHandler(ut.TestCase):
    
    def testHandler(self):

        test = pt.DataHandler("dialogues.tsv")
        self.assertGreaterEqual(a= len(test.phrases), b= 1)
        self.assertGreaterEqual(len(test.transitions), 1)
        self.assertGreaterEqual(a= len(test.DfToNp), b= 1)
        self.dataHandler = test

    def testGrader(self):

        test = pt.DataHandler("dialogues.tsv")
        labels = [['й', 'ц', 'у', 'к', 'е', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ъ'], 
                ['ф', 'ы', 'в', 'а', 'п', 'р', 'о', 'л', 'д', 'ж', 'э', ''],
                ['я', 'ч', 'с', 'м', 'и', 'т', 'ь', 'б', 'ю', '', '', '']]
        gtest = gd.NiterateAlgo(labels, test.DfToNp, 3,[1,1,1,2,1])
        self.assertTrue(len(gtest) > 2)
    
    def testVerticalAlteration(self):

        test = pt.DataHandler("dialogues.tsv")
        labels = [['й', 'ц', 'у', 'к', 'е', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ъ'], 
                ['ф', 'ы', 'в', 'а', 'п', 'р', 'о', 'л', 'д', 'ж', 'э', ''],
                ['я', 'ч', 'с', 'м', 'и', 'т', 'ь', 'б', 'ю', '', '', '']]
        result = gd.Grades.NcalcVerticalAlteration(labels, pt.convoluteParts(labels), test.DfToNp)
        self.assertTrue(result > 0)

    def testUniformity(self):
        test = pt.DataHandler("dialogues.tsv")
        labels = [['й', 'ц', 'у', 'к', 'е', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ъ'],
                  ['ф', 'ы', 'в', 'а', 'п', 'р', 'о', 'л', 'д', 'ж', 'э', ''],
                  ['я', 'ч', 'с', 'м', 'и', 'т', 'ь', 'б', 'ю', '', '', '']]
        result = gd._debug_uniformity(labels, test.DfToNp)
        self.assertTrue(result > 0)
