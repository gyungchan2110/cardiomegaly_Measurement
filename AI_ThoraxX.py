
# In[]
import numpy as np

import cv2 
import csv  
import os 
from skimage.morphology import skeletonize
import measurement


height = 2048
def MakeDataSet(inputDataPath):

    #Folders = ["test", "train","validation"]

    #for folder in Folders: 
    Xpath = inputDataPath + "/Thorax(x)"
    Ypath = inputDataPath + "/Thorax(y)"
    dst = inputDataPath + "/LandMark"

    if not os.path.isdir(dst):
        os.mkdir(dst)

    #path = inputDataPath 
    csvDataPath = dst + "/xLevel.csv"
    for file in os.listdir(Xpath):
        
        if(file[-4:] != ".png"):
            continue 

        filepath = Xpath + "/" + file  

        mask = cv2.imread(filepath, 0)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = np.asarray(mask)
        height, width = mask.shape

        hist1 = np.zeros( height )
        indeces = np.arange(height)
        

        for i in range(0, height):
            hist1[i] = mask[i,:].sum()

        xLevel = np.argmax(hist1)

        xl = float(xLevel) / float(height)




        filepath = Ypath + "/" + file  

        mask = cv2.imread(filepath, 0)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        

        hist2 = np.zeros( width )
        indeces = np.arange(width)
        

        for i in range(0, width):
            hist2[i] = mask[:,i].sum()

        yLevel = np.argmax(hist2)

        yl = float(yLevel) / float(width)

        newmask = np.zeros((height,width))

        newmask = cv2.rectangle(newmask, (yLevel - 10, xLevel-10),(yLevel + 10, xLevel + 10), thickness = -1, color = (255,255,255))

        cv2.imwrite(dst + "/" + file, newmask)

        f = open(csvDataPath,"a", encoding='utf-8', newline='')
        fwriter = csv.writer(f)

        strLine = []
        strLine.append(file)
        strLine.append(height)
        strLine.append(width)
        strLine.append(xl)
        strLine.append(yl)

        fwriter.writerow(strLine)
        f.close()
        print(file)


#datasetPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/1_AS"
datasetPaths = []
datasetPaths.append("D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/1_AS")
datasetPaths.append("D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/2_AR")
datasetPaths.append("D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/3_MS")
datasetPaths.append("D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/4_MR")
datasetPaths.append("D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/5_AS+AR")
datasetPaths.append("D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Abnormal/6_MS_MR")

for datasetPath in datasetPaths:
    MakeDataSet(datasetPath)