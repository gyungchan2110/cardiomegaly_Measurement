# In[]
import measurement 
import time
import os
from datetime import datetime
import csv
import cv2
import numpy as np
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="0"
base =  "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)"

OriginalImgs = [base + "/Imgs_OriginalData_2k2k/Imgs/Normal"]
                # base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/1_AS",
                # base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/2_AR",
                # base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/3_MS",
                # base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/4_MR",
                # base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/5_AS+AR",
                # base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/6_MS_MR" ]
               

Masks = [base + "/Masks_OriginalData_OriginalSize/Normal"]
        # base + "/Masks_OriginalData_OriginalSize/Abnormal/1_AS",
        # base + "/Masks_OriginalData_OriginalSize/Abnormal/2_AR",
        # base + "/Masks_OriginalData_OriginalSize/Abnormal/3_MS",
        # base + "/Masks_OriginalData_OriginalSize/Abnormal/4_MR",
        # base + "/Masks_OriginalData_OriginalSize/Abnormal/5_AS+AR",
        # base + "/Masks_OriginalData_OriginalSize/Abnormal/6_MS_MR" ]

Dst = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/DiaphragmLevelData"

currentTime = datetime.now()
for i, imgPath in enumerate(OriginalImgs):

    for filename in os.listdir(imgPath):
        
        # if (filename != "Img_20180130_162633.png"):
        #     continue
        print(filename)

        Diaphragm = Masks[i] + "/Diaphragm/" + filename 
        rib_9 = Masks[i] + "/Rib(9)/" + filename 
        rib_10 = Masks[i] + "/Rib(10)/" + filename 

        diaphragm = measurement.get_Point(Diaphragm)
        rib9 = measurement.get_Point(rib_9)
        rib10 = measurement.get_Point(rib_10)


        m_diaphragmLevel = measurement.diaphragm_Level(rib9, rib10, diaphragm)
        
        folder = ""
        if(m_diaphragmLevel == 2):
            folder = "Class_2"
        elif(m_diaphragmLevel == 1):
            folder = "Class_1"
        else:
            folder = "Class_0"
        
        
        shutil.copy2(imgPath + "/" + filename, Dst + "/" + folder + "/" + filename)