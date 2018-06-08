
# In[]
import measurement 
import matplotlib.pyplot as plt
import curvature
import time
import os
maskPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks/Normal/Aortic Knob/Img_20180130_170115.png"
maskPath2 = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks/Normal/Pulmonary Conus/Img_20180130_170115.png"

os.environ["CUDA_VISIBLE_DEVICES"]="0"
start_time = time.time()
carina = measurement.get_DicomSpacingValue("Img_20180130_170115")
#sd = measurement.get_Vertical_Line_points(maskPath2)

#a = measurement.join_line_elements(carina, sd)
print(carina)
print("--- %s seconds ---" %(time.time() - start_time))
# plt.plot(a)
# plt.show()
# pts = measurement.Curvature(aortic)
# maskPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks/Normal/Lt Lower CB/Img_20180130_170115.png"
# aortic = measurement.get_Vertical_Line_points(maskPath)
# plt.plot(aortic)
# plt.show()
# pts = measurement.Curvature(aortic)

# In[]
import measurement 
import time
import os
from datetime import datetime
import csv
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
base =  "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)"

OriginalImgs = [
                base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/1_AS",
                base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/2_AR",
                base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/3_MS",
                base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/4_MR",
                base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/5_AS+AR",
                base + "/Imgs_OriginalData_2k2k/Imgs/Abnormal/6_MS_MR" ]
               

Masks = [
        base + "/Masks_OriginalData_OriginalSize/Abnormal/1_AS",
        base + "/Masks_OriginalData_OriginalSize/Abnormal/2_AR",
        base + "/Masks_OriginalData_OriginalSize/Abnormal/3_MS",
        base + "/Masks_OriginalData_OriginalSize/Abnormal/4_MR",
        base + "/Masks_OriginalData_OriginalSize/Abnormal/5_AS+AR",
        base + "/Masks_OriginalData_OriginalSize/Abnormal/6_MS_MR" ]


currentTime = datetime.now()
for i, imgPath in enumerate(OriginalImgs):
    currentTime = datetime.now()
    ReportFileName = 'D:/Temp/TestReport_%04d%02d%02d_%02d%02d%02d.csv' %(currentTime.year, currentTime.month, currentTime.day,currentTime.hour, currentTime.minute, currentTime.second)
    #ReportFileName = imgPath + ReportFileName
    
    rows = []
    rows.append("PatientID")
    rows.append("filename")
    rows.append("Thorax_X_Length")
    rows.append("Distance_Rt_Axis")
    rows.append("Distance_Lt_Axis")
    rows.append("CT_Ratio")

    for k in range(7):
        rows.append("Length to Distance Ratio")
        rows.append("Curvature_Max")
        rows.append("Curvature_Min")
        rows.append("Curvature_Mean")
    rows.append("Carina_Angle")
    rows.append("Level_Diaphragm")
    rows.append("Area_Cardiac")
 
    f = open(ReportFileName, 'a', encoding='utf-8', newline='')
    f_writer = csv.writer(f)
    f_writer.writerow(rows)
    

    for filename in os.listdir(imgPath):
        
        if (filename != "Img_20180130_162633.png"):
            continue
        print(filename)
        rows = []
        start_time = time.time()

        thoraxYPath = Masks[i] + "/Thorax(y)/" + filename 
        thoraxxPath = Masks[i] + "/Thorax(x)/" + filename 
        RtLowerCB = Masks[i] + "/Rt Lower CB/" + filename 
        LtLowerCB = Masks[i] + "/Lt Lower CB/" + filename 
        RtUpperCB = Masks[i] + "/Rt Upper CB/" + filename 
        Diaphragm = Masks[i] + "/Diaphragm/" + filename 
        rib_9 = Masks[i] + "/Rib(9)/" + filename 
        rib_10 = Masks[i] + "/Rib(10)/" + filename 
        Aortic = Masks[i] + "/Aortic Knob/" + filename 
        Conus = Masks[i] + "/Pulmonary Conus/" + filename 
        LAA = Masks[i] + "/LAA/" + filename 
        DAO = Masks[i] + "/DAO/" + filename 
        Carina = Masks[i] + "/Carina/" + filename 

        files = []
        files.append(thoraxYPath)
        files.append(thoraxxPath)
        files.append(RtLowerCB)
        files.append(LtLowerCB)
        files.append(RtUpperCB)
        files.append(Diaphragm)
        files.append(rib_9)
        files.append(rib_10)
        files.append(Aortic)
        files.append(Conus)
        files.append(LAA)
        files.append(DAO)
        files.append(Carina)

        for file in files : 
           # path = file + "/" + "Img_20180130_162633.png"
            img = cv2.imread(file)
            img = np.asarray(img, dtype = "uint8")
            print(img.shape)
            ret, thresh = cv2.threshold(img[:,:,0], 127, 255, cv2.THRESH_BINARY)
            temp, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE  )
            cv2.drawContours(img, contours, 0, color = (255,255,255), thickness = -1 )
            cv2.imwrite(file, img)
        h_spacing, w_spacing,PID = measurement.get_DicomSpacingValue(filename[:-4])
        rows.append(str(PID))
        rows.append(filename)
        thoraxys = measurement.get_Vertical_Line_points(thoraxYPath)
        
        aortic = measurement.get_Vertical_Line_points(Aortic)
        conus = measurement.get_Vertical_Line_points(Conus)
        laa = measurement.get_Vertical_Line_points(LAA)
        lts = measurement.get_Vertical_Line_points(LtLowerCB)

        dao = measurement.get_Vertical_Line_points(DAO)

        rts = measurement.get_Vertical_Line_points(RtLowerCB)
        rtupper = measurement.get_Vertical_Line_points(RtUpperCB)
        

        
        carina = measurement.get_Horizontal_Line_points(Carina)

        




        carina_level, level_index = measurement.get_carina_level(carina)



        diaphragm = measurement.get_Point(Diaphragm)
        rib9 = measurement.get_Point(rib_9)
        rib10 = measurement.get_Point(rib_10)

        # m_thorax_x_length = measurement.length_Thorax_X(thoraxxPath)
        rows.append(filename[:-4])
        m_thoraxx_length, thoraxx_Level = measurement.length_Thorax_X(thoraxxPath,h_spacing, w_spacing)
        print(m_thoraxx_length, thoraxx_Level)
        rows.append(str(m_thoraxx_length))
        m_rtDistance, m_ltDistance = measurement.length_RtLowerCB_Thorax_y(thoraxys, rts, lts, diaphragm,h_spacing, w_spacing)
        rows.append(str(m_rtDistance))
        rows.append(str(m_ltDistance))
        m_CTRatio = (m_rtDistance + m_ltDistance)/m_thoraxx_length
        rows.append(str(m_CTRatio))

        m_ratio = measurement.get_distanceLengthRatio(aortic,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(aortic,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_ratio = measurement.get_distanceLengthRatio(conus,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(conus,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_ratio = measurement.get_distanceLengthRatio(laa,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(laa,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_ratio = measurement.get_distanceLengthRatio(lts,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(lts,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_ratio = measurement.get_distanceLengthRatio(rts,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(rts,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_ratio = measurement.get_distanceLengthRatio(rtupper,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(rtupper,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_ratio = measurement.get_distanceLengthRatio(dao,h_spacing, w_spacing)
        rows.append(str(m_ratio))
        m_Cur_MAx, m_Cur_min, m_Cur_Mean = measurement.Curvature(dao,h_spacing, w_spacing)
        rows.append(str(m_Cur_MAx))
        rows.append(str(m_Cur_min))
        rows.append(str(m_Cur_Mean))

        m_carinaAngle = measurement.carina_angle(carina, carina_level, level_index, h_spacing, w_spacing)
        rows.append(str(m_carinaAngle))

        m_diaphragmLevel = measurement.diaphragm_Level(rib9, rib10, diaphragm)
        rows.append(str(m_diaphragmLevel))

        m_cardiacArea = measurement.Area(rtupper, rts, aortic, conus, laa,lts,carina_level,thoraxx_Level,h_spacing, w_spacing)
        rows.append(str(m_cardiacArea))
        f = open(ReportFileName, 'a')
        #f.write(filename + "," + str(m_thoraxx_length) + "," + str(m_rtDistance) + "," + str(m_ltDistance) + "," + str(m_aortic_ratio) + "," + str(m_conus_ratio) + "," + str(m_laa_ratio) + "," + str(m_ltlower_ratio) + "," + str(m_rtlower_ratio) + "," + str(m_rtupper_ratio) + "," + str(m_dao_ratio) + "," + str(m_carinaAngle) + "," + str(m_diaphragmLevel) + "," + str(m_cardiacArea) +"\n" )
        f_writer.writerow(rows)
        print(filename + " : --- %s seconds ---" %(time.time() - start_time))
        #break
    print("Folder Done!!")
    f.close()
# In[]
import measurement 
rib_9 = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks/Normal/Rib(9)/Img_20180130_170115.png"
rib_10 = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks/Normal/Rib(10)/Img_20180130_170115.png"
Diaphragm = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks/Normal/Diaphragm/Img_20180130_170115.png"
measurement.diaphragm_Level(rib_9, rib_10, Diaphragm)