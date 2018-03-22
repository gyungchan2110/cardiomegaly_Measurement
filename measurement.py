from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from  operator import eq
import math
import curvature
import time
import csv
import SimpleITK as sitk



def get_DicomSpacingValue(filename):
    metaFile = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/BasicData_MetaFile_Ex.csv"
    PID = 0
    f = open(metaFile, 'r', encoding='utf-8', newline='')
    reader = csv.reader(f)
    filePath = ""
    for line in reader:
        datas = line
        if(eq(datas[2], filename)) : 
            filePath = datas[0] + "/" + datas[1] + "/" + datas[1] + ".dcm"
            PID = int(datas[3])
            break
    f.close()
    if eq(filePath, ""):
        return 0., 0.
    img = sitk.ReadImage(filePath)
    try:
        spacings = img.GetMetaData("0018|1164") 
    except :
        spacings = img.GetMetaData("0028|0030")
    
    splits = spacings.split("\\")
    return float(splits[0]), float(splits[1]),PID




def Curvature(line_pts, h_spacing, w_spacing):
    line_pts = np.asarray(line_pts)
    x = line_pts[line_pts > 0]
    y = np.arange(len(x))
    x = x * h_spacing
    y = y * w_spacing
    a = curvature.curvature_splines(x, y)

    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('aortic', color=color)
    # ax1.plot(y, x, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('curvature', color=color)  # we already handled the x-label with ax1
    # ax2.plot(y, a, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    a = a[1:-1]
    # print(a.max())
    # print(a.min())
    # print(a.mean())
    return a.max(), a.min(), a.mean()


def length_Thorax_X(Thorax_x_Path, h_spacing, w_spacing):
    
    img = cv2.imread(Thorax_x_Path)
    img = np.array(img, dtype = "int16")
    
    height, width, depth = img.shape
    hist1 = np.zeros( height )
    indeces = np.arange(height)
    ret, thresh = cv2.threshold(img[:,:,0], 127, 255, cv2.THRESH_BINARY)

    for i in range(0, height):
        hist1[i] = thresh[i,:].sum()

    
    hist1 = hist1 /255
    temp = hist1[hist1>0]
    length = len(temp)
    thorax_level = 0
    if length is not 0:
        thorax_level = np.dot(hist1, indeces) / (length * hist1.max())
    return hist1.max() * w_spacing, thorax_level

def length_RtLowerCB_Thorax_y(thoraxys, rts, lts, diaphragm, h_spacing, w_spacing):
    
    height = len(thoraxys)

    thoraxys = np.asarray(thoraxys)
    rts = np.asarray(rts)
    lts = np.asarray(lts)
    
    rt_lengths = np.zeros(height)
    lt_lengths = np.zeros(height)
    rt_lengths = thoraxys - rts
    lt_lengths = lts - thoraxys
   
    for i in range(0, height):
        if(rt_lengths[i] < 0 or rts[i] == 0):
            rt_lengths[i] = 0

        if(lt_lengths[i] < 0 or lts[i] == 0):
            lt_lengths[i] = 0
    #print(diaphragm)
    rt_lengths = rt_lengths[:diaphragm]
    lt_lengths = lt_lengths[:diaphragm]

    return rt_lengths.max()*w_spacing, lt_lengths.max()*w_spacing


def carina_angle(carina_pts, level, levelindex, h_spacing, w_spacing):
    carina_pts = np.asarray(carina_pts)
    nonzero = carina_pts[carina_pts>0]
    distance = np.zeros(len(nonzero))
    nonzero = nonzero - level

    for i in range(0,len(distance)):
        distance[i] = math.sqrt( nonzero[i]* nonzero[i]* h_spacing*h_spacing + w_spacing*w_spacing*(i-levelindex)*(i-levelindex)  )
    
    left = 0 
    for i in range(0, len(distance)):
        if(distance[i] <= 5):
            left = nonzero[i] * h_spacing / distance[i]
            #print(i, distance[i], nonzero[i])
            break
    right = 0
    for i in range(len(distance)-1, 0, -1):
        if(distance[i] <= 5):
            right = nonzero[i] * h_spacing / distance[i]
            #print(i, distance[i], nonzero[i])
            break
    pi = 3.14159265359

    angle = math.acos(left) + math.acos(right)
    #print(math.acos(left)* 180/pi, math.acos(right)* 180/pi, angle * 180/pi)
    return angle * 180/pi

def diaphragm_Level(rib9, rib10,diaphragm): 
    

    diaphragm_Level = 2

    if(diaphragm > rib10) : 
        diaphragm_Level = 0
    elif (diaphragm <= rib10 and diaphragm > rib9):
        diaphragm_Level = 1
    else:
        diaphragm_Level = 2
    return diaphragm_Level

def Area(rtupper, rtlower, aortic, conus, laa, lts, carina_level, thoraxx_level, h_spacing, w_spacing):
    rtupper = np.asarray(rtupper)
    rtlower = np.asarray(rtlower)
    rt = join_line_elements(rtupper, rtlower)

    aortic = np.asarray(aortic)
    conus = np.asarray(conus)
    laa = np.asarray(laa)
    lts = np.asarray(lts)

    temp = join_line_elements(aortic, conus)
    temp = join_line_elements(temp, laa)
    lt = join_line_elements(temp, lts)

    startlevel = 0
    startIndex = 0
    endlevel = 0
    endIndex = 0
    flag = False
    for i in range(0, len(rt)):
        if(rt[i] > 0 and not flag):
            startlevel = rt[i]
            startIndex = i
            flag = True
        if(rt[i]== 0 and flag):
            endlevel = rt[i-1]
            endIndex = i - 1
            break

    rt[0:startIndex] = startlevel
    rt[endIndex:len(rt) -1] = endlevel 

    startlevel = 0
    startIndex = 0
    endlevel = 0
    endIndex = 0
    flag = False
    for i in range(0, len(lt)):
        if(lt[i] > 0 and not flag):
            startlevel = lt[i]
            startIndex = i
            flag = True
        if(lt[i]== 0 and flag):
            endlevel = lt[i-1]
            endIndex = i - 1
            break


    lt[0:startIndex] = startlevel
    lt[endIndex:len(lt) -1] = endlevel       
    diff = lt - rt
    area = diff[int(carina_level):int(thoraxx_level)].sum()
    return area * h_spacing * w_spacing

def get_distanceLengthRatio(lines, h_spacing, w_spacing):
    #lines = get_Vertical_Line_points(line_Path)
    lines = np.asarray(lines)
    actualLine = lines[lines>0]
    size = len(actualLine)
    indeces = np.ones(size - 1)
    diff = np.diff(actualLine)
    sq = np.multiply(diff, diff) 
    sq = sq * (h_spacing * h_spacing) + indeces * (w_spacing * w_spacing)
    sqr = np.zeros(size - 1)
    for i in range(size - 1):
        sqr[i] = math.sqrt(sq[i])
    length = sqr.sum() 
    distance = math.sqrt( h_spacing * h_spacing * (actualLine[size-1] - actualLine[0] ) * (actualLine[size-1] - actualLine[0]) + w_spacing * w_spacing*(size - 1) * (size - 1))    
    return length/distance



def join_line_elements(first_line, second_line):
    height = len(first_line)

    endIndex = 0
    for i in range(1, height):
        if(first_line[i] == 0 and first_line[i-1]> 0):
            endIndex = i
            break
    startIndex = 0
    for i in range(1, height):
        if(second_line[i - 1] == 0 and second_line[i] > 0):
            startIndex = i
            break

    first_line = np.asarray(first_line)
    second_line = np.asarray(second_line)
    joint_line = first_line + second_line
    if(endIndex < startIndex) : 
        for i in range(endIndex, startIndex):
            joint_line[i] = max(first_line[endIndex],second_line[startIndex])

    else:
        for i in range(startIndex, endIndex+1):
            #if(first_line[i] > second_line[i]):
            joint_line[i] = max(first_line[i],second_line[i])
            #else:
             #   joint_line[i] = second_line[i]

    return joint_line


def get_Vertical_Line_points(line_Path):
    line_Img = cv2.imread(line_Path)
    line_Img = np.array(line_Img, dtype = "int16")
    ret, Bin_line_Img = cv2.threshold(line_Img[:,:,0], 127, 255, cv2.THRESH_BINARY)
    
    height, width, depth = line_Img.shape
    #print(line_Img.shape)
    points = np.zeros(height)
    indeces = np.arange(width)

    for i in range(0, height):
        temp = Bin_line_Img[i,:]
        temp_= temp[temp>127]
        length = len(temp_)
        if length is 0:
            points[i] = 0
        else:
            points[i] = np.dot(temp, indeces) / ( 255 * length)

    return points


def get_Horizontal_Line_points(line_Path):
    line_Img = cv2.imread(line_Path)
    line_Img = np.array(line_Img, dtype = "int16")
    ret, Bin_line_Img = cv2.threshold(line_Img[:,:,0], 127, 255, cv2.THRESH_BINARY)
    
    height, width, depth = line_Img.shape
    #print(line_Img.shape)
    points = np.zeros(width)
    indeces = np.arange(height)

    for i in range(0, width):
        temp = Bin_line_Img[:,i]
        temp_= temp[temp>127]
        length = len(temp_)
        if length is 0:
            points[i] = 0
        else:
            points[i] = np.dot(temp, indeces) / ( 255 * length)

    return points

def get_Point(point_Path):
    
    point_Img = cv2.imread(point_Path)
    point_Img = np.array(point_Img, dtype = "int16")

    height, width, depth = point_Img.shape
    y_proj = np.zeros( height )
    y_count = np.zeros( height )
    x_indeces = np.arange(height)
    x_proj = np.zeros( width )
    x_count = np.zeros( width )
    y_indeces = np.arange(width)

    ret, Bin_Diaph = cv2.threshold(point_Img[:,:,0], 127, 255, cv2.THRESH_BINARY)
 
    # for i in range(height):
    #     temp = Bin_Diaph[i,:]
    #     temp_= temp[temp>127]
    #     length = len(temp_)
    #     if length is 0:
    #         y_proj[i] = 0
    #     else:
    #         y_proj[i] = np.dot(temp, y_indeces) / ( 255 * length)

    for i in range(width):
        temp = Bin_Diaph[:,i]
        temp_= temp[temp>127]
        length = len(temp_)
        if length is 0:
            x_proj[i] = 0
        else:
            x_proj[i] = np.dot(temp, x_indeces) / ( 255 * length)

    return int(x_proj.max())

def get_carina_level(carina_pts):
    
    carina_pts = np.asarray(carina_pts)
    temp = carina_pts[carina_pts>0]
    carina_level = temp.min()
    level_Index = 0

    for i in range(len(temp)):
        if temp[i] == carina_level:
            level_Index = i
            break

    return carina_level, level_Index
