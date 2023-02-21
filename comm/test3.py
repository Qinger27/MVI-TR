import numpy as np
import os
import cv2
from PIL import Image
import shutil
import pydicom as dicom
import matplotlib.pyplot as plt
import pandas as pd
import re


def L2RGB(img_path):
    img = Image.open(img_path).convert("RGB")
    img.save(img_path)  # 原地保存


def cal_HU(dicom_path):
    img_ds = dicom.read_file(dicom_path)
    rescaleIntercept = np.float(img_ds.RescaleIntercept)
    rescaleSlope = np.float(img_ds.RescaleSlope)
    img_pixel = img_ds.pixel_array
    img_ct = img_pixel * rescaleSlope + rescaleIntercept  # 转换成HU
    return img_ct


def HU_conversion(img_data, WW=400, WL=40):
    """
    CT HU value conversion.0000 0 000 000000000000000000000000000000000000
    Arguments
        img_data: Two-dimensional array of CT value pixels. (-1024HU - 3071HU)
        WW: int, set Window Width as 400 to focus on the liver part in abdominal CT scans.
        WL: int, set Window Level as 40 to focus on the liver part in abdominal CT scans.
    Returns
        Two-dimensional array of CT pixels after HU conversion.(0-255 intensity)
    """
    img_temp = img_data
    min = (2 * WL - WW) / 2.0 + 0.5
    max = (2 * WL + WW) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = ((img_temp - min) * dFactor).astype(int)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp


def name2id(mvi_root):
    patient_dic = {}
    for patient in os.listdir(mvi_root):
        if os.path.isdir(os.path.join(mvi_root, patient)):
            patient_id = re.findall('\d+', patient)
            for i in patient_id:
                m = i
            patient_dic[m] = patient
    return patient_dic

