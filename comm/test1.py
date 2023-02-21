# 均衡化
import cv2
from PIL import Image
import numpy as np
import SimpleITK as sitk
import imageio
import os

def read_nii(nii_file):
    img = sitk.ReadImage(nii_file)
    img_fdata = sitk.GetArrayFromImage(img)
    return img_fdata


def save_fig(nii_file, max_slice, sava_path):
    fdata = read_nii(nii_file)
    (z, x, y) = fdata.shape
    print(z)
    max_mask = fdata[max_slice, :, :]
    print(np.max(max_mask))
    # print(max_mask.shape)
    # 黑色为0, 白色是255
    imageio.imwrite(sava_path, max_mask*255)
    return sava_path


def masked_img(origin_img, mask, masked_path):
    origin_img = cv2.imread(origin_img)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = mask / np.max(mask)
    image = np.zeros(origin_img.shape)
    image[:, :, 0] = origin_img[:, :, 0] * mask
    image[:, :, 1] = origin_img[:, :, 1] * mask
    image[:, :, 2] = origin_img[:, :, 2] * mask
    cv2.imwrite(masked_path, image)
    return masked_path


def draw_roi(img):
    src = cv2.imread(img)
    proimage0 = src.copy()  # 复制原图

    roi = cv2.selectROI(windowName="roi", img=src, showCrosshair=False, fromCenter=False)  # 感兴趣区域ROI
    x, y, w, h = roi
    cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)  # 在图像绘制区域
    cv2.imshow("roi", src)

    # 进行裁剪
    ImageROI = proimage0[y:y + h, x:x + w]
    cv2.imwrite("croped.png", ImageROI)
    cv2.imshow("ImageROI", ImageROI)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_roi(img, save_path, img_size=224):
    img_data = cv2.imread(img)
    resized_data = cv2.resize(img_data, (img_size, img_size))
    cv2.imwrite(save_path, resized_data)
    return save_path


def rect_roi(img_path, save_path):
    img_data = cv2.imread(img_path)  # np.array -> [h, w, c]
    # print(img_data.shape)
    # tuple的每一个元素, 从一个维度来确定非零元素的位置`
    img_nonzero = np.nonzero(img_data)
    max_row = np.max(img_nonzero[0])
    min_row = np.min(img_nonzero[0])

    max_column = np.max(img_nonzero[1])
    min_column = np.min(img_nonzero[1])
    bound_row = (int)((max_row - min_row)*0.12)
    bound_column = (int)((max_column - min_column)*0.12)

    # 进行裁剪
    image_roi = img_data[min_row - bound_row: max_row + bound_row, min_column - bound_column: max_column + bound_column]
    cv2.imwrite(save_path, image_roi)

    return save_path


def main(nii_path, nii_max_slice, nii_mask_save_path, v_png, merged_save_path, final_save_path):
    # 读取 nii 文件 并取最大层面保存为 png
    mask_img = save_fig(nii_path, nii_max_slice, nii_mask_save_path)
    # 把 mask 和 原始的 v_img 作融合
    merge_img = masked_img(v_png, mask_img, merged_save_path)
    # 剪裁 roi
    roi_img = rect_roi(merge_img, final_save_path)
    final_img = resize_roi(roi_img, final_save_path)
    print("The final roi img is save ... " + final_img)


if __name__ == '__main__':

    nii_path, nii_max_slice, nii_mask_save_path, v_png, merged_save_path, final_save_path = \
        r"C:\Users\Lenovo\Desktop\mvi_figs\samples\105 yu qinyi-v.nii.gz",\
        127,\
        'nii_mask.png',\
        r"C:\Users\Lenovo\Desktop\mvi_figs\samples\105_clp_y_v.png", \
        '105_clp_y_v_merged.png',\
        '105_clp_y_v_final.png'

    main(nii_path, nii_max_slice, nii_mask_save_path, v_png, merged_save_path, final_save_path)
    rect_roi(nii_mask_save_path, 'nii_mask_croped.png')
    resize_roi('nii_mask_croped.png', 'nii_mask_resied.png')