### 本文件用于均衡化
import cv2
import matplotlib.pyplot as plt
import os


def process_img(root_path, root_new_path):
    if not os.path.exists(root_new_path):
        os.makedirs(root_new_path)

    for status in os.listdir(root_path):
        status_path = os.path.join(root_path, status)
        status_new_path = os.path.join(root_new_path, status)
        if not os.path.exists(status_new_path):
            os.makedirs(status_new_path)
        for img in os.listdir(status_path):
            img_path = os.path.join(status_path, img)
            img_new_path = os.path.join(status_new_path, img)
            imgdata = cv2.imread(img_path, 0)
            clahe = cv2.createCLAHE(2, (8, 8))
            equ1 = clahe.apply(imgdata)
            cv2.imwrite(img_new_path, equ1)


def equ_global(ima_path, mask_path):
    img = cv2.imread(ima_path, 0)  # 0 表示转换成灰度格式
    mask = cv2.imread(mask_path, 0)
    # # 全局均衡化
    equ_img = cv2.equalizeHist(img)
    cv2.imwrite("equ.png", equ_img)
    imghist = cv2.calcHist([equ_img], [0], mask, [256], [0, 256])
    plt.hist(imghist.ravel(), 256)
    plt.savefig("test3.png")
    plt.show()


def equ_adap(ima_path, mask_path):

    img = cv2.imread(ima_path, 0)  # 0 表示转换成灰度格式
    mask = cv2.imread(mask_path, 0)
    equ_img = cv2.equalizeHist(img)
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(2, (8, 8))
    equ1 = clahe.apply(img)
    imghist = cv2.calcHist([equ1], [0], mask, [256], [0, 256])
    plt.hist(imghist.ravel(), 256)
    plt.savefig("test2.png")
    plt.show()

    equhist = cv2.calcHist([equ_img], [0], mask, [256], [0, 256])
    plt.plot(equhist, color='g')
    plt.show()









