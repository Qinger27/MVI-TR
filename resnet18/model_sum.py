import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
from resnetModel import getResnet18
from resnet50 import getResnet50
from models import getResNet101
import torch
from trainResnet18 import load_data, read_split_data, MyDataSet, evaluate, predict, draw_figs
from tqdm import tqdm
import sys
sys.path.append("../vit")
from vit_model import vit_base_patch16_224
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_fscore_support, precision_recall_curve
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import pandas as pd


@torch.no_grad()
def predict(model, device, train_loader):

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()

    sample_num = 0

    data_loader = tqdm(train_loader, file=sys.stdout)

    # plot_roc_curve()
    acc, prec, rec, f1, = 0, 0, 0, 0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))

        print(pred)
        pred = torch.nn.Softmax(dim=1)(pred)
        print("==================== after softmax ==========")
        print(pred)
        pred_classes = torch.max(pred, dim=1)[1]
        np.save("pred_")
        pred_max = torch.max(pred, dim=1)[0]
        print("===========predicted result =============")
        print(pred_classes)
        print("===========labels =======================")
        print(labels)
        acc = accuracy_score(labels.cpu(), pred_classes.cpu(), normalize=True)
        fpr, tpr, thresholds = roc_curve(labels.cpu(), pred[:, 1].cpu(), pos_label=1)
        # fpr, tpr, thresholds = roc_curve(labels.cpu(), pred_max.cpu(), pos_label=1)
        # roc_auc = roc_auc_score(labels.cpu(), pred[:, 1].cpu())
        roc_auc = roc_auc_score(labels.cpu(), pred_max.cpu())
        # print(f"fpr: {fpr}")
        # print(f"tpr: {tpr}")
        # print(f"thresholds: {thresholds}")

        prec, rec, f1, _ = precision_recall_fscore_support(labels.cpu(), pred_classes.cpu(), average="binary")
        # acc += acc
        # roc_auc += roc_auc
        # prec += prec
        # rec += rec
        # f1 += f1
        data_loader.desc = "[predict acc: {:.3f}, auc: {:.3f}, prec: {:.3}, rec: {:.3}, f1_score: {:.3}"\
            .format(acc, roc_auc, prec, rec, f1)

        precisions, recallls, thresh = precision_recall_curve(labels.cpu(), pred[:, 1].cpu(), pos_label=1, sample_weight=None)

        # plt.plot(recalls, precisions)
        # plt.show()

    return acc, tpr, fpr, roc_auc, precisions, recallls, thresh

@torch.no_grad()
def results(model, model_name, train_loader):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sample_num = 0

    data_loader = tqdm(train_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred = torch.nn.Softmax(dim=1)(pred)
        # print("==================== after softmax ==========")
        # print(pred)
        pred_classes = torch.max(pred, dim=1)[1]
        np.save(f"pred_{model_name}.npy", pred.cpu().detach().numpy())
        np.save(f"label_{model_name}.npy", labels.cpu().detach().numpy())
        pred_max = torch.max(pred, dim=1)[0]
        # print("===========labels =======================")
        # print(labels)



def param_count():
    model = getResnet18()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


def flops_count():
    from thop import profile
    model = vit_base_patch16_224()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print(f"flops is {flops}")
    print(f"para is {params}")
def main():

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data_path = "/home/wq/dataLiver/roi_total/"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    for i in range(len(val_images_label)):
        print(val_images_path[i], "   ", val_images_label[i])

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    # total_data = MyDataSet(train_images_path, train_images_label, transforms.Compose([transforms.ToTensor()]))
    val_data = MyDataSet(val_images_path, val_images_label, transforms.Compose([transforms.ToTensor()]))
    # train_loader = torch.utils.data.DataLoader(total_data,
    #                                            batch_size=448,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=0,
    #                                            collate_fn=total_data.collate_fn)

    train_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=111,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=val_data.collate_fn)

    # weights50 = "/home/wq/wangqing/resnet18/weights_total_resnet50/model-14.pth"
    # weights18 = "/home/wq/wangqing/resnet18/weights_total_resnet18/model-8.pth"
    weights101 = "/home/wq/wangqing/resnet18/weights_resnet101/model-257.pth"
    weightsvit = "/home/wq/wangqing/vit/weights4_total/model-2025.pth"
    # weightsvit = "/home/wq/wangqing/vit/weights_vit_768_112_112/model-570.pth"
    # weightsvit1024 = "/home/wq/wangqing/vit/weights_vit_1024_64_64/model-1285.pth"
    # weightsvit512 = "/home/wq/wangqing/vit/weights_vit_512_64_16/model-569.pth"


    # resnet50model = getResnet50()
    # weight1 = torch.load(weights50, map_location=device)
    # resnet50model.load_state_dict(weight1, strict=True)

    # results(resnet50model, "resnet50", train_loader)

    # resnet18model = getResnet18()
    # weight2 = torch.load(weights18, map_location=device)
    # resnet18model.load_state_dict(weight2, strict=True)
    # results(resnet18model, "resnet18", train_loader)

    # resnet101model = getResNet101()
    # weight3 = torch.load(weights101, map_location=device)
    # resnet101model.load_state_dict(weight3, strict=True)
    # # resnet101model = torch.nn.DataParallel(resnet101model)
    # results(resnet101model, "resnet101", train_loader)

    vitmodel = vit_base_patch16_224(num_classes=2)
    weight4 = torch.load(weightsvit, map_location=device)
    vitmodel.load_state_dict(weight4, strict=True)
    results(vitmodel, "vit", train_loader)

    # acc1, tpr1, fpr1, auc1, pre1, rec1, _ = predict(resnet50model, device, train_loader)
    # np.save("fpr1_val.npy", fpr1)
    # np.save("tpr1_val.npy", tpr1)
    # np.save("pre1_val.npy", pre1)
    # np.save("rec1_val.npy", rec1)

    # acc2, tpr2, fpr2, auc2, pre2, rec2, _ = predict(resnet18model, device, train_loader)
    # np.save("fpr2_val.npy", fpr2)
    # np.save("tpr2_val.npy", tpr2)
    # np.save("pre2_val.npy", pre2)
    # np.save("rec2_val.npy", rec2)

    # plt.figure()
    # plt.plot(precisions1, recallls1)
    # plt.savefig("test.png")
    # plt.show()


    # acc3, tpr3, fpr3, auc3, pre3, rec3, _  = predict(resnet101model, device, train_loader)
    # np.save("fpr3_val.npy", fpr3)
    # np.save("tpr3_val.npy", tpr3)
    # np.save("pre3_val.npy", pre3)
    # np.save("rec3_val.npy", rec3)


    # acc4, tpr4, fpr4, auc4, pre4, rec4, _  = predict(vitmodel, device, train_loader)
    # np.save("fpr4_val.npy", fpr4)
    # np.save("tpr4_val.npy", tpr4)
    # np.save("pre4_val.npy", pre4)
    # np.save("rec4_val.npy", rec4)

    # plt.figure()
    # plt.plot(fpr1, tpr1, label="resnet50")
    # plt.plot(fpr2, tpr2, label="resnet18")
    # plt.plot(fpr3, tpr3, label="resnet101")
    # plt.plot(fpr4, tpr4, label="vit")
    # plt.legend(loc='best')
    # plt.title("Deep learning models' result")
    # plt.savefig("roc_curve_on_val_data.png", dpi=500)
    # plt.show()


def draw_rou():

    fpr1 = np.load("fpr1.npy")
    tpr1 = np.load("tpr1.npy")

    fpr2 = np.load("fpr2.npy")
    tpr2 = np.load("tpr2.npy")

    fpr3 = np.load("fpr3.npy")
    tpr3 = np.load("tpr3.npy")

    fpr4 = np.load("fpr4.npy")
    tpr4 = np.load("tpr4.npy")
    plt.figure()
    plt.plot(fpr1, tpr1, label="resnet50")
    plt.plot(fpr2, tpr2, label="resnet18")
    plt.plot(fpr3, tpr3, label="resnet101")
    plt.plot(fpr4, tpr4, label="vit")
    plt.legend(loc='best')
    plt.title("Deep learning models")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive score")
    plt.savefig("roc_curve_on_val_data.png", dpi=500)
    plt.show()


def cam_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet50model = getResnet50()
    weights50 = "/home/wq/wangqing/resnet18/weights_total_resnet50/model-14.pth"
    weight1 = torch.load(weights50, map_location=device)
    resnet50model.load_state_dict(weight1, strict=True)
    target_layers = [resnet50model.layer3]

    # weights101 = "/home/wq/wangqing/resnet18/weights_resnet101/model-257.pth"
    # resnet101model = getResNet101()
    # weight3 = torch.load(weights101, map_location=device)
    # resnet101model.load_state_dict(weight3, strict=True)
    # # resnet101model = torch.nn.DataParallel(resnet101model)
    # target_layers = [resnet101model.encoder.layer2]

    model_name = "resnet50"
    cam_path = './cam_figs_resnet50_v3'
    if not os.path.exists(cam_path):
        os.makedirs(cam_path)

    # resnet18model = getResnet18()
    # weights18 = "/home/wq/wangqing/resnet18/weights_total_resnet18/model-8.pth"
    # weight2 = torch.load(weights18, map_location=device)
    # resnet18model.load_state_dict(weight2, strict=True)
    # print(resnet18model)
    # target_layers = [resnet18model.layer4]

    data_transform = transforms.Compose([transforms.ToTensor()])

    for img1 in os.listdir("/home/wq/dataLiver/roi_total/yy/"):
        image_path = os.path.join("/home/wq/dataLiver/roi_total/yy/", img1)
        img_name = image_path.split('/')[-1].split('.')[0]
        # print(img_name)
        img = Image.open(image_path).convert("RGB")
        img = np.array(img, dtype=np.uint8)

        # [C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=resnet50model, target_layers=target_layers, use_cuda=False)
        target_category = 1  # tabby, tabby cat
        # target_category = 254  # pug, pug-dog

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.xlabel("Pixels in x axis")
        plt.ylabel("Pixels in y axis")

        save_path = os.path.join(cam_path, f"{img_name}_{model_name}.png")
        plt.savefig(save_path)
    # plt.show()


def npy_read():

    load_npy = np.load("pred_vit.npy")
    print(load_npy)

if __name__ == '__main__':
    # param_count()
    # cam_model()
    # main()
    # draw_rou()
    # npy_read()
    flops_count()