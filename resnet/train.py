import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
import json
import torch.optim.lr_scheduler as lr_scheduler
from models import getResNet101, getResnet18, getResnet50, getResnet34
import random

from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score,precision_recall_fscore_support
import torch.optim as optim
from tqdm import tqdm
import sys
import math
import matplotlib.pyplot as plt



class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     print(img.mode)
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def read_split_data(data_path, val_rate=0.2):
    random.seed(0)
    assert os.path.exists(data_path), f"dataset root: {data_path} does not exist."

    classes = [cls for cls in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cls))]
    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))  # v: 索引 k: 类别名称
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        # print(os.listdir(cls_path))
        imgs = [os.path.join(cls_path, i) for i in os.listdir(cls_path)
                if os.path.splitext(i)[-1] in supported]
        imgs_label = class_indices[cls]

        val_path = random.sample(imgs, k=int(len(imgs) * val_rate))

        for img in imgs:
            if img in val_path:
                val_images_path.append(img)
                val_images_label.append(imgs_label)
            else:
                train_images_label.append(imgs_label)
                train_images_path.append(img)
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    return train_images_path, train_images_label, val_images_path, val_images_label


def load_data(train_images_path, train_images_label, val_images_path, val_images_label, data_transform, batch_size):

    train_data = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    val_data = MyDataSet(val_images_path, val_images_label, data_transform['val'])
    # print("train num: ", len(train_data))
    # print("val num: ", len(val_data))
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_data.collate_fn)

    # print("train loader num: ", len(train_loader))

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_data.collate_fn)
    # print("val loader num: ", len(val_loader))
    return train_loader, val_loader


def train_one_epoch(train_loader, epoch, model, criterion, optimizer, device):
    model.train()
    sum_loss = 0.
    acc_num = 0
    samples_num = 0
    optimizer.zero_grad()
    # train_loader = tqdm(train_loader, file=sys.stdout)
    # 从 0 开始
    for i, data in enumerate(train_loader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        # forward
        output = model(imgs)
        loss = criterion(output, labels)
        # backward
        loss.backward()

        pred_score, pred_class = torch.max(output, dim=1)  # 按行取max

        acc_num += accuracy_score(labels.cpu(), pred_class.cpu(), normalize=False)
        sum_loss += loss.detach()
        samples_num += imgs.shape[0]

        # train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.2f}".format(epoch,
        #                                                                         sum_loss / samples_num,
        #                                                                         acc_num / samples_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # update parameter
        optimizer.step()
        optimizer.zero_grad()
    print("[train epoch {}] loss: {:.3f}, acc: {:.2f}".
          format(epoch, sum_loss / samples_num, acc_num / samples_num))

    return sum_loss / samples_num, acc_num / samples_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    # data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / sample_num,
        #                                                                        accu_num.item() / sample_num)
    print("[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / sample_num,
                                                              accu_num.item() / sample_num))
    return accu_loss.item() / sample_num, accu_num.item() / sample_num


@torch.no_grad()
def predict(data_path, model, weight, device):
    model = model.to(device)
    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()

    sample_num = 0
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path, 0)
    # total_data = MyDataSet(train_images_path, train_images_label, transforms.Compose([transforms.ToTensor()]))
    val_data = MyDataSet(val_images_path, val_images_label, transforms.Compose([transforms.ToTensor()]))
    # train_loader = torch.utils.data.DataLoader(total_data,
    #                                            batch_size=301,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=0,
    #                                            collate_fn=total_data.collate_fn)

    data_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=111,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=val_data.collate_fn)
    # data_loader = tqdm(train_loader, file=sys.stdout)

    # plot_roc_curve()
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        print("===========predicted result =============")
        print(pred_classes)
        print("===========labels =======================")
        print(labels)
        acc = accuracy_score(labels.cpu(), pred_classes.cpu(), normalize=True)
        fpr, tpr, thresholds = roc_curve(labels.cpu(), pred[:, 1].cpu(), pos_label=1)
        roc_auc = roc_auc_score(labels.cpu(), pred[:, 1].cpu())
        # print(f"fpr: {fpr}")
        # print(f"tpr: {tpr}")
        # print(f"thresholds: {thresholds}")
        # #
        # print("ok")
        # cam_extractor = SmoothGradCAMpp(model, target_layer='layer1')
        # cam_extractor._hooks_enabled = True
        # model.zero_grad()
        prec, rec, f1, _ = precision_recall_fscore_support(labels.cpu(), pred_classes.cpu(), average="binary")
        data_loader.desc = "[predict acc: {:.3f}, auc: {:.3f}, prec: {:.3}, rec: {:.3}, f1_score: {:.3}"\
            .format(acc, roc_auc, prec, rec, f1)
        return acc, tpr, fpr, roc_auc


def draw_figs(train_loss, val_loss, title, dir_save):
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)
    plt.figure()
    plt.plot(train_loss, label=f'train_{title}')
    plt.plot(val_loss, label=f'val_{title}')
    plt.legend(loc='best')
    plt.ylabel(f'{title}', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title(f"train_{title} v.s. val_{title}")
    # plt.imshow()
    plt.savefig(f'./{dir_save}/{title}.png', dpi=500)


def main():
    data_path = "/home/hpc/users/wangqing/dataLiver/roi_total_equ/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 1000
    batch_size = 16
    lrf = 0.01
    best_acc = 0.8
    model_name = 'resnet18'
    data_transform = {
        "train": transforms.Compose([transforms.RandomRotation(10, expand=False),
                                    transforms.RandomResizedCrop(224, scale=(0.95, 1.05)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()]),
    }

    # # weights_pretrained = "/home/wq/wangqing/resnet18/weights3_resnet50/model-233.pth"
    # weights_pretrained = "/home/hpc/users/wangqing/mviClass/resnet18/weights_resnet18_/model-120.pth"
    # weights_pretrained = "/home/hpc/users/wangqing/mviClass/resnet18/weights_resnet50_/model-344.pth"
    # model = getResnet50()
    # model = getResnet18()
    model = getResNet101()
    model = torch.nn.DataParallel(model)
    # weight = torch.load(weights_pretrained, map_location='cpu')
    # model.load_state_dict(weight)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9, weight_decay=0.00001)
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tb_writer = SummaryWriter()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    weight_path = f'./weights_{model_name}_'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    for epoch in range(1, epochs+1):
        train_loader, val_loader = load_data(train_images_path, train_images_label, val_images_path, val_images_label,
                                             data_transform, batch_size)

        train_loss, train_acc = train_one_epoch(train_loader, epoch, model, criterion, optimizer, device)
        print("[train epoch {}] loss: {:.3f}, acc: {:.2f}".format(epoch,train_loss,train_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device, epoch)
        print("[val epoch {}] loss: {:.3f}, acc: {:.2f}".format(epoch, val_loss, val_acc))
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{weight_path}/model-{epoch}.pth")
    print("======================== best acc: ", best_acc)
    draw_figs(train_loss_list, val_loss_list, "loss", f"ep{epochs}bs{batch_size}_{model_name}")
    draw_figs(train_acc_list, val_acc_list, "accuracy", f"ep{epochs}bs{batch_size}_{model_name}")



def images_transforms():
    img_path = "/home/wq/dataLiver/roi_total/nn/122_clp_n_v.png"
    img_path1 = "/home/wq/dataLiver/roi_total/yy/112_clp_y_v.png"
    trans = transforms.Compose([transforms.RandomRotation(10, expand=False),
                                     transforms.RandomResizedCrop(224, scale=(0.95, 1.05)),
                                     transforms.RandomHorizontalFlip(p=0.5)])
    trans1 = transforms.transforms.RandomRotation(10, expand=False)
    trans2 = transforms.RandomResizedCrop(224, scale=(0.95, 1.05))
    trans3 = transforms.RandomHorizontalFlip(p=0.5)
    img = Image.open(img_path)
    data1 = trans1(img)
    data2 = trans2(img)
    data3 = trans3(img)
    data4 = trans(img)

    img1 = Image.open(img_path1)
    data11 = trans1(img1)
    data22 = trans2(img1)
    data33 = trans3(img1)
    data44 = trans(img1)

    plt.subplot(2, 5, 1), plt.imshow(img), plt.title("The original image(with MVI -)")
    plt.subplot(2, 5, 2), plt.imshow(data1), plt.title("Random rotation")
    plt.subplot(2, 5, 3), plt.imshow(data2), plt.title("Random resized crop")
    plt.subplot(2, 5, 4), plt.imshow(data3), plt.title("Random horizontal flip")
    plt.subplot(2, 5, 5), plt.imshow(data4), plt.title("Final transformation")

    plt.subplot(2, 5, 6), plt.imshow(img1), plt.title("The original image(with MVI +)")
    plt.subplot(2, 5, 7), plt.imshow(data11), plt.title("Random rotation")
    plt.subplot(2, 5, 8), plt.imshow(data22), plt.title("Random resized crop")
    plt.subplot(2, 5, 9), plt.imshow(data33), plt.title("Random resized crop")
    plt.subplot(2, 5, 10), plt.imshow(data44), plt.title("Final transformation")


    plt.show()



if __name__ == '__main__':

    # images_transforms()


    main()
    # weight = "/home/wq/wangqing/resnet18/weights3/model-408.pth"
    # weight1 = "/home/wq/wangqing/resnet18/weights2/model-97.pth"
    # weight2 = "/home/wq/wangqing/resnet18/weights/model-163.pth"
    # model = getResnet18()
    # acc1, tpr1, fpr1, auc1 = predict(model, weight)
    # acc2, tpr2, fpr2, auc2 = predict(model, weight1)
    # acc3, tpr3, fpr3, auc3 = predict(model, weight2)
    #
    # plt.figure()
    # plt.plot(fpr1, tpr1, label="acc: {:.2f}, auc: {:.2f}".format(acc1, auc1))
    # plt.plot(fpr2, tpr2, label="acc: {:.2f}, auc: {:.2f}".format(acc2, auc2))
    # plt.plot(fpr3, tpr3, label="acc: {:.2f}, auc: {:.2f}".format(acc3, auc2))
    # plt.legend(loc='best')
    # plt.savefig("roc_curve_on_val_data.png", dpi=500)
    # plt.show()
    #
    # weights ="/home/wq/wangqing/resnet18/weights_total_resnet50/model-1024.pth"
    # model = getResnet50()
    # predict(model, weights)
    # model = getResNet101()
    # acc1, tpr1, fpr1, auc1 = predict(model, weights)
    # plt.figure()
    # plt.plot(fpr1, tpr1, label="acc: {:.2f}, auc: {:.2f}".format(acc1, auc1))
    # plt.legend(loc='best')
    # plt.savefig("roc_curve_on_total_data.png", dpi=500)
    # data_path = "/home/wq/dataLiver/roi_resized/"
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    # new_data_path = "/home/wq/dataLiver/mvi_cropped/"
    # if not os.path.exists(new_data_path):
    #     os.makedirs(new_data_path)
    #
    # train_nn_path = os.path.join(new_data_path, 'train', 'nn')
    # train_yy_path = os.path.join(new_data_path, 'train', 'yy')
    # if not os.path.exists(train_nn_path):
    #     os.makedirs(train_nn_path)
    # if not os.path.exists(train_yy_path):
    #     os.makedirs(train_yy_path)
    #
    #
    # val_nn_path = os.path.join(new_data_path, 'val', 'nn')
    # val_yy_path = os.path.join(new_data_path, 'val', 'yy')
    # if not os.path.exists(val_nn_path):
    #     os.makedirs(val_nn_path)
    # if not os.path.exists(val_yy_path):
    #     os.makedirs(val_yy_path)
    #
    # for i in range(len(train_images_path)):
    #     if train_images_label[i] == 0 or train_images_label[i] == '0':
    #         shutil.copy(train_images_path[i], train_nn_path)
    #     elif train_images_label[i] == 1 or train_images_label[i] == '1':
    #         shutil.copy(train_images_path[i], train_yy_path)
    #
    # for i in range(len(val_images_path)):
    #     if val_images_label[i] == 0 or val_images_label[i] == '0':
    #         shutil.copy(val_images_path[i], val_nn_path)
    #     elif val_images_label[i] == 1 or val_images_label[i] == '1':
    #         shutil.copy(val_images_path[i], val_yy_path)

