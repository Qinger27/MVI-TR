import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn
import sys
sys.path.append("../resnet18")
from trainResnet18 import load_data, read_split_data, train_one_epoch, evaluate, predict, draw_figs
from vit_model import vit_base_patch16_224 as create_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img


def draw_lr(list, title="learning_rate"):
    plt.figure()
    plt.plot(list, label='lr')
    plt.ylabel(f'{title}', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title("learning rate")
    plt.savefig(f'{title}.png', dpi=500)


def main():
    # data_path = "/home/hpc/users/wangqing/dataLiver/roi_total/"
    data_path = "/home/wq/dataLiver/roi_total/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 500
    batch_size = 112
    lrf = 0.01
    best_acc = 0.937
    model_name = 'vit_768_112'
    data_transform = {
        "train": transforms.Compose([transforms.RandomRotation(10, expand=False),
                                     transforms.RandomResizedCrop(224, scale=(0.95, 1.05)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.GaussianBlur(kernel_size=5),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()]),
    }
    # weight2 = "/home/wq/wangqing/vit/weights_vit_1024_64_128/model-735.pth"
    # weight2 = "/home/wq/wangqing/vit/weights_vit_1024_64_64/model-1285.pth"
    weight2 = "/home/wq/wangqing/vit/weights4_total/model-2025.pth"
    weight_path = f'./weights_{model_name}_{batch_size}'
    model = create_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(weight2, map_location=device))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001)
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tb_writer = SummaryWriter()

    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []
    lr_list = []
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    for epoch in range(1+569, 1+569 + epochs):

        train_loader, val_loader = load_data(train_images_path, train_images_label, val_images_path, val_images_label,
                                             data_transform, batch_size)

        train_loss, train_acc = train_one_epoch(train_loader, epoch, model, criterion, optimizer, device)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device, epoch)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        lr_list.append(optimizer.param_groups[0]["lr"])

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./{weight_path}/model-{epoch}.pth")

    # np.save(f"{model_name}_train_loss", train_loss_lst)
    # np.save(f"{model_name}_train_acc", train_acc_lst)
    # np.save(f"{model_name}_val_loss", val_loss_lst)
    # np.save(f"{model_name}_val_loss", val_loss_lst)

    draw_figs(train_loss_lst, val_loss_lst, "loss", f"epochs{epochs}bs{batch_size}_{model_name}")
    draw_figs(train_acc_lst, val_acc_lst, "accuracy",  f"epochs{epochs}bs{batch_size}_{model_name}")
    draw_lr(lr_list)


def cam_main(model, weight_path, img_path, img_name, cam_path, model_name):
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))

    target_layers = [model.blocks[-4].norm1]
    data_transform = transforms.Compose([transforms.ToTensor()])

    assert os.path.exists(img_path), f"file: {img_path} does not exist..."

    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.xlabel("Pixels in x axis")
    plt.ylabel("Pixels in y axis")

    save_path = os.path.join(cam_path, f"{img_name}_{model_name}.png")
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':
    # main()
    # weight = "/home/hpc/users/wangqing/mviClass/vit/weights2_total/model-195.pth"
    # weight1 = "/home/hpc/users/wangqing/mviClass/vit/weights_total/model-363.pth"
    # weight2 = "/home/hpc/users/wangqing/mviClass/vit/weights1_total/model-97.pth"
    # model = create_model(num_classes=2)
    # acc1, tpr1, fpr1, auc1 = predict(model, weight)
    # acc2, tpr2, fpr2, auc2 = predict(model, weight1)
    # acc3, tpr3, fpr3, auc3 = predict(model, weight2)
    # plt.figure()
    # plt.plot(fpr1, tpr1, label="acc: {:.2f}, auc: {:.2f}".format(acc1, auc1))
    # plt.plot(fpr2, tpr2, label="acc: {:.2f}, auc: {:.2f}".format(acc2, auc2))
    # plt.plot(fpr3, tpr3, label="acc: {:.2f}, auc: {:.2f}".format(acc3, auc3))
    # plt.legend(loc='best')
    # plt.savefig("roc_curve_on_total_data.png", dpi=500)
    # plt.show()
    # # #
    print("000000000000000000000000000000")
    cam_path = './cam_figs_vit_v3'
    if not os.path.exists(cam_path):
        os.makedirs(cam_path)
    for img1 in os.listdir("/home/wq/dataLiver/roi_total/yy/"):
        img_path = os.path.join("/home/wq/dataLiver/roi_total/yy/", img1)
        img_name = img_path.split('/')[-1].split('.')[0]
        # print(img_name)
        weight = "/home/wq/wangqing/vit/weights4_total/model-2025.pth"
        model = create_model(num_classes=2)

        cam_main(model, weight, img_path, img_name, cam_path=cam_path, model_name="vit")
