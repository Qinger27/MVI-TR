import matplotlib.pyplot as plt
import numpy as np


def draw_box(x):
    x = np.array(x)  # 随机生成5行9列 [10, 100]之间的数
    # print(x)  # 打印数据
    plt.title("F1-score of different supervised models")
    plt.grid(True)  # 显示网格
    plt.boxplot(x, labels=['ResNet50', 'Transformer-based model', 'ResNet101', 'ResNet18'],
                patch_artist={'color': 'yellow'},
                boxprops={'color': 'black'},
                showmeans=True)  # 绘制箱线图
    plt.savefig("F1score_box.png")
    plt.show()  # 显示图片

