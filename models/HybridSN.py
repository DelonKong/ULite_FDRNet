import torch
import torch.nn as nn
import torch.nn.functional as F


class overall(nn.Module):
    # 定义各个层的部分
    def __init__(self, rate, band, Classes, patch_size):
        super(overall, self).__init__()
        self.S = patch_size
        self.L = band

        # self.conv_block = nn.Sequential()
        ## convolutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        self.relu3 = nn.ReLU()

        # 不懂 inputX经过三重3d卷积的大小
        inputX = self.get2Dinput()
        inputConv4 = inputX.shape[1] * inputX.shape[2]
        # conv4 （24*24=576, 19, 19），64个 3x3 的卷积核 ==>（（64, 17, 17）
        self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3))

        # # self-attention
        # self.sa1 = nn.Conv2d(64, 64 // rate, kernel_size=1)
        # self.sa2 = nn.Conv2d(64 // rate, 64, kernel_size=1)

        # 全连接层（256个节点） # 64 * 17 * 17 = 18496
        tmp = 64 * (patch_size-8) * (patch_size-8)
        self.dense1 = nn.Linear(tmp, 256)
        # 全连接层（128个节点）
        self.dense2 = nn.Linear(256, 128)
        # 最终输出层(16个节点)
        self.dense3 = nn.Linear(128, Classes)

        # 让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        # 参考: https://blog.csdn.net/yangfengling1023/article/details/82911306
        # self.drop = nn.Dropout(p = 0.4)
        # 改成0.43试试
        self.drop = nn.Dropout(p=0.43)
        self.soft = nn.Softmax(dim=1)
        pass

    # 辅助函数，没怎么懂，求经历过三重卷积后二维的一个大小
    def get2Dinput(self):
        # torch.no_grad(): 做运算，但不计入梯度记录
        with torch.no_grad():
            x = torch.zeros((1, 1, self.L, self.S, self.S))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x
        pass

    # 必须重载的部分，X代表输入
    def forward(self, x):
        # F在上文有定义torch.nn.functional，是已定义好的一组名称
        # out = self.relu1(self.conv1(x))
        # out = self.relu2(self.conv2(out))
        # out = self.relu3(self.conv3(out))
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = F.relu(self.conv4(out))

        # # Squeeze 第三维卷成1了
        # weight = F.avg_pool2d(out, out.size(2))  # 参数为输入，kernel
        # # 参考: https://blog.csdn.net/qq_21210467/article/details/81415300
        # # 参考: https://blog.csdn.net/u013066730/article/details/102553073
        #
        # # Excitation: sa（压缩到16分之一）--Relu--fc（激到之前维度）--Sigmoid（保证输出为0至1之间）
        # weight = F.relu(self.sa1(weight))
        # weight = F.sigmoid(self.sa2(weight))
        # out = out * weight

        # flatten: 变为 18496 维的向量，
        out = out.view(out.size(0), -1)

        out = F.relu(self.dense1(out))
        out = self.drop(out)
        out = F.relu(self.dense2(out))
        out = self.drop(out)
        out = self.dense3(out)

        # 添加此语句后出现LOSS不下降的情况，参考：https://www.e-learn.cn/topic/3733809
        # 原因是CrossEntropyLoss()=softmax+负对数损失（已经包含了softmax)。如果多写一次softmax，则结果会发生错误
        # out = self.soft(out)
        # out = F.log_softmax(out)

        return out


def HybridSN(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = overall(16, band=204, Classes=16, patch_size=patch_size)

    elif dataset == 'pu':
        model = overall(16, band=103, Classes=9, patch_size=patch_size)

    elif dataset == 'whulk':
        model = overall(16, band=270, Classes=9, patch_size=patch_size)

    elif dataset == 'hrl':
        model = overall(16, band=176, Classes=14, patch_size=patch_size)

    elif dataset == 'IP':
        model = overall(16, band=200, Classes=16, patch_size=patch_size)

    elif dataset == 'whuhc':
        model = overall(16, band=274, Classes=16, patch_size=patch_size)

    return model


if __name__ == "__main__":
    t = torch.randn(size=(64, 1, 200, 7, 7))
    print("input shape:", t.shape)
    net = HybridSN(dataset='IP', patch_size=7)
    print("output shape:", net(t).shape)

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
