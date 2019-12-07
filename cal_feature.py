import numpy as np
import math
import os
import torch
import torchvision
from PIL import Image
# from tqdm import tqdm  # 进度条用于一些提示工作
import copy
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


# -----------------------------------FC layers
class ArcFC(nn.Module):
    r"""
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output_layer sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 s=30.0,
                 m=0.50,
                 easy_margin=False):
        """
        ArcMargin
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print('=> in dim: %d, out dim: %d' % (self.in_features, self.out_features))

        self.s = s
        self.m = m

        # 根据输入输出dim确定初始化权重
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # L2 normalize and calculate cosine
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # phi: cos(θ+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # ----- whether easy margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)  # device='cuda'
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output_layer)

        return output


# RepNet网络结构
class RepNet(torch.nn.Module):
    def __init__(self,
                 out_ids,
                 out_attribs):
        """
        Network definition
        :param out_ids:
        :param out_attribs:
        """
        super(RepNet, self).__init__()

        self.out_ids, self.out_attribs = out_ids, out_attribs
        print('=> out_ids: %d, out_attribs: %d' % (self.out_ids, self.out_attribs))

        # Conv1
        self.conv1_1 = torch.nn.Conv2d(in_channels=3,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (0)
        self.conv1_2 = torch.nn.ReLU(inplace=True)  # (1)
        self.conv1_3 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (2)
        self.conv1_4 = torch.nn.ReLU(inplace=True)  # (3)
        self.conv1_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)  # (4)

        self.conv1 = torch.nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.conv1_3,
            self.conv1_4,
            self.conv1_5
        )

        # Conv2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=128,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (5)
        self.conv2_2 = torch.nn.ReLU(inplace=True)  # (6)
        self.conv2_3 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=128,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (7)
        self.conv2_4 = torch.nn.ReLU(inplace=True)  # (8)
        self.conv2_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)  # (9)

        self.conv2 = torch.nn.Sequential(
            self.conv2_1,
            self.conv2_2,
            self.conv2_3,
            self.conv2_4,
            self.conv2_5
        )

        # Conv3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (10)
        self.conv3_2 = torch.nn.ReLU(inplace=True)  # (11)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (12)
        self.conv3_4 = torch.nn.ReLU(inplace=True)  # (13)
        self.conv3_5 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))  # (14)
        self.conv3_6 = torch.nn.ReLU(inplace=True)  # (15)
        self.conv3_7 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)  # (16)

        self.conv3 = torch.nn.Sequential(
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv3_4,
            self.conv3_5,
            self.conv3_6,
            self.conv3_7
        )

        # Conv4_1
        self.conv4_1_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (17)
        self.conv4_1_2 = torch.nn.ReLU(inplace=True)  # (18)
        self.conv4_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (19)
        self.conv4_1_4 = torch.nn.ReLU(inplace=True)  # (20)
        self.conv4_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (21)
        self.conv4_1_6 = torch.nn.ReLU(inplace=True)  # (22)
        self.conv4_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (23)

        self.conv4_1 = torch.nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2,
            self.conv4_1_3,
            self.conv4_1_4,
            self.conv4_1_5,
            self.conv4_1_6,
            self.conv4_1_7
        )

        # Conv4_2
        self.conv4_2_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (17)
        self.conv4_2_2 = torch.nn.ReLU(inplace=True)  # (18)
        self.conv4_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (19)
        self.conv4_2_4 = torch.nn.ReLU(inplace=True)  # (20)
        self.conv4_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (21)
        self.conv4_2_6 = torch.nn.ReLU(inplace=True)  # (22)
        self.conv4_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (23)

        self.conv4_2 = torch.nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2,
            self.conv4_2_3,
            self.conv4_2_4,
            self.conv4_2_5,
            self.conv4_2_6,
            self.conv4_2_7
        )

        # Conv5_1
        self.conv5_1_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (24)
        self.conv5_1_2 = torch.nn.ReLU(inplace=True)  # (25)
        self.conv5_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (26)
        self.conv5_1_4 = torch.nn.ReLU(inplace=True)  # (27)
        self.conv5_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (28)
        self.conv5_1_6 = torch.nn.ReLU(inplace=True)  # (29)
        self.conv5_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (30)

        self.conv5_1 = torch.nn.Sequential(
            self.conv5_1_1,
            self.conv5_1_2,
            self.conv5_1_3,
            self.conv5_1_4,
            self.conv5_1_5,
            self.conv5_1_6,
            self.conv5_1_7
        )

        # Conv5_2
        self.conv5_2_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (24)
        self.conv5_2_2 = torch.nn.ReLU(inplace=True)  # (25)
        self.conv5_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (26)
        self.conv5_2_4 = torch.nn.ReLU(inplace=True)  # (27)
        self.conv5_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))  # (28)
        self.conv5_2_6 = torch.nn.ReLU(inplace=True)  # (29)
        self.conv5_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)  # (30)

        self.conv5_2 = torch.nn.Sequential(
            self.conv5_2_1,
            self.conv5_2_2,
            self.conv5_2_3,
            self.conv5_2_4,
            self.conv5_2_5,
            self.conv5_2_6,
            self.conv5_2_7
        )

        # FC6_1
        self.FC6_1_1 = torch.nn.Linear(in_features=25088,
                                       out_features=4096,
                                       bias=True)  # (0)
        self.FC6_1_2 = torch.nn.ReLU(inplace=True)  # (1)
        self.FC6_1_3 = torch.nn.Dropout(p=0.5)  # (2)
        self.FC6_1_4 = torch.nn.Linear(in_features=4096,
                                       out_features=4096,
                                       bias=True)  # (3)
        self.FC6_1_5 = torch.nn.ReLU(inplace=True)  # (4)
        self.FC6_1_6 = torch.nn.Dropout(p=0.5)  # (5)

        self.FC6_1 = torch.nn.Sequential(
            self.FC6_1_1,
            self.FC6_1_2,
            self.FC6_1_3,
            self.FC6_1_4,
            self.FC6_1_5,
            self.FC6_1_6
        )

        # FC6_2
        self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)

        self.FC6_2 = torch.nn.Sequential(
            self.FC6_2_1,
            self.FC6_2_2,
            self.FC6_2_3,
            self.FC6_2_4,
            self.FC6_2_5,
            self.FC6_2_6
        )

        # FC7_1
        self.FC7_1 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)  # (6): 4096, 1000

        # FC7_2
        self.FC7_2 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)  # (6): 4096, 1000

        # ------------------------------ extra layers: FC8 and FC9
        self.FC_8 = torch.nn.Linear(in_features=2000,  # 2048
                                    out_features=1024)  # 1024

        # attribute classifiers: out_attribs to be decided
        self.attrib_classifier = torch.nn.Linear(in_features=1000,
                                                 out_features=out_attribs)

        # Arc FC layer for branch_2 and branch_3
        self.arc_fc_br2 = ArcFC(in_features=1000,
                                out_features=out_ids,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)
        self.arc_fc_br3 = ArcFC(in_features=1024,
                                out_features=out_ids,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)

        # construct branches
        self.shared_layers = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.branch_1_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_1,
            self.conv5_1,
        )

        self.branch_1_fc = torch.nn.Sequential(
            self.FC6_1,
            self.FC7_1
        )

        self.branch_1 = torch.nn.Sequential(
            self.branch_1_feats,
            self.branch_1_fc
        )

        self.branch_2_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_2,
            self.conv5_2
        )

        self.branch_2_fc = torch.nn.Sequential(
            self.FC6_2,
            self.FC7_2
        )

        self.branch_2 = torch.nn.Sequential(
            self.branch_2_feats,
            self.branch_2_fc
        )

    def forward(self,
                X,
                branch,
                label=None):
        """
        :param X:
        :param branch:
        :param label:
        :return:
        """
        # batch size
        N = X.size(0)

        if branch == 1:  # train attributes classification
            X = self.branch_1_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_1_fc(X)

            assert X.size() == (N, 1000)

            X = self.attrib_classifier(X)

            assert X.size() == (N, self.out_attribs)

            return X

        elif branch == 2:  # get vehicle fine-grained feature
            if label is None:
                print('=> label is None.')
                return None
            X = self.branch_2_feats(X)

            # reshape and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_2_fc(X)

            assert X.size() == (N, 1000)

            X = self.arc_fc_br2.forward(input=X, label=label)

            assert X.size() == (N, self.out_ids)

            return X

        elif branch == 3:  # overall: combine branch_1 and branch_2
            if label is None:
                print('=> label is None.')
                return None
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)

            # reshape and connect to FC layers
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)

            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)

            # feature fusion
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)

            assert fusion_feats.size() == (N, 2000)

            # connect to FC8: output 1024 dim feature vector
            X = self.FC_8(fusion_feats)

            # connect to classifier: arc_fc_br3
            X = self.arc_fc_br3.forward(input=X, label=label)

            assert X.size() == (N, self.out_ids)

            return X

        elif branch == 4:  # test pre-trained weights
            # extract features
            X = self.branch_1_feats(X)

            # flatten and connect to FC layers
            X = X.view(N, -1)
            X = self.branch_1_fc(X)

            assert X.size() == (N, 1000)

            return X

        elif branch == 5:
            # 前向运算提取用于Vehicle ID的特征向量
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)

            # reshape and connect to FC layers
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)

            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)

            # feature fusion
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)

            assert fusion_feats.size() == (N, 2000)

            # connect to FC8: output 1024 dim feature vector
            X = self.FC_8(fusion_feats)

            assert X.size() == (N, 1024)

            return X

        else:
            print('=> invalid branch')
            return None


# 特征图计算
# 获取每张测试图片对应的特征向量
def gen_feature_map(resume, imgs_path, batch_size=16):
    """
    根据图相对生成每张图象的特征向量, 映射: img_name => img_feature vector
    :param resume:
    :param imgs_path:
    :return:
    """
    # 设置启动设备，使用一块GPU
    # 启动模型，RepNet
    net = RepNet(out_ids=10086,
                 out_attribs=257).to(device)
    print('=> Mix difference network:\n', net)

    # 从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            # 加载模型
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 图像数据变换，定义一个图像数据读取进来应该如何进行装换，图像的装换操作，一共也是RGB三个通道， 图片数据同时也要进行变化为224*224的
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load model, image and forward
    data, features = None, None

    # 这块开始就是提取特征， 注意这里就开始for循环了
    for i, img_path in enumerate(imgs_path):
        # load image
        img = Image.open(img_path)

        # tuen to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # image data transformations
        img = transforms(img)
        img = img.view(1, 3, 224, 224)

        if data is None:
            data = img
        else:
            # 将两个张量拼接在一起，因为这个batchsize是16， 其实我们可以设置batchsize为1
            data = torch.cat((data, img), dim=0)

        # 如果这个data满足了这个条件， 满足了batch的条件，将图片直接进行加载了，直接跑一遍网络，然后开始提取特征
        if data.shape[0] % batch_size == 0 or i == len(imgs_path) - 1:

            # collect a batch of image data
            data = data.to(device)

            output = net.forward(X=data,
                                 branch=5,
                                 label=None)

            # 这个前向传播直接得到的是特征图。

            batch_features = output.data.cpu().numpy()
            # 最后将feature存到features
            if features is None:
                features = batch_features
            else:
                features = np.vstack((features, batch_features))

            # clear a batch of images
            data = None

    # generate feature map
    feature_map = {}
    for i, img_path in enumerate(imgs_path):
        feature_map[img_path] = features[i]

    print('=> feature map size: %d' % (len(feature_map)))
    # 最后就能得到特征图
    return feature_map


# 这个函数是用来计算相似度，和文本的相似度比较类似，都是计算cos的值
# 夹角越小，两个值就越接近，那么cos值也就越接近于1
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# 统计阈值和准确率: Car Match数据集，就是按照车辆匹配去整个数据集
# 我们的目的是将单个批次的数据取出来，然后选择合适的阈值， 其中统计阈值这部分也比较重要
# 按照题目中给出的最佳阈值是0.295
def test_car_match_data(resume,
                        pair_set_txt,
                        img_root,
                        batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = line.strip().split()

            imgs_path.append(img_root + '/' + line[0])
            imgs_path.append(img_root + '/' + line[1])

            pairs.append(line)

    # 最后这个数据类似于这种
    pairs = []
    imgs_path = []

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # 计算特征向量字典，计算特征向量字典这一步是关键
    # 这步开始计算特征图
    # **** 这边要进行改进，其实获得单张的图片路径，或者是两张图片的这个特征向量就行。

    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    # 计算所有pair的sim， sim就是得到的所谓的值
    # **** 这步是对得到的特征值进行比较 ******
    # 其中sims这一步表示的是
    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_root + '/' + pair[0]
        img_path_2 = img_root + '/' + pair[1]
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    # ***** 计算准确度，这个用于训练阶段 *****
    # 因为是用于训练阶段的代码，所以暂时还是用不到
    # acc, th = cal_accuracy(sims, labels)
    # print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    # return acc, th


# 得到两张图片的特征图
def get_feature_two(img1, img2):
    # 0000200.jpg 0000201.jpg
    img_root = "E:/python_code/xjtu/data/VehicleID_V1.0/image"
    resume = "E:/python_code/xjtu\data/VehicleID_V1.0/epoch_14.pth"
    batch_size = 1

    img_path1 = img_root + '/' + img1
    img_path2 = img_root + '/' + img2

    imgs_path = [img_path1, img_path2]
    # 计算特征向量字典，计算特征向量字典这一步是关键
    # 这步开始计算特征图
    # **** 这边要进行改进，其实获得单张的图片路径，或者是两张图片的这个特征向量就行。

    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    # print(feature_map)
    print("特征计算完成")
    feature1 = feature_map[img_path1]
    feature2 = feature_map[img_path2]
    print(img1 + "的特征为：")
    print(feature1)
    print(img2 + "的特征为：")
    print(feature2)

    sim = cosin_metric(feature1, feature2)
    print("计算得到的相似度为")
    print(sim)



    # 按理说这个应该可以得到一个字典一样的数据

    # 计算所有pair的sim， sim就是得到的所谓的值
    # **** 这步是对得到的特征值进行比较 ******
    # 其中sims这一步表示的是
    # sims, labels = [], []
    # for pair in pairs:
    #     img_path_1 = img_root + '/' + pair[0]
    #     img_path_2 = img_root + '/' + pair[1]
    #     sim = cosin_metric(feature_map[img_path_1],
    #                        feature_map[img_path_2])
    #     label = int(pair[2])
    #     sims.append(sim)
    #     labels.append(label)


def read_data(pair_set_txt, img_root):
    # 这个文件是用空格符来进行分割的字符串， 包括图片1的路径，图片2的路径，和图片的label
    # 因为这里主要是提取特征，所以只用到了图片路径
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = line.strip().split()

            imgs_path.append(img_root + '/' + line[0])
            imgs_path.append(img_root + '/' + line[1])

            pairs.append(line)

    # 最后这个数据类似于这种
    pairs = []
    # line就是一行行的数据，
    # 然后这个imgs_path就是图片路径数据
    imgs_path = []

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()
    # 但是这部排序的目的是？


def math_test():
    # 前面的这个计算就是求解平方和
    a = np.array([1, 2, 3, 4])
    result = np.linalg.norm(a)
    print(result)

    print("数学计算")
    b = 1 + 4 + 9 + 16
    result2 = math.sqrt(b)
    print(result2)

    print("dot计算")
    # 这个就是分开乘按照累加
    a1 = np.array([1, 2, 3, 4])
    print(np.dot(a, a1))


if __name__ == '__main__':
    # 作者给出的阈值0.295
    print("特征值计算")
    # 写一个得到特征图的函数
    # 同一id的车辆
    # get_feature_two("0000200.jpg", "0000201.jpg")  # 0.7554931

    # 同一辆车 同一个场景
    # get_feature_two("0000200.jpg", "0000200.jpg")  # 1.0

    # 不同的车
    # get_feature_two("0000200.jpg", "0000189.jpg")  # 0.20846894

    # 颜色相同，车不一样， 白色且都有天窗 # 0.32770383
    get_feature_two("0000030.jpg", "0000730.jpg")




