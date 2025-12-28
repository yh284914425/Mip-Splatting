import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""创建一个衡量学习感知图像块相似度 (LPIPS) 的标准。

    参数：
        net_type (str): 用于比较特征的网络类型：
                        'alex' | 'squeeze' | 'vgg'。默认值：'alex'。
        version (str): LPIPS 的版本。默认值：0.1。
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], '目前仅支持 v0.1'

        super(LPIPS, self).__init__()

        # 预训练网络
        self.net = get_network(net_type)

        # 线性层
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
