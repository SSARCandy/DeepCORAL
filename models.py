import torch
import torch.nn as nn
from torch.autograd import Function, Variable

CUDA = True if torch.cuda.is_available() else False


def feature_covariance_mat(n, d):
    ones_t = torch.ones(n).view(1, -1)
    if CUDA:
        ones_t = ones_t.cuda()

    tmp = ones_t.matmul(d)
    covariance_mat = (d.t().matmul(d) - (tmp.t().matmul(tmp) / n)) / (n - 1)
    return covariance_mat


def forbenius_norm(mat):
    return (mat**2).sum()**0.5


'''
MODELS
'''


class CORAL(Function):
    def forward(self, source, target):
        d = source.shape[1]
        ns, nt = source.shape[0], target.shape[0]
        cs = feature_covariance_mat(ns, source)
        ct = feature_covariance_mat(nt, target)

        self.saved = (source, target, cs, ct, ns, nt, d)

        res = forbenius_norm(cs - ct)**2/(4*d*d)
        res = torch.FloatTensor([res])

        return res if CUDA is False else res.cuda()

    def backward(self, grad_output):
        source, target, cs, ct, ns, nt, d = self.saved
        ones_s_t = torch.ones(ns).view(1, -1)
        ones_t_t = torch.ones(nt).view(1, -1)
        if CUDA:
            ones_s_t = ones_s_t.cuda()
            ones_t_t = ones_t_t.cuda()

        s_gradient = (source.t() - (ones_s_t.matmul(source).t().matmul(ones_s_t)/ns)).t().matmul(cs - ct) / (d*d*(ns - 1))
        t_gradient = (target.t() - (ones_t_t.matmul(target).t().matmul(ones_t_t)/nt)).t().matmul(cs - ct) / (d*d*(nt - 1))
        t_gradient = -t_gradient

        return s_gradient*grad_output, t_gradient*grad_output


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.sharedNet = AlexNet()
        self.source_fc = nn.Linear(4096, num_classes)
        self.target_fc = nn.Linear(4096, num_classes)

        # initialize according to CORAL paper experiment
        self.source_fc.weight.data.normal_(0, 0.005)
        self.target_fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.sharedNet(source)
        source = self.source_fc(source)

        target = self.sharedNet(target)
        target = self.source_fc(target)
        return source, target


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
