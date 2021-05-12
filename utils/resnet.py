import torch
import torch.nn as nn

# Construct customized ResNet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
 
    def __init__(self, block, input_ch, layers_ch, layers, is_upsample=0, num_classes=1000):
        self.inplanes = layers_ch[0]
        self.is_upsample = is_upsample
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, layers_ch[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(layers_ch[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer1 = self._make_layer(block, layers_ch[0], layers[0])
        self.layer2 = self._make_layer(block, layers_ch[1], layers[1])
        self.layer3 = self._make_layer(block, layers_ch[2], layers[2])
        self.layer4 = self._make_layer(block, layers_ch[3], layers[3])
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        if self.is_upsample == 1:
            x = self.upsample(x)
        else:
            x = self.maxpool(x)
        x = self.layer3(x)
        if self.is_upsample == 1:
            x = self.upsample(x)
        else:
            x = self.maxpool(x)
        x = self.layer4(x)
 
        return x
