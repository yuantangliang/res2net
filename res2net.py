import mxnet as mx
from mxnet.gluon import nn
import math


class Bottle2neck(nn.HybridBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2D(in_channels=inplanes,  channels=width*scale, kernel_size=1, use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu1 = nn.Activation(activation='relu')
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2D(pool_size=3, strides=stride, padding=1)
        self.convs = []
        self.bns = []
        self.relus = []
        for i in range(self.nums):
          self.convs.append(nn.Conv2D(in_channels=width, channels=width, kernel_size=3, strides=stride, padding=1, use_bias=False))
          self.bns.append(nn.BatchNorm())
          self.relus.append(nn.Activation(activation="relu"))

        for conv in self.convs:
            self.register_child(conv)

        for bn in self.bns:
            self.register_child(bn)

        for relu in self.relus:
            self.register_child(relu)

        self.conv3 = nn.Conv2D(in_channels=width*scale, channels=planes * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = nn.BatchNorm()

        self.relu3 = nn.Activation(activation='relu')
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def hybrid_forward(self, F, x, *args, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        spx = F.split(out, num_outputs=self.nums+1, axis=1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relus[i](self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = F.concat(out, sp)
        if self.scale != 1 and self.stype=='normal':
          out = F.concat(out, spx[self.nums])
        elif self.scale != 1 and self.stype=='stage':
          out = F.concat(out, self.pool(spx[self.nums]))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.elemwise_add(out, residual)
        out = self.relu3(out)

        return out

class Res2Net(nn.HybridBlock):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        super(Res2Net, self).__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2D(channels=64,kernel_size=7,strides=2, padding=3,
                               use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu = nn.Activation('relu')
        self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], idx=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, idx=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, idx=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, idx=3)
        self.avgpool = nn.GlobalAvgPool2D()
        self.fc = nn.Dense(num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, idx=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix="down_"+str(idx))
            downsample.add(nn.Conv2D(in_channels=self.inplanes, channels=planes * block.expansion,
                      kernel_size=1, strides=stride, use_bias=False))
            downsample.add(nn.BatchNorm())

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        seq = nn.HybridSequential(prefix="nn_"+str(idx))
        for layer in layers:
            seq.add(layer)
        return seq

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def res2net50(**kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    return model

def res2net50_26w_4s(**kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    return model

def res2net101_26w_4s(**kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth = 26, scale = 4, **kwargs)
    return model

def res2net50_26w_6s(**kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 6, **kwargs)
    return model

def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 8, **kwargs)
    return model

def res2net50_48w_2s(**kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 48, scale = 2, **kwargs)
    return model

def res2net50_14w_8s(**kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 14, scale = 8, **kwargs)
    return model


def viz_model(model):
    node_attrs = {"fixedsize": "false"}
    plot = mx.viz.plot_network(model(mx.symbol.var("data"))[0],
                               save_format="pdf",
                               shape={"data":(1,3,240,240)},
                               node_attrs=node_attrs)
    plot.view()


if __name__ == '__main__':
    net = res2net50()
    data = mx.nd.ones((1,3,240,240))
    net.initialize()
    net(data)
    print(net)
    viz_model(net)
