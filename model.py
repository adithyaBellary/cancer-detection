#%%
import torch
import torchvision.models as models
from torch import torch, nn
import numpy as np


#%%
class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SubBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3),
                               stride=self.stride,
                               padding=1,
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3),
                               stride=1,
                               padding=1,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _id = x
        output = self.conv1(x) #here
        output = self.batch_norm1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.batch_norm2(output)

        if self.downsample is not None:
            # print('downsample:', self.downsample)
            _id = self.downsample(x)

        #THE RESENT PART
        output += _id
        output = self.relu(output)

        return output


#%%
class Cancer_model(nn.Module):
    def __init__(self, block, path, layer_sizes=[2,3,4,5], residual = True):
        super(Cancer_model, self).__init__()
        self.k_size_1 = 50
        self.k_size_2 = 30
        self.residual = residual
        self.layer_sizes = layer_sizes
        self.block = block
        self.in_planes = 64

        self.path = path



        #we want to get the output to have three 
        
        
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=64,
                               kernel_size=(7,7),
                               stride=2,
                               padding=3)


        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3),
                                     stride=2,
                                     padding=1)

        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2),
                                      stride=2,
                                      padding=1)

        # numfeatures is C from a tensot of size (N, C, H, W) - basically the number of channels
        # default eps = 1e-5
        # default momentum = 0.1 
        num_feat = 64
        epsilon = 1e-5
        mom = 0.1       
        self.batch_norm = nn.BatchNorm2d(num_features=num_feat,
                                         eps=epsilon, 
                                         momentum=mom )

        # self.mod_list_1 = nn.ModuleList([ nn.Conv2d(in_channels=,
        #                                             out_channels=,
        #                                             kernel_size=,
        #                                             stride=)   for i in range(3)])                          
        self.relu = nn.ReLU(inplace=True)
        #try using leaky RELU instead of real RELU
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # self.mod_list = nn.ModuleList([self.conv1, self.conv2, self.max_pool1])
        # self.nn_seq = nn.Sequential(*[self.conv1, self.conv2, self.max_pool1])

        self.subBlock1 = self.make_subBlock(num_ch=64,
                                            num_blocks=layer_sizes[0],
                                            stride=1)
        self.subBlock2 = self.make_subBlock(num_ch=128,
                                            num_blocks=layer_sizes[1],
                                            stride=2)
        self.subBlock3 = self.make_subBlock(num_ch=256,
                                            num_blocks=layer_sizes[2],
                                            stride=2)
        self.subBlock4 = self.make_subBlock(num_ch=512,
                                            num_blocks=layer_sizes[3],
                                            stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_val = 250
        self.fc1 = nn.Linear(in_features=512,
                             out_features=self.mid_val)

        self.fc2 = nn.Linear(in_features=self.mid_val,
                             out_features=2)


    def make_subBlock(self, num_ch, num_blocks, stride):
        #take care of adding stuff and batch norm
        if (stride != 1) or (self.in_planes != num_ch):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_planes,
                          out_channels=num_ch,
                          kernel_size=(1,1),
                          stride=stride),
                nn.BatchNorm2d(num_ch)
            )
        else:
            downsample = None
        # print('setting num inplanes', self.in_planes)
        block_list = []
        block_list.append(self.block(in_channels=self.in_planes, 
                                     out_channels=num_ch, 
                                     stride=stride, 
                                     downsample=downsample))
        
        self.in_planes = num_ch
        for _ in range(1, num_blocks):
            block_list.append(self.block(in_channels=self.in_planes, 
                                         out_channels=num_ch))
        
        return nn.Sequential(*block_list)

    def forward(self, x):
        #input size is 500, 1000

        #base model off of resnet
        #use stride > 1 to downsample

        #1 input channel
        # print('shape of x:', x.size())

        # print(x)
        x = self.conv1(x)
        # print('size after the first conv:', x.size())
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        
        x = self.subBlock1(x)
        # print('shape of x after the first block:', x.size())
        x = self.subBlock2(x)
        x = self.subBlock3(x)
        x = self.subBlock4(x)

        # print('shape of x ', x.size())
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), -1)
        # print('shape of x after:', x.size())
        
        x = self.fc1(x)

        x = self.fc2(x)

        return x

#%%
#initialize the cancer model
def get_model(device, path = None):
    model = Cancer_model(block=SubBlock, 
                         layer_sizes=[2,3,4,5],
                         residual=True,
                         path=path).double().to(device)

    return model

# def get_empty_model(device):
#     model = Cancer_model(*args, **kwards).double().to(device)
#     return model
