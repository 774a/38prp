import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models

device=torch.device("cuda:1"if torch.cuda.is_available() else"cpu")

class ResidualBlock(nn.Module):
    def __init__ (self,channels):
        super(ResidualBlock,self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)
    )
    def forward(self,x):
        return F.relu(self.conv(x)+x)

class ImfwNet(nn.Module):
    def __init__(self):
        super(ImfwNet,self).__init__()
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding=4),
            nn.Conv2d(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.unsample = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding=4),
        )
    def forward(self,x):
        x=self.downsample(x)
        x=self.res_blocks(x)
        x=self.unsample(x)
        return x
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
dataset=ImageFolder("/Disk1/yonglu/iCAN/Data/v-coco/coco/images",transform=data_transform)
data_loader=Data.DataLoader(dataset,batch_size=4,shuffle=True,num_workers=8,pin_memory=True)
vgg16=models.vgg16(pretrained=True)
vgg=vgg16.features.to(device).eval()
def load_image(img_path,shape=None):
    image = Image.open(img_path)
    size=image.size
    if shape is not None:
        size=shape
    in_transform = transforms.Compose(
        [transforms.Resize(size),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]
    )
    image=in_transform(image)[:3,:,:].unsqueeze(dim=0)
    return image
def im_convert(tensor):
    tensor=tensor.cpu()
    image=tensor.data.numpy().squeeze()
    image=image.transpose(1,2,0)
    image=image*np.array((0.229,0.224,0.225))+ np.array((0.485,0.456,0.406))
    image=image.clip(0,1)
    return image
style=load_image("/Disk2/xuliang/ava_object_box/label_results/style.jpg",shape=(256,256)).to(device)
#plt.figure()
#plt.imshow(im_convert(style))
#plt.show()

def gram_matrix(tensor):
    b,c,h,w=tensor.size()
    tensor=tensor.view(b,c,h*w)
    tensor_t=tensor.transpose(1,2)
    gram=tensor.bmm(tensor_t)/(c*h*w)
    return gram

def get_features(image,model,layers=None):
    if layers is None:
        layers={"3":"relu1_2",
                "8":"relu2_2",
                "15":"relu3_3",
                "22":"relu4_3"}
    features={}
    x=image
    for name,layer in model._modules.items():
        x=layer(x)
        if name in layers:
            features[layers[name]]=x
    return features
fwnet=ImfwNet().to(device)
style_layer={"3":"relu1_2",
                "8":"relu2_2",
                "15":"relu3_3",
                "22":"relu4_3"}
content_layer={"15":"relu3_3"}
style_features=get_features(style,vgg,layers=style_layer)
style_grams={layer:gram_matrix(style_features[layer]) for layer in style_features}

style_weight=1e5
content_weight=1
tv_weight=1e-5
optimizer=optim.Adam(fwnet.parameters(),lr=1e-3)
fwnet.train()
since=time.time()
print("shit")
for epoch in range(4):
    print("Epoch:{}".format(epoch+1))
    content_loss_all=[]
    style_loss_all=[]
    tv_loss_all=[]
    all_loss=[]
    for step,batch in enumerate(data_loader):
        optimizer.zero_grad()
        content_images=batch[0].to(device)
        transformed_images=fwnet(content_images)
        transformed_images=transformed_images.clamp(-2.1,2.7)
        content_features=get_features(content_images,vgg,layers=content_layer)
        transformed_features=get_features(transformed_images,vgg)
        content_loss=F.mse_loss(transformed_features["relu3_3"],content_features["relu3_3"])
        content_loss=content_weight*content_loss
        y=transformed_images
        tv_loss=(torch.sum(torch.abs(y[:,:,:,:-1]-y[:,:,:,1:]))+torch.sum(torch.abs(y[:,:,:-1,:]-y[:,:,1:,:])))
        tv_loss=tv_weight*tv_loss
        style_loss=0
        transformed_grams={layer:gram_matrix(transformed_features[layer])
                           for layer in transformed_features}
        for layer in style_grams:
            transformed_gram=transformed_grams[layer]
            style_gram=style_grams[layer]
            style_loss+=F.mse_loss(transformed_gram,style_gram.expand_as(transformed_gram))
        style_loss=style_weight*style_loss
        loss=style_loss+content_loss+tv_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        content_loss_all.append(content_loss.item())
        style_loss_all.append(style_loss.item())
        tv_loss_all.append(tv_loss.item())
        all_loss.append(loss.item())
        if step % 5000==0:
            print("step:{};content loss:{:.3f};style loss:{:.3f};tv loss:{:.3f}".format(step,content_loss.item(),style_loss.item(),tv_loss.item(),loss.item()))
            time_use=time.time()-since
            print("Train complete in {:.0f}m {:.0f}s".format(time_use//60,time_use%60))
            #plt.figure()
            im=transformed_images[1,...]
            #plt.imshow(im_convert(im))
            #plt.show()
torch.save(fwnet.state_dict(),"/Disk2/xuliang/ava_object_box/label_results/imfwnet_dict.pkl")