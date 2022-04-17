import os
import pandas as pd
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Function
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torchvision import models
import torch.optim as optim

cuda = True
cudnn.benchmark = True

model_name = "efficientnet_b4"

LR = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 224 #227
# FT_OUT_SIZE = 512
N_EPOCH = 100

# dataset_root = '/content/drive/MyDrive/4B/SYDE-522/data'
# output_root = '/content/drive/MyDrive/4B/SYDE-522/submission/04032022_resnet50_aug'
dataset_root = './data'
output_root = './submission/04092022_efficientnetb4_aug_entropy2'
source_dataset_name = 'train_set'
target_dataset_name = 'test_set'

source_image_root = os.path.join(dataset_root, source_dataset_name)
target_image_root = os.path.join(dataset_root, target_dataset_name)

train_label_list = os.path.join(dataset_root, 'train_labels.csv')

os.makedirs(output_root, exist_ok=True)


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list=None, transform=None, train=False):
        self.root = data_root
        self.transform = transform

        # we only pass data_list if it's training set
        if data_list is not None:
            df = pd.read_csv(data_list)
            self.img_paths_temp = df['dir'].to_list()
            
            self.img_paths = []
            self.img_labels = []
            self.domain_labels = []

            if 'label2' in df.columns:
                self.img_labels_temp = df['label2'].to_list()
            else: 
                self.img_labels_temp = ['0' for i in range(len(self.img_paths))]

            # if 'label1' in df.columns:
            #     self.domain_labels_temp = df['label1'].to_list()
            # else: 
            self.domain_labels_temp = [0 for i in range(len(self.img_paths_temp))]

            # if train:
            #     for i in range(len(self.img_paths_temp)):
            #         if int(self.domain_labels_temp[i]) == 2:
            #             # if int(self.domain_labels_temp[i]) == 0:
            #             #     self.img_paths.append(self.img_paths_temp[i])
            #             #     self.domain_labels.append(0)
            #             #     self.img_labels.append(self.img_labels_temp[i])
            #             # elif int(self.domain_labels_temp[i]) == 2:
            #             self.img_paths.append(self.img_paths_temp[i])
            #             self.domain_labels.append(0)
            #             self.img_labels.append(self.img_labels_temp[i])
            #             # elif int(self.domain_labels_temp[i]) == 1:
            #             #     self.img_paths.append(self.img_paths_temp[i])
            #             #     self.domain_labels.append(1)
            #             #     self.img_labels.append(self.img_labels_temp[i])

            self.img_paths = self.img_paths_temp
            self.domain_labels = self.domain_labels_temp
            self.img_labels = self.img_labels_temp

        else:
            # Walk through test folder - we don't need labels
            self.img_paths = [f for root,dirs,files in os.walk(data_root) for f in files if f.endswith('.png')]
            self.img_labels = ['0' for i in range(len(self.img_paths))]
            self.domain_labels = ['0' for i in range(len(self.img_paths))]

        self.n_data = len(self.img_paths)

    def __getitem__(self, item):
        img_paths, labels, domain_labels = self.img_paths[item%self.n_data], self.img_labels[item%self.n_data], self.domain_labels[item%self.n_data]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:

            if isinstance(self.transform, list):
                tform = self.transform[int(domain_labels)]
            else:
                tform = self.transform

            imgs = tform(imgs)
            labels = int(labels)
            domain_labels = int(domain_labels)

        return imgs, labels, domain_labels, img_paths

    def __len__(self):
        return self.n_data


# Preprocess data
def preprocess_multiple_fn(mus, stds):
    tforms = []

    for i in range(len(mus)):
        tforms.append(preprocess_fn(mu=mus[i], std=stds[i]))
    
    return tforms

def preprocess_fn(mu=(0.6399, 0.6076, 0.5603), std=(0.3065, 0.3082, 0.3353), aug=False):
  if aug:
    img_transform = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, saturation=0.5),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=std) 
    ])
  else:
    img_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=std) 
    ])

  return img_transform

def prep_dataloader(image_root, label_list=None, img_transform=None, 
                    drop_last=False, shuffle=True, train=True):
    dataset = GetLoader(
        data_root=image_root,
        data_list=label_list,
        transform=img_transform,
        train=train
    )

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last)
    
    return dataset, dataloader



# if False, then we are feature extracting
def set_parameter_requires_grad(model, finetune):
    for param in model.parameters():
        param.requires_grad = finetune
        


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class CNNModel(nn.Module):

    def __init__(self, model_name="resnet18"):
        super(CNNModel, self).__init__()
        self.ft_out_size = 0

        if model_name == "resnet18":
            self.feature = models.resnet18(pretrained=True) 
            self.feature.fc = nn.Identity()
            self.ft_out_size = 512
        if model_name == "resnet50":
            self.feature = models.resnet50(pretrained=True) 
            self.feature.fc = nn.Identity()
            self.ft_out_size = 2048
        elif model_name == "vgg16":
            self.feature = models.vgg16_bn(pretrained=True) 
            self.feature.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # original is (7,7)
            self.feature.classifier = nn.Identity()
            self.ft_out_size = 512
        elif model_name == "efficientnet_b4":
            self.feature = models.efficientnet_b4(pretrained=True) 
            self.feature.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # original is (7,7)
            self.feature.classifier = nn.Identity()
            self.ft_out_size = 1792
        elif model_name == "efficientnet_b0":
            self.feature = models.efficientnet_b0(pretrained=True) 
            self.feature.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # original is (7,7)
            self.feature.classifier = nn.Identity()
            self.ft_out_size = 1280
        else:
            # Need some default model?
            self.feature = nn.Sequential()
            self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
            self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
            self.feature.add_module('f_pool1', nn.MaxPool2d(2))
            self.feature.add_module('f_relu1', nn.ReLU(True))
            self.feature.add_module('f_conv2', nn.Conv2d(64, 128, kernel_size=3))
            self.feature.add_module('f_bn2', nn.BatchNorm2d(128))
            self.feature.add_module('f_drop1', nn.Dropout2d())
            self.feature.add_module('f_pool2', nn.MaxPool2d(2))
            self.feature.add_module('f_relu2', nn.ReLU(True))
            self.feature.add_module('f_conv3', nn.Conv2d(128, 256, kernel_size=3))
            self.feature.add_module('f_bn3', nn.BatchNorm2d(256))
            self.feature.add_module('f_drop3', nn.Dropout2d())
            self.feature.add_module('f_pool3', nn.MaxPool2d(2))
            self.feature.add_module('f_relu4', nn.ReLU(True))
            self.feature.add_module('f_conv4', nn.Conv2d(256, 256, kernel_size=3))
            self.feature.add_module('f_bn4', nn.BatchNorm2d(256))
            self.feature.add_module('f_drop4', nn.Dropout2d())
            self.feature.add_module('f_pool4', nn.MaxPool2d(2))
            self.feature.add_module('f_relu4', nn.ReLU(True))
            self.feature.add_module('f_conv5', nn.Conv2d(256, 512, kernel_size=3))
            self.feature.add_module('f_bn5', nn.BatchNorm2d(512))
            self.feature.add_module('f_drop5', nn.Dropout2d())
            self.feature.add_module('f_pool5', nn.MaxPool2d(2))
            self.feature.add_module('f_relu5', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.ft_out_size, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 7))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.ft_out_size, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        


    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
        feature = self.feature(input_data)
        feature = feature.view(-1, self.ft_out_size)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output



def test(net, epoch):

    # load data
    # img_transform_source = preprocess_multiple_fn(mus=[(0.5085, 0.4832, 0.4396),
    #                                                     (0.5550, 0.5085, 0.4579),
    #                                                    (0.8077, 0.7829, 0.7358)], 
    #                                               stds=[(0.1780, 0.1779, 0.1907),
    #                                                     (0.1880, 0.1917, 0.2060),
    #                                                     (0.2239, 0.2283, 0.2437)])
    
    # img_transform_target = preprocess_fn(mu=(0.9566, 0.9566, 0.9566), std=(0.1752, 0.1752, 0.1752))

    # img_transform_source = preprocess_fn()
    img_transform_target = preprocess_fn(mu=(0.9566, 0.9566, 0.9566), std=(0.1752, 0.1752, 0.1752))


    # dataset_source, dataloader_source = prep_dataloader(
    #     image_root=os.path.join(source_image_root, 'train_set'),
    #     label_list=train_label_list,
    #     img_transform=img_transform_source,
    #     shuffle=False,
    #     train=True
    # )

    dataset_target, dataloader_target = prep_dataloader(
        image_root=os.path.join(target_image_root, 'test_set'),
        img_transform=img_transform_target,
        shuffle=False
    )

    net.eval()

    if cuda:
        net = net.cuda()

    # train_pths, train_preds = inference(net, dataloader_source, cuda=cuda, alpha=alpha)
    # train_results = pd.DataFrame({'id': train_pths, 'label': train_preds})
    # train_results_pth = os.path.join(output_root, '%s_train_epoch%s.csv' % (datetime.now().strftime("%m%d%Y"), epoch))
    # train_results.to_csv(train_results_pth, index=False)

    test_pths, test_preds = inference(net, dataloader_target, cuda=cuda, alpha=alpha)
    test_results = pd.DataFrame({'id': test_pths, 'label': test_preds})
    test_results_pth = os.path.join(output_root, '%s_test_epoch%s.csv' % (datetime.now().strftime("%m%d%Y"), epoch))
    test_results.to_csv(test_results_pth, index=False)

    # print('epoch: %d, accuracy of the train dataset: %f' % (epoch, compare(train_label_list, train_results_pth, train=True)))

    # Secret
    test_label_list = os.path.join(dataset_root, 'dummy_test_labels.csv')
    print('epoch: %d, accuracy of the test dataset: %f' % (epoch, compare(test_label_list, test_results_pth)))

def inference(net, dataloader, cuda=True, alpha=0.0):
    preds = []
    pths = []
    for input_img, _,_, img_paths in dataloader: 

        if cuda:
            input_img = input_img.cuda()

        class_output, _ = net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        pths = pths + list(img_paths)
        preds = preds + list(pred.squeeze(1).cpu().numpy())
    return pths, preds

def compare(true_labels, predicted_labels, train=False):
    combined_df = pd.read_csv(true_labels)
    predicted_df = pd.read_csv(predicted_labels)
    combined_df['label'] = combined_df['dir'].map(predicted_df.set_index('id')['label'])

    # predicted_df = pd.read_csv(true_labels)
    # combined_df = pd.read_csv(predicted_labels)
    # combined_df['label2'] = combined_df['id'].map(predicted_df.set_index('dir')['label2'])


    true_labels = np.array(combined_df['label2'].to_list())
    pred_labels = np.array(combined_df['label'].to_list())

    return np.sum(true_labels == pred_labels) / len(true_labels)



manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
# img_transform_source = preprocess_multiple_fn(mus=[(0.5085, 0.4832, 0.4396),
#                                                     (0.5550, 0.5085, 0.4579),
#                                                     (0.8077, 0.7829, 0.7358)], 
#                                                 stds=[(0.1780, 0.1779, 0.1907),
#                                                     (0.1880, 0.1917, 0.2060),
#                                                     (0.2239, 0.2283, 0.2437)])
# img_transform_target = preprocess_fn(mu=(0.9566, 0.9566, 0.9566), std=(0.1752, 0.1752, 0.1752))

img_transform_source = preprocess_fn(aug=True)
img_transform_target = preprocess_fn(mu=(0.9566, 0.9566, 0.9566), std=(0.1752, 0.1752, 0.1752), aug=True)


dataset_source, dataloader_source = prep_dataloader(
    image_root=os.path.join(source_image_root, 'train_set'), # TODO should we unnest
    label_list=train_label_list,
    img_transform=img_transform_source,
    train=True
)

dataset_target, dataloader_target = prep_dataloader(
    image_root=os.path.join(target_image_root, 'test_set'),
    img_transform=img_transform_target
)

# load model
my_net = CNNModel(model_name=model_name)

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=LR)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

set_parameter_requires_grad(my_net, True)
# set_parameter_requires_grad(my_net.feature, False)

# training
for epoch in range(N_EPOCH):

    len_dataloader = max(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / N_EPOCH / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        alpha = 1. / (1. + np.exp(-10 * (epoch-N_EPOCH//2))) 

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label, s_domain_label, _ = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.LongTensor(batch_size)

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            s_domain_label = s_domain_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        domain_label.resize_as_(s_domain_label).copy_(s_domain_label)

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        if i == len(dataloader_target):
            data_target_iter = iter(dataloader_target)
        data_target = data_target_iter.next()
        t_img, _, _, _ = data_target

        batch_size = len(t_img) # TODO: why?

        input_img = torch.FloatTensor(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        domain_label = torch.ones(batch_size) 
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        entropy_loss = torch.mean(class_output)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label + 0.01*alpha*entropy_loss
        err.backward()
        optimizer.step()

        i += 1

        print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
            % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

    # torch.save(my_net, 'output/model_epoch_{}.pth'.format(epoch))
    test(my_net, epoch)
    my_net.train()

print('done')