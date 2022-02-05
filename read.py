# Read a json file and print it

import json
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision.models import resnet50


# # Dataloader for COCO styled dataset 
# class DUO(data.Dataset):
#     """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

#     Args:
#         root (string): Root directory where images are downloaded to.
#         annFile (string): Path to json annotation file.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.ToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """

#     def __init__(self, root, annFile, transform=None, target_transform=None):
#         from pycocotools.coco import COCO
#         self.root = root
#         self.coco = COCO(annFile)
#         self.ids = list(self.coco.imgs.keys())
#         self.transform = transform
#         self.target_transform = target_transform

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
#         """
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         target = coco.loadAnns(ann_ids)

#         path = coco.loadImgs(img_id)[0]['file_name']

#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target


#     def __len__(self):
#         return len(self.ids)

#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225]),
# ])

# data = DUO('/home/tejas/DUO/images/test/', '/home/tejas/DUO/annotations/instances_test.json',transform=transform)
# train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)


# class DETR(nn.Module):
#     """
#     Demo DETR implementation.

#     Demo implementation of DETR in minimal number of lines, with the
#     following differences wrt DETR in the paper:
#     * learned positional encoding (instead of sine)
#     * positional encoding is passed at input (instead of attention)
#     * fc bbox predictor (instead of MLP)
#     The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
#     Only batch size 1 supported.
#     """
#     def __init__(self, num_classes, hidden_dim=256, nheads=8,
#                  num_encoder_layers=6, num_decoder_layers=6):
#         super().__init__()

#         # create ResNet-50 backbone
#         self.backbone = resnet50()
#         del self.backbone.fc

#         # create conversion layer
#         self.conv = nn.Conv2d(2048, hidden_dim, 1)

#         # create a default PyTorch transformer
#         self.transformer = nn.Transformer(
#             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

#         # prediction heads, one extra class for predicting non-empty slots
#         # note that in baseline DETR linear_bbox layer is 3-layer MLP
#         self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
#         self.linear_bbox = nn.Linear(hidden_dim, 4)

#         # output positional encodings (object queries)
#         self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

#         # spatial positional encodings
#         # note that in baseline DETR we use sine positional encodings
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

#     def forward(self, inputs):
#         # propagate inputs through ResNet-50 up to avg-pool layer
#         x = self.backbone.conv1(inputs)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)

#         # convert from 2048 to 256 feature planes for the transformer
#         h = self.conv(x)

#         # construct positional encodings
#         H, W = h.shape[-2:]
#         pos = torch.cat([
#             self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
#             self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
#         ], dim=-1).flatten(0, 1).unsqueeze(1)

#         # propagate through the transformer
#         h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
#                              self.query_pos.unsqueeze(1)).transpose(0, 1)
        
#         # finally project transformer outputs to class labels and bounding boxes
#         return {'pred_logits': self.linear_class(h), 
#                 'pred_boxes': self.linear_bbox(h).sigmoid()}

# model = DETR(num_classes=4)
# model.train()

# # Train the model 
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# # Define Device 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Move model to device
# model = model.to(device)


# for i, sample in enumerate(train_loader):
#     print("HI")
#     image = sample['image']
#     annotation = sample['annotation']
#     category = sample['category']
#     print(category['name'])
#     print(annotation['bbox'])
#     print(annotation['segmentation'])

#     # 
#     image = image.numpy()
#     image = np.transpose(image, (1, 2, 0))
#     plt.imshow(image)
#     plt.show()
#     if i == 3:
#         break
#     print(image)


import csv 
filename="test.csv" 

file = open('/home/tejas/DUO/annotations/instances_test.json', 'r')
data=json.loads(file.read())

images: ['file_name', 'height', 'width', 'id']
annotations: ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id', 'ignore']
categories: ['name', 'id']

fields = ['image_id','width','height','bbox','source'] 
rows = []
for key,value in data.items():
    if(key == 'annotations'):
        for i in range(len(value)):
            # print(data['images'][value[i]['image_id']-1]['id'],end=",")
            # print(data['images'][value[i]['image_id']-1]['width'],end=",")
            # print(data['images'][value[i]['image_id']-1]['height'],end=",")
            # print(value[i]['bbox'],end=",")
            # print(data['categories'][value[i]['category_id']-1]['name'])
            rows.append([data['images'][value[i]['image_id']-1]['id'],data['images'][value[i]['image_id']-1]['width'],data['images'][value[i]['image_id']-1]['height'],value[i]['bbox'],data['categories'][value[i]['category_id']-1]['name']])

with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)

# rootdir ='/home/tejas/DUO/images/train' 

# for i in range(9):
#     img = mpimg.imread(rootdir + data['images'][i]['file_name'])
#     plt.subplot(331 + i)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(data['images'][i]['file_name'])

# plt.show()


# for i in range(9):
#     img = mpimg.imread(rootdir + data['images'][i]['file_name'])
#     plt.subplot(331 + i)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(data['images'][i]['file_name'])
#     for j in range(len(data['annotations'])):
#         #print(data['annotations'][j]['image_id'])
#         if data['annotations'][j]['image_id'] == (i+1):
#             # Plot a rectangle for a bounding box
#             anchor = data['annotations'][j]['bbox'][0:2]
#             width = data['annotations'][j]['bbox'][2]
#             height = data['annotations'][j]['bbox'][3]

#             plt.gca().add_patch(plt.Rectangle(anchor, width, height, fill=False, edgecolor='red', linewidth=2))
#             plt.gca().text(data['annotations'][j]['bbox'][0], data['annotations'][j]['bbox'][1], data['categories'][data['annotations'][j]['category_id']-1]['name'], bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
# plt.show()

