from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os 
import glob
import csv
import torch
import cv2
import numpy as np
from PIL import Image
import random

class MyCustomDataset(Dataset):
    def __init__(self,root,transforms=None,k=0,mode="train"):
        super(MyCustomDataset, self).__init__()
        self.root = root
        #self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        
        self.transforms = transforms

        self.images, self.labels = self.load_csv('images.csv')
        
        if mode == "train":
          self.images = self.images[:int((k*0.1) * len(self.images))] + self.images[int((k+1)*0.1* len(self.images)):]
          self.labels = self.labels[:int((k*0.1) * len(self.labels))] + self.labels[int((k+1)*0.1* len(self.labels)):]
          
        elif mode == "test":
          self.images = self.images[int((k*0.1) * len(self.images)):int((k+1)*0.1* len(self.images))]
          self.labels = self.labels[int((k*0.1) * len(self.labels)):int((k+1)*0.1* len(self.labels))]  
        else:
          pass 
        
    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        label = torch.tensor(label)
        f = open(img,"r")
        line = f.readline()
        line = line[:-1]
        x=[]
        y=[]
        while line:
         line = f.readline()
         line = line[:-1]
         l = line.split("\t")
         l = line.split(" ")
         #l = line.split("\t")
         if  len(l)>1 and l[1]!='' and l[0]!='':
            x.append(float(l[0]))
            y.append(float(l[1]))

        y_max = max(y)
        y_min = min(y)
        for index in range(0,len(y)):
          y[index] = (y[index]-y_min)/(y_max - y_min)*2 -1
        y = torch.tensor(y, dtype=torch.float)
        y = y.unsqueeze(dim=0)
        return (y, label)
 
    def __len__(self):
        return len(self.images) 

    def load_csv(self, filename):

            if not os.path.exists(os.path.join(self.root, filename)): 
              images = []
              for name in self.name2label.keys():   

                images += glob.glob(os.path.join(self.root, name, '*.txt'))

              random.shuffle(images)
              with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    
                    label = self.name2label[name]

                    writer.writerow([img, label])
                print('writen into csv file:', filename)

            # read from csv file
            images, labels = [], []
            with open(os.path.join(self.root, filename), "r") as f:
               reader = csv.reader(f)
               for row in reader:
                 img, label = row
                 label = int(label)
                 images.append(img)
                 labels.append(label)
                 assert len(images) == len(labels)
               return images, labels

