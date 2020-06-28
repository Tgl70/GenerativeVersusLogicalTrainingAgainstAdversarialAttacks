from PIL import Image
import numpy as np
import torch
import cv2
import time


class MyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, dtype, train, transform):
        self.dataset = dataset
        self.dtype = dtype
        self.train = train
        self.transform = transform
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        if dataset == 'mnist' or dataset == 'fashion_mnist' or dataset == 'cifar10'  or dataset == 'gtsrb':
            n_classes = 10
        
        if dtype == 'datasetA' or dtype == 'datasetB' or dtype == 'datasetC':
            data = np.load(f"datasets/{dataset}/datasetA/X_train.npy")
            [self.X_train.append(d) for d in data]

            data = np.load(f"datasets/{dataset}/datasetA/y_train.npy")
            [self.y_train.append(torch.tensor(int(d))) for d in data]

            data = np.load(f"datasets/{dataset}/datasetA/X_test.npy")
            [self.X_test.append(d) for d in data]

            data = np.load(f"datasets/{dataset}/datasetA/y_test.npy")
            [self.y_test.append(torch.tensor(int(d))) for d in data]

        if dtype == 'datasetB':
            for i in range(n_classes):
                data = np.load(f"datasets/{dataset}/datasetB/{i}_images.npy")
                # Bringing the pixels in [0, 1] from [-1, 1]
                data = (data + 1) / 2.0
                data = data.astype(np.float64)
                for j in range(len(data)):
                    self.X_train.append(data[j])
                    self.y_train.append(torch.tensor(i))

        if dtype == 'datasetC':
            data = np.load(f"datasets/{dataset}/datasetC/X_train.npy")
            [self.X_train.append(d) for d in data]   

            data = np.load(f"datasets/{dataset}/datasetC/y_train.npy")
            [self.y_train.append(torch.tensor(int(d))) for d in data]

        if dtype == 'testAdversarial':
            for i in range(n_classes):
                data = np.load(f"datasets/{dataset}/testAdversarial/{i}_images.npy")
                # Bringing the pixels in [0, 1] from [-1, 1]
                data = (data + 1) / 2.0
                data = data.astype(np.float64)
                for j in range(len(data)):
                    self.X_test.append(data[j])
                    self.y_test.append(torch.tensor(i))

    def __getitem__(self, index):
        if self.train:
            img = self.X_train[index]
            label = self.y_train[index]
        else:
            img = self.X_test[index]
            label = self.y_test[index]

        if self.dataset == 'cifar10':
            img = Image.fromarray(img, 'RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)
