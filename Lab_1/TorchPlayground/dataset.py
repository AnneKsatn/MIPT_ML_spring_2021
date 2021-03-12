#Генерация данных


from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import datasets

import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(processName)-10s %(name)s - %(levelname)s: %(message)s")

class Circles(Dataset):

    def __init__(self, n_samples, shuffle, noise, random_state=0, factor=.8):
        self.X, self.y = datasets.make_circles(n_samples=n_samples, shuffle=shuffle,
                                               noise=noise, random_state=random_state, factor=factor)

        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)

        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.show()


class Moons(Dataset):

    def __init__(self, n_samples, shuffle, noise, random_state=0):
        self.X, self.y = datasets.make_moons(n_samples=n_samples, shuffle=shuffle,
                                               noise=noise, random_state=random_state)

        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)

        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.show()





if __name__ == '__main__':

    logger = logging.getLogger("dataset")
    handler = logging.StreamHandler()

    file_handler = logging.FileHandler("logs.log")
    logger.addHandler(file_handler)

    circles = Circles(n_samples=5000, shuffle=True, noise=0.1, random_state=0, factor=0.8)
    print(circles.X)
    print(circles.y)

    print(len(circles))

    logger.info(f'element 0 {circles[0]}')
    logger.info(f'element 10 {circles[10]}')

    #circles.plot_data()

    train_dataset = Circles(n_samples=50, shuffle=True, noise=0.1, random_state=0, factor=0.5)
    test_dataset = Circles(n_samples=10, shuffle=True, noise=0.1, random_state=0, factor=0.5)


    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    for i, (x,y) in enumerate(train_dataloader):
        print("Batch: ", i)
        print(x, y)