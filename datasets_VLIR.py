import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', test_set_size=-1):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.train_files_A = sorted(glob.glob(os.path.join(root, 'TV*.*')))
        self.train_files_B = sorted(glob.glob(os.path.join(root, 'TY*.*')))

        # if test_set_size >= 1:
        #    self.test_files_A = self.train_files_A[-test_set_size:]
        #    self.test_files_B = self.train_files_B[-test_set_size:]

        #    self.train_files_A = self.train_files_A[:-test_set_size]
        #    self.train_files_B = self.train_files_B[:-test_set_size]
        # decrease the size of training set to overfit the model

        if test_set_size >= 1:
            self.test_files_A = self.train_files_A[80:90]
            self.test_files_B = self.train_files_B[80:90]
            self.train_files_A = self.train_files_A[:80]
            self.train_files_B = self.train_files_B[:80]

    def __getitem__(self, index):
        return self.getImages(index, self.train_files_A, self.train_files_B)

    def getImages(self, index, srcA, srcB):
        item_A = Image.open(srcA[index % len(srcA)])

        if self.unaligned:
            item_B = Image.open(srcB[random.randint(0, len(srcB) - 1)])
        else:
            item_B = Image.open(srcB[index % len(srcB)])

        return {'A': self.transform(item_A.convert('RGB')), 'B': self.transform(item_B.convert('RGB'))}


    def __len__(self):
        return max(len(self.train_files_A), len(self.train_files_B))

    def getTestPair(self, index):
        return self.getImages(index, self.test_files_A, self.test_files_B)
