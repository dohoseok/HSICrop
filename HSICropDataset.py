import torch
import numpy as np
import os
import pickle as pk


class HSICropData(torch.utils.data.Dataset):
    def __init__(self, image_set, root, **hyperparam):
        self.patch_size = hyperparam['patch_size']
        self.root = root
        self.expand_dims = hyperparam['expand_dims']
        self.center_pixel = hyperparam['center_pixel']

        self.images = np.load(os.path.join(root, f'{image_set}_image.npy'))
        self.labels = np.load(os.path.join(root, f'{image_set}_label.npy'))

        if self.center_pixel == True and self.patch_size > 1:
            self.addMirror()
            self.data_size = 128
        else:
            self.data_size = 129 - self.patch_size

        print("image shape", self.images.shape)

        assert (self.images.shape[0] == self.labels.shape[0])


    def __len__(self):
        return self.images.shape[0] * self.data_size * self.data_size


    def __getitem__(self, index):
        fileindex = index // (self.data_size * self.data_size)
        loc_index = index - fileindex * self.data_size * self.data_size
        x = loc_index // self.data_size
        y = loc_index - x * self.data_size

        label = self.labels[fileindex]
        if self.center_pixel:
            label = label[x, y]
        else:
            label = label[x:x+self.patch_size, y:y+self.patch_size]

        img = self.images[fileindex]
        if self.patch_size == 1:
            img_data = img[:, x, y]
        else:
            img_data = img[:, x:x+self.patch_size, y:y+self.patch_size]
        if self.expand_dims is not None:
            img_data = np.expand_dims(img_data, axis=self.expand_dims)
        return torch.tensor(img_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


    def addMirror(self):
        dx = self.patch_size // 2
        num_images, bands, h, w = self.images.shape
        if dx != 0:
            mirror = np.zeros((num_images, bands, h + 2 * dx, w + 2 * dx))
            mirror[:, :, dx:-dx, dx:-dx] = self.images
            for i in range(dx):
                mirror[:, :, :, i] = mirror[:, :, :, 2 * dx - i]
                mirror[:, :, i, :] = mirror[:, :, 2 * dx - i, :]
                mirror[:, :, :, -i - 1] = mirror[:, :, :, -(2 * dx - i) - 1]
                mirror[:, :, -i - 1, :] = mirror[:, :, -(2 * dx - i) - 1, :]
            self.images = mirror