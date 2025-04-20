import torch
from torch.utils.data import Dataset, DataLoader
import struct
from array import array
import numpy as np
from network import LeNet5


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None, include_only: list[int] = None):
        images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0  # [N, 1, 28, 28]
        labels = torch.tensor(labels, dtype=torch.long)

        if include_only is not None:
            include_only_tensor = torch.tensor(include_only, dtype=torch.long)
            mask = torch.isin(labels, include_only_tensor)
            images = images[mask]
            labels = labels[mask]

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):        
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())        

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())

        images = np.zeros((size, rows, cols), dtype=np.uint8)
        for i in range(size):
            start = i * rows * cols
            end = (i + 1) * rows * cols
            images[i] = np.array(image_data[start:end]).reshape(rows, cols)

        return images, labels

    def load_dataset(self, include_only: list[int] = None):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

       
        train_dataset = MNISTDataset(x_train, y_train, include_only=include_only)
        test_dataset = MNISTDataset(x_test, y_test, include_only=include_only)

        return train_dataset, test_dataset
    
if __name__ == "__main__":
    mnist_loader = MnistDataloader('train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
                                    't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    train_dataset, test_dataset = mnist_loader.load_dataset()

    x,y = mnist_loader.load_dataset([1])

    train_loader = DataLoader(x, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64)
    model = LeNet5()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    for input, label in train_loader:
        # criterion = ()
        # print(input.shape)
        # print(label.shape)

        model.train()

        pred = model(input)

        optimizer.zero_grad()

        loss = torch.nn.functional.cross_entropy(pred, label)

        # train_loss += loss.item()
        loss.backward()

        optimizer.step()

    