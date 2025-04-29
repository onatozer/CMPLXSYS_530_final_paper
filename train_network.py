import torch
from torch import nn
from network import LeNet5
from torch.utils.data import DataLoader
from dataset import MnistDataloader

def train_model(model: nn.Module, num_epochs = 5, even_odd: bool = False):
    mnist_loader = MnistDataloader('train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
                                    't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    train_dataset, test_dataset = mnist_loader.load_dataset()

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(num_epochs):
        train_loss = 0
        for input, label in train_loader:
            # criterion = ()
            # print(input.shape)
            # print(label.shape)
            
            if even_odd:
                label = torch.remainder(label,2)

            model.train()

            pred = model(input)

            optimizer.zero_grad()

            loss = torch.nn.functional.cross_entropy(pred, label)

            train_loss += loss.item()
            loss.backward()

            optimizer.step()
            
        print(f"EPOCH {epoch} average MSE Loss {train_loss/len(train_loader)}")

    model.save_model("model.pt")


if __name__ == "__main__":
    # model = LeNet5()

    # train_model(model)

    model = LeNet5(reduction_factor=8)
    # model.save_model("evenodd_untrained_8.pt")

    # train_model(model, num_epochs= 1, even_odd= True)
    # model.save_model("evenodd_one_epoch_8.pt")

    train_model(model, num_epochs= 5)
    model.save_model("fully_trained_8.pt")
    