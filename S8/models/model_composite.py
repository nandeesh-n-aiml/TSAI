# Library imports
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class Model_Composite(nn.Module):
    def __init__(self):
        super(Model_Composite, self).__init__()
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_losses = []
        self.test_losses = []
        self.norm_type = 'bn'
        self.n_groups = 2

    def model_train(self, device, train_loader, criterion, optimizer):
        self.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # init
            optimizer.zero_grad()
            
            # prediction
            y_pred = self(data)
            
            # calculate loss
            loss = criterion(y_pred, target)
            train_loss += loss.item()

            # backpropagation
            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        
        self.train_accuracy.append(100*correct/processed)
        self.train_losses.append(train_loss)

    def model_test(self, device, test_loader, criterion):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # calculate loss
        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)
        
        self.test_accuracy.append(100. * correct / len(test_loader.dataset))
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy Diff: {}\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            abs(round(self.test_accuracy[-1] - self.train_accuracy[-1], 4))
        ))

    def get_norm(self, n_channels):
        if self.norm_type == 'bn':
            return nn.BatchNorm2d(n_channels)
        elif self.norm_type == 'ln':
            return nn.GroupNorm(1, n_channels)
        elif self.norm_type == 'gn':
            return nn.GroupNorm(self.n_groups, n_channels)

    def plot_accuracy(self):
        epochs = list(range(1, len(self.train_accuracy) + 1))
        plt.plot(epochs, self.train_accuracy, label='Train accuracy')
        plt.plot(epochs, self.test_accuracy, label='Test accuracy')
        plt.legend()

    def plot_loss(self):
        epochs = list(range(1, len(self.train_accuracy) + 1))
        plt.plot(epochs, self.train_losses, label='Train loss')
        plt.plot(epochs, self.test_losses, label='Test loss')
        plt.legend()

    def get_incorrect_pred(self, device, test_loader, top_n=10):
        with torch.no_grad():
            data, target = next(iter(test_loader))
            data, target = data.to(device), target.to(device)
            output = self(data)
            pred = output.argmax(dim=1)
            compare = pred.eq(target)
            incorrect_indx = torch.where((compare == False), 1, 0).nonzero()
            top_n_pred = incorrect_indx[:top_n].squeeze()
            return data[top_n_pred], target[top_n_pred], pred[top_n_pred]
