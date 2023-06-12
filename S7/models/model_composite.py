import torch
import torch.nn as nn
# import torch.nn.functional as F
from tqdm import tqdm

class Model_Composite(nn.Module):
    def __init__(self):
        super(Model_Composite, self).__init__()
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_losses = []
        self.test_losses = []

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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy Diff: {}\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            abs(round(self.test_accuracy[-1] - self.train_accuracy[-1], 4))
        ))


    def plot_accuracy(self):
        pass

    def plot_loss(self):
        pass
