import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        bias = kwargs.get('bias', True)
        self.use_device = kwargs.get('device', 'cpu')
        self.train_acc = []
        self.train_losses = []
        self.test_acc = []
        self.test_losses = []

        # Model architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=bias)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=bias)
        self.fc1 = nn.Linear(4096, 50, bias=bias)
        self.fc2 = nn.Linear(50, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def get_correct_pred_cnt(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def model_train(self, train_loader, optimizer, criterion):
        """
        Trains the neural network based on the train_loader and computes the training loss based on the criterion.
        """
        self.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.use_device), target.to(self.use_device)
            optimizer.zero_grad()

            # Predict
            pred = self(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += self.get_correct_pred_cnt(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))
        return self.train_acc[-1], self.train_losses[-1]

    def model_test(self, test_loader, criterion):
        """
        Evaluates the neural network based on the test_loader and computes the testing loss based on the criterion.
        """
        self.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.use_device), target.to(self.use_device)

                output = self(data)
                test_loss += criterion(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                correct += self.get_correct_pred_cnt(output, target)

        test_loss /= len(test_loader.dataset)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)
        return self.test_acc[-1], self.test_losses[-1]
