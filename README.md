# Description
This repository contains deep learning model architectures built using pytorch.

---

# Usage

The code snippet below imports required classes and methods.

```
from utils import device, build_loader
from model import Net
```

The code snippet below creates a train and test data loader.
```
train_data = # your training dataset
test_data = # your training dataset
batch_size = 512
train_loader, test_loader = build_loader(batch_size, train_data, test_data)
```

The code snippet below creates a model. `model_train` trains the model on train_loader and `model_test` tests the model on the test_loader.
```
model = Net().to(device)
for epoch in range(1, num_epochs+1):
    model.model_train(train_loader, optimizer, criterion)
    model.model_test(test_loader, criterion)
```

To visualize the model's performance, execute the below script.
```
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(model.train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(model.train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(model.test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(model.test_acc)
axs[1, 1].set_title("Test Accuracy")
```