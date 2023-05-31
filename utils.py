# Importing libraries
import torch
from tqdm import tqdm

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(batch_size, train_data, test_data):
    """ """
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    return train_loader, test_loader


def get_correct_pred_cnt(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += get_correct_pred_cnt(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )

    train_acc = 100 * correct / processed
    train_loss = train_loss / len(train_loader)
    return train_acc, train_loss


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss

            correct += get_correct_pred_cnt(output, target)

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    test_acc = 100.0 * correct / len(test_loader.dataset)
    return test_acc, test_loss
