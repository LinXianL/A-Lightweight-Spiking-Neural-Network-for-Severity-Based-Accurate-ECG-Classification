import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from MyEncoder import *
def train_one_epoch(net, data_loader, optimizer, encoder, device, T, scaler=None, num_classes=4):
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    for ecg, label in data_loader:
        optimizer.zero_grad()
        ecg = ecg.to(device)
        label = label.to(device)
        label_onehot = F.one_hot(label, num_classes).float()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out_fr = 0.
                for t in range(T):
                    encoded_ecg = encoder(ecg)
                    encoded_ecg = data_convert2(encoded_ecg, device)
                    out_fr += net(encoded_ecg)
                out_fr = out_fr / T
                loss = F.mse_loss(out_fr, label_onehot)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = 0.
            for t in range(T):
                encoded_ecg = encoder(ecg)
                encoded_ecg = data_convert2(encoded_ecg, device)
                out_fr += net(encoded_ecg)
            out_fr = out_fr / T
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

        preds = out_fr.argmax(1)
        for i in range(num_classes):
            class_mask = (label == i)
            class_correct[i] += ((preds == label) & class_mask).float().sum().item()
            class_total[i] += class_mask.float().sum().item()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (preds == label).float().sum().item()
        functional.reset_net(net)

    train_loss /= train_samples
    train_acc /= train_samples
    class_acc = [c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return train_loss, train_acc, class_acc

def evaluate(net, data_loader, encoder, device, T, num_classes=4):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for ecg, label in data_loader:
            ecg = ecg.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, num_classes).float()
            out_fr = 0.
            for t in range(T):
                encoded_ecg = encoder(ecg)
                encoded_ecg = data_convert2(encoded_ecg, device)
                out_fr += net(encoded_ecg)
            out_fr = out_fr / T
            loss = F.mse_loss(out_fr, label_onehot)
            preds = out_fr.argmax(1)
            for i in range(num_classes):
                class_mask = (label == i)
                class_correct[i] += ((preds == label) & class_mask).float().sum().item()
                class_total[i] += class_mask.float().sum().item()
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (preds == label).float().sum().item()
            functional.reset_net(net)
    test_loss /= test_samples
    test_acc /= test_samples
    class_acc = [c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return test_loss, test_acc, class_acc
