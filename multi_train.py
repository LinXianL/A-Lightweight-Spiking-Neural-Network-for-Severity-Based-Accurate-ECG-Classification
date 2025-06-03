import os
import time
import argparse
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import rnn
import matplotlib
import matplotlib.pyplot as plt
from spikingjelly.activation_based import monitor, neuron, encoding, functional, surrogate, layer
import datetime
import time
import torch
from MyDataset import*
from MyEncoder import*
from models import SharedSNN
from config import get_args

def main():
    
    args = get_args('multi_train')
    net = SharedSNN(tau=args.tau)
    if args.load_parm:
        checkpoint = torch.load('./output/multi_train/checkpoint_multi_best.pth')
        # 应用权重到当前网络
        net.load_state_dict(checkpoint['net'])
        print("成功加载预训练权重！")

    print(net)

    net.to(args.device)

    task_num_classes=[3,6,4]

    train_dataset_path_list = ['./DATASET/320dataset/dataset_1/train_data1(320,3).csv',
                                './DATASET/320dataset/dataset2/train_data2(320).csv',
                                './DATASET/320dataset/dataset3/train_data3(320).csv',
                                ]
    test_dataset_path_list = ['./DATASET/320dataset/dataset_1/test_data1(320,3).csv',
                                './DATASET/320dataset/dataset2/test_data2(320).csv',
                                './DATASET/320dataset/dataset3/test_data3(320).csv']
    train_datasets = [CSVDataset(filepath=train_dataset_path) for train_dataset_path in train_dataset_path_list]

    test_datasets = [CSVDataset(filepath=test_dataset_path) for test_dataset_path in test_dataset_path_list]


    
    train_data_loaders = [data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=False,
    ) for train_dataset in train_datasets]
    train_iters = [iter(dl) for dl in train_data_loaders]

    test_data_loaders = [data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=False,
    ) for test_dataset in test_datasets]

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    
    out_dir=args.out_dir

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    encoder = DeltaModulator(args.device) 
   
    for epoch in range(start_epoch, args.epochs):
        print(f'epoch ={epoch}')

        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = [0] * 3
        total_samples = [0] * 3

        for step in range(len(train_data_loaders[2])):  # 假设以任务1的长度为准
            optimizer.zero_grad()
            total_loss = 0

            for task_id in range(3):
                try:
                    ecg, label = next(train_iters[task_id])
                except StopIteration:
                    train_iters[task_id] = iter(train_data_loaders[task_id])
                    ecg, label = next(train_iters[task_id])

                ecg, label = ecg.to(args.device), label.to(args.device)
                label_onehot = F.one_hot(label, task_num_classes[task_id]).float()

                out_fr = 0.
                for t in range(args.T):
                    encoded_ecg = encoder(ecg)
                    encoded_ecg = data_convert2(encoded_ecg, args.device)
                    out_fr += net(encoded_ecg, task_id)
                out_fr /= args.T

                loss = F.mse_loss(out_fr, label_onehot)
                total_loss += loss  # ✅ 累加 loss，不在这里 backward

                train_loss += loss.item() * label.numel()
                train_acc[task_id] += (out_fr.argmax(1) == label).float().sum().item()
                total_samples[task_id] += label.size(0)

            total_loss.backward()  # ✅ 所有 loss 累加后，统一反向传播
            optimizer.step()
            functional.reset_net(net)

        # 日志输出（可调用你原有的 log_training_details 函数）
        for i in range(3):
            acc = train_acc[i] / total_samples[i]
            print(f"Task {i}: Train Acc = {acc:.4f}")
       

        net.eval()
        task_correct = [0] * len(test_data_loaders)
        task_total = [0] * len(test_data_loaders)
        class_correct = [ [0]*num_classes for num_classes in task_num_classes ]  # 每个任务每个类别的正确计数
        class_total = [ [0]*num_classes for num_classes in task_num_classes ]    # 每个任务每个类别的总样本数

        for task_id, test_loader in enumerate(test_data_loaders):
            for ecg, label in test_loader:
                ecg, label = ecg.to(args.device), label.to(args.device)

                out_fr = 0.
                for t in range(args.T):
                    encoded_ecg = encoder(ecg)
                    encoded_ecg = data_convert2(encoded_ecg, args.device)
                    out_fr += net(encoded_ecg, task_id)
                out_fr /= args.T

                pred = out_fr.argmax(1)
                task_correct[task_id] += (pred == label).float().sum().item()
                task_total[task_id] += label.size(0)
                
                # 统计每个类别的正确率
                for c in range(task_num_classes[task_id]):
                    class_mask = (label == c)
                    class_total[task_id][c] += class_mask.sum().item()
                    class_correct[task_id][c] += (pred[class_mask] == label[class_mask]).float().sum().item()

        test_acc = [0, 0, 0]
        for i in range(3):
            acc = task_correct[i] / task_total[i]
            print(f"\nTask {i} Overall Test Acc = {acc:.4f}")
            test_acc[i] = acc
            
            # 打印每个类别的准确率
            for c in range(task_num_classes[i]):
                class_acc = class_correct[i][c] / class_total[i][c] if class_total[i][c] > 0 else 0  # 避免除以0
                print(f"  Class {c} Acc = {class_acc:.4f} ({class_correct[i][c]}/{class_total[i][c]})")
            
            for c in range(task_num_classes[i]):
                class_acc = class_correct[i][c] / class_total[i][c] if class_total[i][c] > 0 else 0

        save_max = False
        if np.mean(test_acc) > max_test_acc:
            max_test_acc = np.mean(test_acc)
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_multi_best.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest_ALL.pth'))       

if __name__ == '__main__':
    main()
