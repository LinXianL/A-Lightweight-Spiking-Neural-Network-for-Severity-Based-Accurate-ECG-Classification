import argparse
import sys
import os
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp, device
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import monitor, neuron, encoding, functional, surrogate, layer
from MyEncoder import *
from MyDataset import *
from config import get_args
from models import SNN_3
from trainer import train_one_epoch,evaluate
def main():

    args = get_args('class3')

    model = SNN_3(2.0)
    pretrained_ckpt_path = './output/multi_train/checkpoint_multi_best.pth'
    # 加载预训练参数
    checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
    pretrained_dict = checkpoint['net']
    new_state_dict = {
        'layer.1.weight': pretrained_dict['shared_layer.1.weight'],
        'layer.3.weight': pretrained_dict['task_heads.2.0.weight'],
        'layer.5.weight': pretrained_dict['task_heads.2.2.weight'],
    }
    model.load_state_dict(new_state_dict, strict=False)

    # 冻结第一层参数
    for param in model.layer[1].parameters():
        param.requires_grad = False
    print("layer[1].weight是否可训练:", model.layer[1].weight.requires_grad)

    model.to(args.device)
    print(model)


    train_dataset_path = "./DATASET/320dataset/dataset3/train_data3(320).csv"
    test_dataset_path = "./DATASET/320dataset/dataset3/test_data3(320).csv"
    train_dataset = CSVDataset(filepath=train_dataset_path)
    test_dataset = CSVDataset(filepath=test_dataset_path)

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        num_workers=args.j,
        pin_memory=False,
    )

    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        num_workers=args.j,
        pin_memory=False,
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    encoder = DeltaModulator(device=args.device)

    start_epoch = 0
    max_test_acc = -1
    best_last_train_class_acc = 0
    best_last_test_class_acc = 0 
    for epoch in range(start_epoch, args.epochs):
       #训练
        train_loss, train_acc, train_class_acc = train_one_epoch(model, train_data_loader, optimizer, encoder, args.device, args.T, scaler=scaler, num_classes=4)
        # 测试
        test_loss, test_acc, test_class_acc = evaluate(model, test_data_loader, encoder, args.device, args.T, num_classes=4)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True
        # 记录并打印历史最佳最后一个类别的正确率
        if epoch == start_epoch:
            best_last_train_class_acc = train_class_acc[-1] if len(train_class_acc) > 0 else 0
            best_last_test_class_acc = test_class_acc[-1] if len(test_class_acc) > 0 else 0
        else:
            if len(train_class_acc) > 0 and train_class_acc[-1] > best_last_train_class_acc:
                best_last_train_class_acc = train_class_acc[-1]
            if len(test_class_acc) > 0 and test_class_acc[-1] > best_last_test_class_acc:
                best_last_test_class_acc = test_class_acc[-1]

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            pass
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max1.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        print(args)
        print(out_dir)
        print(f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        print(f'train_class_acc={train_class_acc}')
        print(f'test_class_acc={test_class_acc}')
        print(f'best_last_train_class_acc={best_last_train_class_acc:.4f}')
        print(f'best_last_test_class_acc={best_last_test_class_acc:.4f}')
        
if __name__ == '__main__':
    main()
