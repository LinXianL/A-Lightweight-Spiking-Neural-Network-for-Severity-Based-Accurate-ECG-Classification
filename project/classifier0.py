import argparse
import sys
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import monitor, neuron, encoding, functional, surrogate, layer
from MyEncoder import *
from MyLog import *
from MyDataset import *
from config import get_args

class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(283, 100, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(100, 15, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)



def main():
    args = get_args()
    print(args)

    net = SNN(tau=args.tau)

    print(net)

    net.to(args.device)

    for name, param in net.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data}")
        print(f"Parameter shape: {param.size()}")

    train_dataset_path ="/disk2/dongfh/spikingjelly_linxl/spikingjelly_yucz/projectxxx/DATASET/dataset_15/train_data15(320).csv"
    test_dataset_path = '/disk2/dongfh/spikingjelly_linxl/spikingjelly_yucz/projectxxx/DATASET/dataset_15/test_data15(320).csv'
    train_dataset = CSVDataset(filepath=train_dataset_path)
    # print(train_dataset)
    test_dataset = CSVDataset(filepath=test_dataset_path)

    
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        num_workers=args.j,
        pin_memory=False,

    )
    # print(train_data_loader.dataset)
    
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
    
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    out_dir = os.path.join(args.out_dir, f'15_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_{current_time}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    delta=args.delta
    print(f'delta={delta}!!!')

    encoder = DeltaModulator(delta=delta,device=args.device)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for img, label in train_data_loader:

            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            label_onehot = F.one_hot(label, 15).float()
            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        encoded_img = data_convert2(encoded_img,args.device)
                        out_fr += net(encoded_img)
                        print(f't: {t}')
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    encoded_img = data_convert2(encoded_img,args.device)
                    out_fr += net(encoded_img)

                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0


        with torch.no_grad():
            for img, label in test_data_loader:

                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 15).float()

                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    encoded_img = data_convert2(encoded_img,args.device)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max15.pth'))
            linear_layer = net.layer[1]  # 获取模型的第二层 (即 layer.Linear(2*300, 250))
            # 获取线性层的输入和输出维度
            input_dim = linear_layer.in_features
            output_dim = linear_layer.out_features
            # 使用维度变量构造文件路径
            model_file_path = f"/disk2/dongfh/spikingjelly_linxl/spikingjelly_yucz/projectxxx/final_layer1_param/15_linear_layer_params({input_dim}*{output_dim})_delta({delta}).pth"
            # 确保目录存在
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            torch.save(linear_layer.state_dict(), model_file_path)  # 保存参数到文件
            print("模型参数已保存到:", model_file_path)

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest15.pth'))
        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        # 日志文件路径
        log_file_path = f"/home/dongfh/spikingjelly_linxl/spikingjelly_yucz/log/{input_dim}*{output_dim}/delta({delta})/15分.txt"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # log_to_file(log_file_path, args, epoch, train_loss, train_acc, test_loss, test_acc, max_test_acc, train_speed, test_speed, start_time, train_class5_acc, test_class5_acc, fr_monitor)
        print("日志已保存到:", log_file_path)

    # 保存绘图用数据
    net.eval()


if __name__ == '__main__':
    main()
