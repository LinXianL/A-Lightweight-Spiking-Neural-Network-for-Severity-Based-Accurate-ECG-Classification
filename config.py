import argparse

def get_args(task='default'):
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    # 通用参数
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-resume', default=False, type=bool)
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    # 针对不同任务的参数
    if task == 'class15':
        parser.add_argument('-delta', default=0.06, type=float, help='delta value for training')
        parser.add_argument('-data-dir', default='./data/mnist', type=str, help='root dir of MNIST dataset')
        parser.add_argument('-out-dir', type=str, default='./output/classifier0', help='root dir for saving logs and checkpoint')
    elif task == 'class1':
        parser.add_argument('-delta', default=0.006, type=float, help='delta value for training')
        parser.add_argument('-data-dir', default='./data/class1', type=str, help='root dir of class1 dataset')
        parser.add_argument('-out-dir', type=str, default='./output/classifier1', help='root dir for saving logs and checkpoint')
    elif task == 'class2':
        parser.add_argument('-delta', default=0.006, type=float, help='delta value for training')
        parser.add_argument('-data-dir', default='./data/class2', type=str, help='root dir of class2 dataset')
        parser.add_argument('-out-dir', type=str, default='./output/classifier2', help='root dir for saving logs and checkpoint')
    elif task == 'class3':
        parser.add_argument('-delta', default=0.006, type=float, help='delta value for training')
        parser.add_argument('-data-dir', default='./data/class3', type=str, help='root dir of class3 dataset')
        parser.add_argument('-out-dir', type=str, default='./output/classifier3', help='root dir for saving logs and checkpoint')
    elif task == 'multi_train':
        parser.add_argument('-delta', default=0.006, type=float, help='delta value for training')
        parser.add_argument('-data-dir', default='./data/class3', type=str, help='root dir of class3 dataset')
        parser.add_argument('-out-dir', type=str, default='./output/multi_train', help='root dir for saving logs and checkpoint')
        parser.add_argument('-load_parm',type=bool,default=False)
    return parser.parse_args() 