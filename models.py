import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
class SharedSNN(nn.Module):
    def __init__(self, tau, num_classes_list=[3,6,4]):
        super().__init__()
        self.shared_layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(148, 40, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

        self.task_heads = nn.ModuleList()
        for num_classes in num_classes_list:
            self.task_heads.append(nn.Sequential(
                layer.Linear(40, 20, bias=False),
                neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
                layer.Linear(20, num_classes, bias=False),
                neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            ))

    def forward(self, x, task_id):
        shared_out = self.shared_layer(x)
        return self.task_heads[task_id](shared_out)
class SNN_1(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(148,40 , bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(40, 20, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(20, 3, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class SNN_2(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(148, 40, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(40, 20, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(20, 6, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class SNN_3(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(148, 40, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(40, 20, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Linear(20, 4, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        return self.layer(x)