import torch
import ppgs


class WrapperModel(torch.nn.Module):

    def __init__(self, modules, argListLengths=None):
        super().__init__()
        self.module_list= torch.nn.ModuleList(modules)
        self.argListLengths = argListLengths

    def forward(self, *args):
        output = args
        if self.argListLengths:
            for module, numArgs in zip(self.module_list, self.argListLengths):
                output = module(*output[:numArgs] if isinstance(output, tuple) else output)
        else:
            for module in self.modules:
                output = module(output)
        return output