import torch
import ppgs


class WrapperModel(torch.nn.Module):
    """Creates a wrapper module that acts like a sequential module but for submodules which take a different number of parameters each"""

    def __init__(self, modules, argListLengths=None):
        """Initializes a WrapperModule with modules which take argListLengths number of arguments respectively
        ex: [Transformer, Convolution], [2, 1]
        in the case of a transformer which needs a mask
        """
        super().__init__()
        self.module_list= torch.nn.ModuleList(modules)
        self.argListLengths = argListLengths

    def forward(self, *args):
        """Forward call. Pass arguments for the first module in the ModuleList"""
        output = args
        if self.argListLengths:
            for module, numArgs in zip(self.module_list, self.argListLengths):
                output = module(*output[:numArgs] if isinstance(output, tuple) else output)
        else:
            for module in self.modules:
                output = module(output)
        return output