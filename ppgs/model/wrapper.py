import torch

import ppgs


class LengthsWrapperModel(torch.nn.Module):
    """Creates a wrapper module that acts like a sequential module but for submodules which take a different number of parameters each"""

    def __init__(self, modules, length_modules=None):
        """Initializes a WrapperModule with modules which take argListLengths number of arguments respectively
        """
        super().__init__()
        self.module_list = torch.nn.ModuleList(modules)
        self.length_modules = [False]*len(modules) if length_modules is None else length_modules

    def forward(self, input, lengths=None):
        """Forward call. Pass arguments for the first module in the ModuleList"""
        output = input
        mask = lengths
        for module, module_needs_lengths in zip(self.module_list, self.length_modules):
            if module_needs_lengths:
                output = module(output, mask)
            else:
                output = module(output)
        return output