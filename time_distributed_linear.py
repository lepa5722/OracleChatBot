from torch import nn
import torch

class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        if x_reshape.dtype != torch.float32:
            x_reshape = x_reshape.float()

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class LinearLayer(nn.Module):
    def __init__(self,
                 input_size,
                 size,
                 use_time_distributed=True,
                 batch_first=False):
        super(LinearLayer, self).__init__()

        self.use_time_distributed = use_time_distributed
        self.input_size = input_size
        self.size = size
        if use_time_distributed:
            self.layer = TimeDistributed(nn.Linear(input_size, size), batch_first=batch_first)
        else:
            self.layer = nn.Linear(input_size, size)

    def forward(self, x):
        return self.layer(x)


