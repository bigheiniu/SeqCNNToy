import torch
import torch.nn as nn

class SeqClassifyModel(nn.Module):
    def __init__(self, in_chanell, out_chanell, kenel_size, num_class):
        super(SeqClassifyModel, self).__init__()
        self.cnn1 = nn.Conv2d(kernel_size=kenel_size, in_channels=in_chanell, out_channels=out_chanell)
        self.fc = nn.Linear(out_chanell, num_class)

    def forward(self, seq_data):
        # seq_data shape: batch_size, in_channel, width, height
        # seq_cnn shape: batch_size, out_channel, width_new, height_new
        seq_cnn = self.cnn1(seq_data)
        seq_act = torch.relu(seq_cnn)
        # view => function like numpy reshape
        # seq_act type: Tensor
        seq_max_pool,_ = torch.max(seq_act.view(seq_act.shape[0], seq_act.shape[1], -1), dim=-1)
        output = self.fc(seq_max_pool)
        return output

