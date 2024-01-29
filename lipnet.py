import torch
import torch.nn as nn
import torch.nn.init as init
import math


# class LipNet(torch.nn.Module):
#     def __init__(self, num_cls, dropout_p=0.5):
#         super(LipNet, self).__init__()
#         self.conv1 = nn.Conv3d(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))   # 灰度图
#         self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
#
#         self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
#         self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
#
#         self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
#         self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
#
#         self.gru1 = nn.GRU(4800, 256, 1, bidirectional=True)
#         self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)
#
#         self.FC = nn.Linear(512, num_cls)
#         self.dropout_p = dropout_p
#
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout3d = nn.Dropout3d(self.dropout_p)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self._init()
#
#     def _init(self):
#         init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
#         init.constant_(self.conv1.bias, 0)
#
#         init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
#         init.constant_(self.conv2.bias, 0)
#
#         init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
#         init.constant_(self.conv3.bias, 0)
#
#         init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
#         init.constant_(self.FC.bias, 0)
#
#         for m in (self.gru1, self.gru2):
#             stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
#             for i in range(0, 256 * 3, 256):
#                 init.uniform_(m.weight_ih_l0[i: i + 256],
#                               -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0[i: i + 256])
#                 init.constant_(m.bias_ih_l0[i: i + 256], 0)
#                 init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
#                               -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
#                 init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool1(x)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool2(x)
#
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool3(x)
#
#         # (B, C, T, H, W)->(T, B, C, H, W)
#         x = x.permute(2, 0, 1, 3, 4).contiguous()
#         # (B, C, T, H, W)->(T, B, C*H*W)
#         x = x.reshape(x.size(0), x.size(1), -1)
#
#         self.gru1.flatten_parameters()
#         self.gru2.flatten_parameters()
#         x, _ = self.gru1(x)
#         x = self.dropout(x)
#         x, _ = self.gru2(x)
#         x = self.dropout(x)
#         x = self.FC(x)   # T B V
#         # x = x.permute(1, 0, 2).contiguous()   # B T V
#         return x


class LipNet(nn.Module):
    def __init__(self, vocab_size, dropout=0.5):
        super(LipNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(dropout),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(dropout),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(dropout)
        )
        
        rnn_size = 256
        # T B C*H*W
        self.gru1 = nn.GRU(3072, rnn_size, 1, bidirectional=True)
        self.drp1 = nn.Dropout(dropout)
        # T B F
        self.gru2 = nn.GRU(rnn_size * 2, rnn_size, 1, bidirectional=True)
        self.drp2 = nn.Dropout(dropout)
        # T B V
        self.fc = nn.Linear(rnn_size * 2, vocab_size)   # 含空白字符

        self.adapter = nn.Sequential(nn.Linear(512, 512//4), nn.GELU(), nn.Linear(512//4, 512))
        #self.sc = nn.Parameter(torch.zeros(1, 50))
        #self.adanet = nn.Linear(50, 512, bias=True)
        #self.adanet2 = nn.Linear(50, 512, bias=True)

        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        init.kaiming_normal_(self.fc.weight, nonlinearity='sigmoid')
        init.constant_(self.fc.bias, 0)
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + rnn_size))
            for i in range(0, rnn_size * 3, rnn_size):
                init.uniform_(m.weight_ih_l0[i: i + rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + rnn_size])
                init.constant_(m.bias_ih_l0[i: i + rnn_size], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + rnn_size],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + rnn_size])
                init.constant_(m.bias_ih_l0_reverse[i: i + rnn_size], 0)

    def reset_params(self):
        #nn.init.xavier_uniform_(self.fc.weight)
        #nn.init.zeros_(self.fc.bias)
        #self.gru2.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x):
        x = self.conv(x)   # B C T H W
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # T B C H W
        x = x.reshape(x.size(0), x.size(1), -1)  # T B CHW
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, _ = self.gru1(x)
        x = self.drp1(x)
        x = self.adapter(x) + x
        #x = self.adanet(self.sc).unsqueeze(0) + x
        x, _ = self.gru2(x)
        x = self.drp2(x)
        #x = self.adanet2(self.sc).unsqueeze(0) + x
        x = self.fc(x)   # T B V
        return x


