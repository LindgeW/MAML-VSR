import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conformer import Conformer
from batch_beam_search import beam_decode


class PositionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.xavier_normal_(self.weight.weight)

    def forward(self, x):
        embeddings = self.weight(x)
        return x+embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, L, D)
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, bias=True, dropout=0.1):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.fc_q = nn.Linear(hid_dim, hid_dim, bias)
        self.fc_k = nn.Linear(hid_dim, hid_dim, bias)
        self.fc_v = nn.Linear(hid_dim, hid_dim, bias)
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q, K, V = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # energy = torch.einsum('bnqh,bnkh->bnqk', Q, K) * scale
        energy = torch.matmul(Q, K.transpose(-1, -2).contiguous()) * self.scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:  # [batch size, 1, 1, key len]
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.transpose(1, 2).contiguous().reshape(bs, -1, self.hid_dim)
        # x = [batch size, query len, n heads, head dim] -> [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x, attention


class FeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, ffn_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, ffn_dim)
        self.fc_2 = nn.Linear(ffn_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, ffn dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 ffn_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, True, dropout)
        self.feedforward = FeedforwardLayer(hid_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]
        _src, _ = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        _src = self.feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src


class CausalConv(nn.Module):
    def __init__(self, in_channel, hid_dim, out_channel, kernel_size=3, dilation=1, bid=False, dropout=0.1):
        super(CausalConv, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.bid = bid
        self.fwd_caus_conv = nn.Sequential(
            nn.ConstantPad1d((self.padding, 0), 0),  # F.pad(x, (self.padding, 0))
            nn.Conv1d(in_channel, hid_dim, kernel_size, padding=0, dilation=dilation),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.ConstantPad1d((self.padding, 0), 0),
            nn.Conv1d(hid_dim, out_channel, kernel_size, padding=0, dilation=dilation),
            nn.Dropout(dropout),
        )

        if self.bid:
            self.bwd_caus_conv = nn.Sequential(
                nn.ConstantPad1d((self.padding, 0), 0),  # F.pad(x, (self.padding, 0))
                nn.Conv1d(in_channel, hid_dim, kernel_size, padding=0, dilation=dilation),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                nn.ConstantPad1d((self.padding, 0), 0),
                nn.Conv1d(hid_dim, out_channel, kernel_size, padding=0, dilation=dilation),
                nn.Dropout(dropout),
            )

        self.layer_norm = nn.LayerNorm(out_channel)

    def forward(self, x):  # (B, T, D)
        out = self.fwd_caus_conv(x.transpose(1, 2)).transpose(1, 2)  # BTD to BDT to BTD
        if self.bid:
            rev_x = torch.flip(x, dims=[1])  # 沿着第二个dim进行反转
            rev_out = self.bwd_caus_conv(rev_x.transpose(1, 2)).transpose(1, 2)  # BTD to BDT to BTD
            rev_out = torch.flip(rev_out, dims=[1])
            # out = torch.cat((out, rev_out), dim=-1)
            out = out + rev_out + x
            return out
        return self.layer_norm(x + out)


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 ffn_dim,
                 norm_before=True,
                 dropout=0.1):
        super().__init__()
        self.norm_before = norm_before
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, True, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, True, dropout)
        self.feedforward = FeedforwardLayer(hid_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        # self.conv_layer = CausalConv(hid_dim, hid_dim*2, hid_dim, 3, dropout=dropout)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        # tgt = [batch size, tgt len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        # src_mask = [batch size, 1, 1, src len]
        residual = tgt
        if self.norm_before:
            tgt = self.self_attn_layer_norm(tgt)
        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)  # Masked Multi-Head Self-Attention
        # tgt = [batch size, tgt len, hid dim]
        x = residual + self.dropout(_tgt)
        if not self.norm_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.norm_before:
            x = self.enc_attn_layer_norm(x)
        _tgt, attention = self.encoder_attention(x, enc_src, enc_src, src_mask)
        # tgt = [batch size, tgt len, hid dim]
        x = residual + self.dropout(_tgt)
        if not self.norm_before:
            x = self.enc_attn_layer_norm(x)

        residual = x
        if self.norm_before:
            x = self.ff_layer_norm(x)
        _tgt = self.feedforward(x)
        # tgt = [batch size, tgt len, hid dim]
        # attention = [batch size, n heads, tgt len, src len]
        x = residual + self.dropout(_tgt)
        if not self.norm_before:
            x = self.ff_layer_norm(x)

        # add conv layer
        # x = self.conv_layer(x)
        return x, attention


class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            *[
                nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4, bias=False),  # H/4 x W/4
                # nn.Conv2d(in_chans, embed_dim//4, kernel_size=2, stride=2),   # H/2 x W/2
                nn.BatchNorm2d(embed_dim // 4),  # BN or LN
                # nn.GELU(),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2, bias=False),  # H/8 x W/8
                nn.BatchNorm2d(embed_dim // 2),
                # nn.GELU(),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),  # H/16 x W/16
                # nn.BatchNorm2d(embed_dim),
            ])

    def forward(self, x):  # (BT, C, H, W)
        x = self.proj(x)
        return x.flatten(2).transpose(-1, -2).squeeze(-2)  # (BT, D, N) -> (BT, N, D)


'''
class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            *[
            nn.Conv2d(in_chans, embed_dim//8, kernel_size=3, stride=2, padding=1, bias=False),   # H/2 x W/2
            nn.BatchNorm2d(embed_dim//8),   # BN or LN
            #nn.GELU(),
            nn.ReLU(),
            nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=3, stride=2, padding=1, bias=False), # H/4 x W/4
            nn.BatchNorm2d(embed_dim//4),
            #nn.GELU(),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False), # H/8 x W/8
            nn.BatchNorm2d(embed_dim//2),
            #nn.GELU(),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1),    # H/16 x W/16
            #nn.BatchNorm2d(embed_dim),
        ])

    def forward(self, x): # (BT, C, H, W)
        x = self.proj(x)
        return x.flatten(2).transpose(-1, -2).squeeze(-2)  # (BT, D, N) -> (BT, N, D)
'''


class EarlyConv(nn.Module):
    def __init__(self, in_chans=1, embed_dim=768, bias=False):
        super().__init__()
        self.conv_layers = nn.Sequential(*[
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(embed_dim // 4),  # BN or LN
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(embed_dim // 2),  # BN or LN
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias),
            # nn.BatchNorm2d(embed_dim),   # BN or LN
            # nn.ReLU(),
        ])
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_layers(x)
        mx = self.gmp(x).squeeze(-1).squeeze(-1)
        # ax = self.gap(x).squeeze(-1).squeeze(-1)
        # out = torch.cat((mx, ax), dim=-1).contiguous()
        return mx


class TubeletEarlyConv(nn.Module):
    def __init__(self, in_chans=1, embed_dim=768, bias=False):
        super().__init__()
        self.conv_layers = nn.Sequential(*[
            # (B, C, T, H, W)
            # nn.Conv3d(in_chans, embed_dim//4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=bias),
            # nn.Conv3d(in_chans, embed_dim//4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=bias),
            # nn.Conv3d(in_chans, embed_dim//4, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=bias),
            nn.Conv3d(in_chans, embed_dim // 4, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1), bias=bias),
            nn.BatchNorm3d(embed_dim // 4),
            # nn.ReLU(),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                      bias=bias),
            nn.BatchNorm3d(embed_dim // 2),
            # nn.ReLU(),
            nn.SiLU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=bias),
            nn.BatchNorm3d(embed_dim),
            # nn.ReLU(),
            nn.SiLU()
        ])

        '''
        self.front3d = nn.Sequential(*[
            # (B, C, T, H, W)
            #nn.Conv3d(in_chans, embed_dim//4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=bias),
            #nn.Conv3d(in_chans, embed_dim//4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=bias),
            #nn.Conv3d(in_chans, embed_dim//4, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=bias),
            nn.Conv3d(in_chans, embed_dim//4, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1), bias=bias),
            nn.BatchNorm3d(embed_dim//4),
            #nn.ReLU(),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            #nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        ])
        self.front2d = nn.Sequential(*[
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=bias),
            nn.BatchNorm2d(embed_dim//2),
            #nn.ReLU(),
            nn.SiLU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=bias),
            nn.BatchNorm2d(embed_dim),
            #nn.ReLU(),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        '''

        # self.gmp = nn.AdaptiveMaxPool3d((None, 1, 1))
        # self.gmp = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.gmp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # (B, T, N, C, H, W)
        B, T, N = x.shape[:3]
        x = x.transpose(1, 2).flatten(0, 1).transpose(1, 2)  # (BN, T, C, H, W) -> (BN, C, T, H, W)
        x = self.conv_layers(x)
        # x = self.gmp(x.transpose(1, 2)).squeeze(-1).squeeze(-1)   # (BN, C, T, H, W) -> (BN, T, C, 1, 1) -> (BN, T, C)
        # x = x.reshape(B, N, T, -1).transpose(1, 2)
        # x = self.gmp(x).squeeze(-1).squeeze(-1)   # (BN, C, T, H, W) -> (BN, C, T, 1, 1) -> (BN, C, T)
        # x = x.reshape(B, N, -1, T).permute(0, 3, 1, 2)
        ## 2D pooling
        x = self.gmp(x.transpose(1, 2).flatten(0, 1)).squeeze(-1).squeeze(
            -1)  # (BN, C, T, H, W) -> (BN, T, C, H, W) -> (BNT, C, 1, 1) -> (BNT, C)
        x = x.reshape(B, N, T, -1).transpose(1, 2)
        # x = self.front3d(x)
        # x = self.front2d(x.transpose(1, 2).flatten(0, 1)).squeeze(-1).squeeze(-1)
        # x = x.reshape(B, N, T, -1).transpose(1, 2)
        return x  # (B, T, N, C)


# 提取Patch图像信息
class VisualFrontEnd2D(nn.Module):
    def __init__(self, in_channel=3, embed_dim=256):
        super(VisualFrontEnd2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):  # B x C x H x W
        out = self.conv(x)
        out = out.squeeze()  # (B, D)
        return out


class VisualFrontEnd3D(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=256):
        super(VisualFrontEnd3D, self).__init__()
        self.stcnn = nn.Sequential(
            nn.Conv3d(in_channel, hidden_dim // 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2),
                      bias=False),
            # nn.Conv3d(in_channel, hidden_dim//4, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(hidden_dim // 4),
            nn.ReLU(True),
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(hidden_dim // 4, hidden_dim // 2, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2),
                      bias=False),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(True),
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(True),
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        # self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(8192, hidden_dim)  # flatten

    def forward(self, x):  # (B, C, T, H, W)
        cnn = self.stcnn(x)  # (B, D, T, H, W)
        cnn = cnn.transpose(1, 2).contiguous()  # (B, T, D, H, W)
        b, t, d, h, w = cnn.size()
        # cnn = cnn.reshape(b * t, d, h, w)
        # out = self.gmp(cnn).reshape(b, t, d)
        out = self.fc(cnn.reshape(b, t, -1))
        return out


class ConformerEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 hid_dim,
                 n_layers,
                 n_heads,
                 ffn_ratio,
                 dropout):
        super().__init__()
        self.hid_dim = hid_dim

        # lip or face ROI
        self.lipcnn = VisualFrontEnd3D(in_channel, hid_dim)

        self.conformer = Conformer(
            input_dim=hid_dim,
            num_heads=n_heads,
            ffn_dim=ffn_ratio * hid_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout)

    def get_mask_from_idxs(self, src, pad_idx=0):  # src = [batch size, src len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def get_mask_from_lengths(self, lengths, max_len=None):
        '''
         param:   lengths - [batch_size]
         return:  mask - [batch_size, max_len]
        '''
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        mask = ids < lengths.unsqueeze(1).expand(-1, max_len)  # True or False
        return mask

    def forward(self, lips, src_lens):
        # lips: (B, T, C, H, W)
        B, T = lips.shape[:2]
        lips = lips.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        feat = self.lipcnn(lips)
        src, _ = self.conformer(feat, src_lens)
        src_mask = self.get_mask_from_lengths(src_lens, T).unsqueeze(1).unsqueeze(2)
        return src, src_mask


class Encoder(nn.Module):
    def __init__(self,
                 in_channel,
                 hid_dim,
                 n_layers,
                 n_heads,
                 ffn_ratio,
                 dropout,
                 max_length=100):
        super().__init__()
        self.hid_dim = hid_dim
        self.visual_front = VisualFrontEnd3D(in_channel, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, max_length)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  ffn_ratio * hid_dim,
                                                  dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
        #self.code = nn.Parameter(torch.zeros(n_layers, 5, hid_dim))
        #self.code = nn.Parameter(torch.zeros(1, 5, hid_dim))
        #nn.init.xavier_uniform_(self.code.data)
        self.adapter = nn.Sequential(nn.Linear(hid_dim, hid_dim//4), nn.GELU(), nn.Linear(hid_dim//4, hid_dim))

    def get_mask_from_idxs(self, src, pad_idx=0):  # src = [batch size, src len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def get_mask_from_lengths(self, lengths, max_len=None):
        '''
         param:   lengths --- [Batch_size]
         return:  mask --- [Batch_size, max_len]
        '''
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        mask = ids < lengths.unsqueeze(1).expand(-1, max_len)  # True or False
        return mask

    def forward(self, x, src_lens):  # (b, t, c, h, w)
        B, T = x.shape[0], x.shape[1]
        x = x.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        feat = self.visual_front(x)
        src = self.pos_embedding(feat)
        #src = self.dropout(feat + pos_embed + self.code.unsqueeze(0))
        src_mask = self.get_mask_from_lengths(src_lens, T).unsqueeze(1).unsqueeze(2)  # (b, 1, 1, t)
        #cmb_mask = torch.cat(((torch.ones(src_mask.shape[0], 1, 1, 5) == 1).to(src_mask.device), src_mask), dim=-1)
        #cmb_src = torch.cat((self.code.expand(src.shape[0], -1, -1), src), dim=1)
        for i, layer in enumerate(self.layers):
            src = layer(src, src_mask)
            src = self.adapter(src) + src
            #cmb_src = torch.cat((self.code[i].expand(src.shape[0], -1, -1), src), dim=1)
            #cmb_src = layer(cmb_src, cmb_mask)
            #src = cmb_src[:, 5:]
        return src, src_mask
        #return cmb_src, cmb_mask


class Encoder2(nn.Module):
    def __init__(self,
                 in_channel,
                 hid_dim,
                 n_layers,
                 n_heads,
                 ffn_ratio,
                 dropout=0.1,
                 max_length=100):
        super().__init__()
        self.hid_dim = hid_dim

        self.visual_front = VisualFrontEnd3D(in_channel, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, max_length)
        # pytorch >= 1.12
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim,
                                                   nhead=n_heads,
                                                   batch_first=True,
                                                   dim_feedforward=ffn_ratio*hid_dim,
                                                   dropout=dropout)
        # norm = nn.LayerNorm
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=None)
        self.dropout = nn.Dropout(dropout)

    def get_mask_from_idxs(self, src, pad_idx=0):  # src = [batch size, src len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def get_mask_from_lengths(self, lengths, max_len=None):
        '''
         param:   lengths --- [Batch_size]
         return:  mask --- [Batch_size, max_len]
        '''
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        mask = ids < lengths.unsqueeze(1).expand(-1, max_len)  # True or False
        return mask

    def forward(self, x, src_lens):  # (b, t, c, h, w)
        B, T = x.shape[0], x.shape[1]
        x = x.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        feat = self.visual_front(x)
        src = self.pos_embedding(feat)
        # src = [batch size, src len, hid dim]
        src_mask = self.get_mask_from_lengths(src_lens, T)  # (b, t)
        src = self.encoder(src, src_key_padding_mask=~src_mask)  # True for padding, and False for valid value
        return src, src_mask.unsqueeze(1).unsqueeze(2)


class Decoder(nn.Module):
    def __init__(self,
                 num_cls,
                 hid_dim,
                 n_layers,
                 n_heads,
                 ffn_ratio,
                 dropout,
                 norm_before=False,
                 pad_idx=0,
                 max_length=100):
        super().__init__()
        self.hid_dim = hid_dim
        self.tgt_pad_idx = pad_idx  # pad token index
        self.norm_before = norm_before
        self.tok_embedding = nn.Embedding(num_cls, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, max_length)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  ffn_ratio * hid_dim,
                                                  norm_before,
                                                  dropout)
                                     for _ in range(n_layers)])
        if self.norm_before:
            self.post_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hid_dim, num_cls - 1)  # excluding bos

    def make_tgt_mask(self, tgt):  # [batch size, tgt len]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # tgt_pad_mask = [batch size, 1, 1, tgt len]
        tgt_len = tgt.shape[1]
        # 下三角(包括对角线)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        # tgt_sub_mask = [tgt len, tgt len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        return tgt_mask

    def forward(self, tgt, enc_src, src_mask):
        # tgt = [batch size, tgt len]
        # enc_src = [batch size, src len, hid dim]
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        # src_mask = [batch size, 1, 1, src len]
        
        tgt_mask = self.make_tgt_mask(tgt)  # [batch size, 1, tgt len, tgt len]

        tgt = self.tok_embedding(tgt) * (self.hid_dim ** 0.5)
        tgt = self.pos_embedding(tgt)
        # tgt = [batch size, tgt len, hid dim]

        for layer in self.layers:
            tgt, attn = layer(tgt, enc_src, tgt_mask, src_mask)
        # tgt = [batch size, tgt len, hid dim]
        # attn = [batch size, n heads, tgt len, src len]
        if self.norm_before:
            tgt = self.post_layer_norm(tgt)  # for layer norm before

        output = self.fc_out(tgt)
        # output = [batch size, tgt len, output dim]
        return output, attn


class Seq2Seq(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
                
        self.encoder = Encoder(opt.in_channel,
                               opt.hidden_dim,
                               opt.enc_layers,
                               opt.head_num,
                               opt.ffn_ratio,
                               opt.drop_attn)
        
        '''
        self.encoder = ConformerEncoder(opt.in_channel,
                                        opt.hidden_dim,
                                        opt.enc_layers,
                                        opt.head_num,
                                        opt.ffn_ratio,
                                        opt.drop_attn)
        '''

        self.decoder = Decoder(opt.tgt_vocab_size,
                               opt.hidden_dim,
                               opt.dec_layers,
                               opt.head_num,
                               opt.ffn_ratio,
                               opt.drop_attn)

    def forward(self, vids, tgt, src_lens=None, tgt_lens=None):
        enc_src, src_mask = self.encoder(vids, src_lens)  # [batch size, src len, hid dim]
        logits, attn = self.decoder(tgt[:, :-1], enc_src, src_mask)
        loss = F.cross_entropy(logits.transpose(-1, -2).contiguous(), tgt[:, 1:].long(),
                               ignore_index=self.opt.tgt_pad_idx)
        return loss

    def greedy_decoding(self, src_vid, src_pts, src_motion, src_lens, bos_id, eos_id, pad_id=0):  # (bs, src len)
        print('greedy decoding ......')
        tgt_len = self.opt.max_dec_len
        bs = src_vid.shape[0]
        tgt_align = torch.tensor([[bos_id]] * bs).to(src_vid.device)  # (bs, 1)
        with torch.no_grad():
            enc_src, src_mask = self.encoder(src_vid, src_pts, src_motion, src_lens)
            ctc_score = self.ctc_dec(enc_src)

            for t in range(tgt_len):
                attn_score, _ = self.decoder(tgt_align, enc_src, src_mask)
                pred = attn_score.argmax(dim=-1)[:, -1]  # (bs, tgt_len) -> (bs, )   # greedy decoding
                # pred = self.topp_decoding(output[:, -1], top_p=0.96)
                # pred = self.topk_decoding(output[:, -1])
                if pred.cpu().tolist() == [eos_id] * bs or pred.cpu().tolist() == [pad_id] * bs:
                    break
                tgt_align = torch.cat((tgt_align, pred.unsqueeze(1)), dim=1).contiguous()

            L = min(ctc_score.shape[1], attn_score.shape[1])
            score = 0.1 * ctc_score[:, :L] + 0.9 * attn_score[:, :L]
        dec_pred = score.argmax(dim=-1)[1:].detach().cpu().numpy()
        # dec_pred = tgt_align[:, 1:].detach().cpu().numpy()  # (bs, tgt_len)
        return dec_pred

    # def greedy_decoding(self, src_inp, bos_id, eos_id, pad_id=0):  # (bs, src len)
    #     tgt_len = self.opt.max_dec_len
    #     bs = src_inp.shape[0]
    #     res = torch.zeros((bs, tgt_len)).to(src_inp.device)
    #     tgt_align = torch.zeros((bs, tgt_len), dtype=torch.long).to(src_inp.device)
    #     tgt_align[:, 0] = bos_id
    #     with torch.no_grad():
    #         enc_src, src_mask = self.encoder(src_inp)
    #         for t in range(tgt_len):
    #             output, attn = self.decoder(tgt_align, enc_src, src_mask)
    #             pred = output.argmax(dim=-1)
    #             if pred[:, t].cpu().tolist() == [eos_id] * bs or pred[:, t].cpu().tolist() == [pad_id] * bs:
    #                 break
    #             res[:, t] = pred[:, t]
    #             if t < tgt_len - 1:
    #                 tgt_align[:, t + 1] = pred[:, t]

    #     dec_pred = res.detach().cpu().numpy()  # (bs, tgt_len)
    #     return dec_pred

    def beam_search_decoding(self, lips, src_lens, bos_id, eos_id, max_dec_len=80, pad_id=0, mode='attn'):
        assert mode in ['ctc', 'attn']
        if mode == 'ctc':
            with torch.no_grad():
                enc_src, src_mask = self.encoder(lips, src_lens)[
                                    :2]  # [batch size, src len, hid dim]
                output = self.ctc_dec(enc_src)
                res = output.argmax(dim=-1)
        else:
            with torch.no_grad():
                enc_src, src_mask = self.encoder(lips, src_lens)
                res = beam_decode(self.decoder, enc_src, src_mask, bos_id, eos_id,
                                  max_output_length=max_dec_len,
                                  beam_size=6)
        return res.detach().cpu()
