import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import pdb

import numpy as np
import random


class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """

    def __init__(self, a_dim=128, v_dim=512, hidden_size=512, map_size=49):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.affine_audio = nn.Linear(a_dim, hidden_size)
        self.affine_video = nn.Linear(v_dim, hidden_size)
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_h = nn.Linear(map_size, 1, bias=False)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):
        # audio: [bs, t, 128]
        # video: [bs, t, 7, 7, 512]
        V_DIM = video.size(-1)
        v_t = video.view(video.size(0) * video.size(1), -1, V_DIM)  # [bs*t, 49, 512]
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))  # [bs*t, 49, 512]
        a_t = audio.view(-1, audio.size(-1))  # [bs*t, 128]
        a_t = self.relu(self.affine_audio(a_t))  # [bs*t, 512]
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2)  # [bs*t, 49, 49] + [bs*t, 49, 1]

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)  # [bs*t, 49]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))  # attention map, [bs*t, 1, 49]
        c_t = torch.bmm(alpha_t, V).view(-1, V_DIM)  # [bs*t, 1, 512]
        video_t = c_t.view(video.size(0), -1, V_DIM)  # attended visual features, [bs, t, 512]
        return video_t


class LSTM_A_V(nn.Module):
    def __init__(self, a_dim, v_dim, hidden_dim=128, seg_num=10):
        super(LSTM_A_V, self).__init__()

        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm_video = nn.LSTM(v_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_fea, v_fea):
        bs, seg_num, a_dim = a_fea.shape
        hidden_a = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda())
        hidden_v = (torch.zeros(2, bs, a_dim).cuda(), torch.zeros(2, bs, a_dim).cuda())
        return hidden_a, hidden_v

    def forward(self, a_fea, v_fea):
        # a_fea, v_fea: [bs, t, 128]
        hidden_a, hidden_v = self.init_hidden(a_fea, v_fea)
        # Bi-LSTM for temporal modeling
        self.lstm_video.flatten_parameters()  # .contiguous()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(a_fea, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v)

        return lstm_audio, lstm_video



class PSP(nn.Module):
    """Postive Sample Propagation module"""

    def __init__(self, a_dim=256, v_dim=256, hidden_dim=256, out_dim=256):
        super(PSP, self).__init__()
        self.v_L1 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_L2 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_fc = nn.Linear(v_dim, out_dim, bias=False)
        self.a_L1 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_L2 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_fc = nn.Linear(a_dim, out_dim, bias=False)
        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # channels = 256
        # #         self.msc = MSC(channels)
        # self.fuse_1x1conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)  # default=0.1
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

        layers = [self.v_L1, self.v_L2, self.a_L1, self.a_L2, self.a_fc, self.v_fc]
        self.init_weights(layers)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)
            # nn.init.orthogonal(layer.weight)
            # nn.init.kaiming_normal_(layer.weight, mode='fan_in')

    def forward(self, a_fea, v_fea, thr_val):
        # a_fea: [bs, t, 256]
        # v_fea: [bs, t, 256]
        # thr_val: the hyper-parameter for pruing process

        v_branch1 = self.dropout(self.activation(self.v_L1(v_fea)))  # [bs, t, hidden_dim]
        v_branch2 = self.dropout(self.activation(self.v_L2(v_fea)))
        a_branch1 = self.dropout(self.activation(self.a_L1(a_fea)))
        a_branch2 = self.dropout(self.activation(self.a_L2(a_fea)))

        beta_va = torch.bmm(v_branch2, a_branch1.permute(0, 2, 1))  # row(v) - col(a), [bs, t, t]
        beta_va = beta_va / torch.sqrt(torch.FloatTensor([v_branch2.shape[2]]).cuda())
        beta_va = F.relu(beta_va)  # ReLU
        beta_av = beta_va.permute(0, 2, 1)  # transpose
        sum_v_to_a = torch.sum(beta_va, dim=-1, keepdim=True)
        beta_va = beta_va / (sum_v_to_a + 1e-8)  # [bs, t, t]

        time = beta_va.shape[-1]
        thr_val = thr_val * 10 / time

        gamma_va = (beta_va > thr_val).float() * beta_va
        sum_v_to_a = torch.sum(gamma_va, dim=-1, keepdim=True)  # l1-normalization
        gamma_va = gamma_va / (sum_v_to_a + 1e-8)

        sum_a_to_v = torch.sum(beta_av, dim=-1, keepdim=True)
        beta_av = beta_av / (sum_a_to_v + 1e-8)
        gamma_av = (beta_av > thr_val).float() * beta_av
        sum_a_to_v = torch.sum(gamma_av, dim=-1, keepdim=True)
        gamma_av = gamma_av / (sum_a_to_v + 1e-8)
        selected_time_segments = []

        pred = torch.zeros(time)
        for t in range(gamma_av.size(1)):
            for t_prime in range(gamma_av.size(2)):
                if t != t_prime:
                    continue
                else:
                    if gamma_av[0, t, t_prime] > 0:
                        selected_time_segments.append(t)
                        pred[t] = 1
                    else:
                        pred[t] = 0

        a_pos = torch.bmm(gamma_va, a_branch2)
        v_psp = v_fea + a_pos
        v_pos = torch.bmm(gamma_av, v_branch1)
        a_psp = a_fea + v_pos
        v_psp = self.dropout(self.relu(self.v_fc(v_psp)))
        a_psp = self.dropout(self.relu(self.a_fc(a_psp)))
        v_psp = self.layer_norm(v_psp)
        a_psp = self.layer_norm(a_psp)
        a_v_fuse = torch.mul(v_psp + a_psp, 0.5)
        return a_v_fuse, v_psp, a_psp, selected_time_segments, pred


class AVSimilarity(nn.Module):
    """ function to compute audio-visual similarity"""

    def __init__(self, ):
        super(AVSimilarity, self).__init__()

    def forward(self, v_fea, a_fea):
        # fea: [bs, t, 256]
        v_fea = F.normalize(v_fea, dim=-1)
        a_fea = F.normalize(a_fea, dim=-1)
        cos_simm = torch.sum(torch.mul(v_fea, a_fea), dim=-1)  # [bs, t]
        return cos_simm


class psp_net(nn.Module):
    '''
    System flow for fully supervised audio-visual event localization.
    '''

    def __init__(self, a_dim=128, v_dim=512, hidden_dim=128, category_num=37):
        super(psp_net, self).__init__()
        self.fa = nn.Sequential(
            nn.Linear(a_dim, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.fv = nn.Sequential(
            nn.Linear(v_dim, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )

        self.relu = nn.ReLU(inplace=True)
        self.attention = AVGA(v_dim=v_dim)
        self.lstm_a_v = LSTM_A_V(a_dim=a_dim, v_dim=hidden_dim, hidden_dim=hidden_dim)
        self.psp = PSP(a_dim=a_dim*2, v_dim=hidden_dim*2)
        self.av_simm = AVSimilarity()

        self.L1 = nn.Linear(2*hidden_dim, 128, bias=False)
        self.L2 = nn.Linear(128, category_num, bias=False)

        self.L3 = nn.Linear(256, 64)
        self.L4 = nn.Linear(64, 2)
        self.l5 = nn.Linear(128, 49, bias=False)
        # layers = [self.L1, self.L2]
        layers = [self.L1, self.L2, self.L3, self.L4]
        self.init_layers(layers)

    def init_layers(self, layers):
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

    def forward(self, audio, video, thr_val):
        # audio: [bs, t, 128]
        # video: [bs, t, 7, 7, 512]
        bs, seg_num, H, W, v_dim = video.shape
        fa_fea = self.fa(audio) # [bs, t, 128]
        video_t = self.attention(fa_fea, video) # [bs, t, 512]
        video_t = self.fv(video_t)
        lstm_audio, lstm_video = self.lstm_a_v(fa_fea, video_t) # [bs, t, 256]
        fusion, final_v_fea, final_a_fea, selected_time_segments, pred = self.psp(lstm_audio, lstm_video, thr_val=thr_val)
        cross_att = self.av_simm(final_v_fea, final_a_fea)
        out = self.relu(self.L1(fusion))
        out = self.L2(out)  # [bs, t, 29]
        return fusion, out, cross_att, selected_time_segments, pred

