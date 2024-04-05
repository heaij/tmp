import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np

# import torch.nn.functional as F
# import torch.nn as nn
# import torch
# import math
#
#
# class MLP(nn.Module):
#     def __init__(self, input_size, layer1_size, layer2_size, output_size, sequence_length):
#         super(MLP, self).__init__()
#         self.layer1 = nn.Linear(input_size * sequence_length, layer1_size)
#         self.layer2 = nn.Linear(layer1_size, layer2_size)
#         self.output = nn.Linear(layer2_size, output_size)

#     def forward(self, x):
#         # Reshape input to [batch_size, input_size * sequence_length]
#         x = x.view(x.size(0), -1)

#         x = F.dropout(self.layer1(x), p=0.1)
#         x = torch.tanh(x)
#         x = F.dropout(self.layer2(x), p=0.1)
#         x = torch.tanh(x)
#         x = self.output(x).squeeze(1)
#         return x


class CNN(nn.Module):
    def __init__(self, input_channels=1, output_size=144):
        super(CNN, self).__init__()
        self.emb_layer = nn.Linear(10, 3)
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x_ = x.transpose(1, 2)  # [batch_size, 10, 144] -> [batch_size, 144, 10]
        x_emb = self.emb_layer(x_)
        x_emb = x_emb.transpose(1, 2)  # [batch_size, 3, 10]
        x_emb = x_emb.squeeze(1)  # Remove the singleton dimension

        out1 = self.pool(F.relu(self.conv1(x_emb)))
        out2 = self.pool(F.relu(self.conv2(out1)))
        out3 = self.pool(F.relu(self.conv3(out2)))
        out4 = self.pool(F.relu(self.conv4(out3)))

        out_flat = out4.view(out4.size(0), -1)  # Flatten the output for fully connected layer
        out_last = self.fc(out_flat)
        return out_last



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        # self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print("middle_output.shape:", x.shape)
        # middle_output.shape: torch.Size([50, 10, 144])
        r_out, _ = self.rnn(x, None)  # None represents zero initial hidden state
        # out = self.out(r_out[:, -1, :])  # return the last value
        # out = self.sigmoid(out)
        out=r_out[:, -1, :]
        out = self.sigmoid(out)


        return out


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=2,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, output_size)

#     def init_hidden(self, x):
#         batch_size = x.shape[0]
#         init_h = torch.zeros(2, batch_size, self.rnn.hidden_size, device=x.device, requires_grad=True)
#         return init_h

#     def forward(self, x, h=None):
#         h = h if h else self.init_hidden(x)
#         out, h = self.rnn(x, h)
#         out = self.fc(out[:, -1, :])
#         return out



# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers):
#         super(GRU, self).__init__()

#         # Encoder
#         self.encoder = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#         )

#         # Decoder
#         self.decoder = nn.GRU(
#             input_size=hidden_size,  # Updated input size to be the hidden size of the encoder
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#         )

#         self.sigmoid = nn.Sigmoid()
#         self.out = nn.Linear(hidden_size, output_size)  # Adjusted output size to match your desired output

#     def forward(self, x):
#         # Encoder forward pass
#         encoder_output, _ = self.encoder(x)

#         # Decoder forward pass
#         decoder_output, _ = self.decoder(encoder_output)

#         # Output layer
#         output = self.out(decoder_output[:, -1, :])

#         return output






class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        # self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print("middle_output.shape:", x.shape)
        # middle_output.shape: torch.Size([50, 10, 144])
        r_out, _ = self.gru(x, None) 
         # None represents zero initial hidden state
        # print(x.shape)
        # print(r_out.shape)
        # out = self.out(r_out[:, -1, :])  # return the last value

        out=r_out[:, -1, :]
        out = self.sigmoid(out)
        # out = self.sigmoid(out)


        return out




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        #self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print("middle_output.shape:", x.shape)
        # middle_output.shape: torch.Size([50, 10, 144])
        r_out, _ = self.lstm(x, None) 
        # print(x.shape)torch.Size#([256, 720, 1])
        # print(r_out.shape) # torch.Size([256, 720, 144])
        #out = self.out(r_out[:, -1, :])  # return the last value
        
        # print(out.shape)torch.Size([256, 192])
        out=r_out[:, -1, :]
        out = self.sigmoid(out)


        return out

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size,num_layers):
#         super(LSTM, self).__init__()
#         # Encoder
#         self.encoder = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#         )

#         # Decoder
#         self.decoder = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#         )

#         self.sigmoid = nn.Sigmoid()
#         self.out = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Encoder
#         encoder_out, (encoder_h_n, encoder_h_c) = self.encoder(x, None)

#         # Decoder
#         decoder_out, (decoder_h_n, decoder_h_c) = self.decoder(x, (encoder_h_n, encoder_h_c))

#         out = self.out(decoder_out[:, -1, :])
#         #out = self.sigmoid(out)
#         return out




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_positional_encoding(d_model, max_len))

    def _get_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].detach()


class Transformer(nn.Module):
    def __init__(self, feature_size, num_layers, output_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_size)
        self.embedding = nn.Linear(feature_size, feature_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # Reshape output to match the desired shape
        output = self.decoder(output[-1]).transpose(1, 2)
        return output



class Segrnn(nn.Module):
    def __init__(self,seq_len,pred_len,enc_in,patch_len,d_model):
        super(Segrnn, self).__init__()
        #print(configs.enc_in,configs.d_model,configs.seq_len,configs.pred_len,configs.patch_len,configs.dropout)

        # remove this, the performance will be bad
        #self.lucky = nn.Embedding(144, 72)
        adj=np.load('/home/server/yixin/Flow-By-Flow-Prediction-main/dataset/abilene_adj.npy')
        self.GCN = GCN(adj, 12,144,16)
        self.seq_len = seq_len  # 720
        self.pred_len = pred_len#96
        self.enc_in = enc_in#7
        self.patch_len = patch_len#48
        self.d_model = d_model#512

        # self.seq_len = 10#720
        # self.pred_len = 4#96
        # self.enc_in = 144#7
        # self.patch_len = 2#48
        # self.d_model = 144#512


        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.linear1=nn.Linear(self.enc_in*2,self.enc_in)
        self.relu = nn.ReLU()
        #self.self_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=2)
        
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.patch_len, out_features=self.patch_len),
            nn.ReLU(),
            nn.Linear(in_features=self.patch_len, out_features=self.patch_len)
            # 添加更多的层...
        )


        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2)) #（2,256）  2,72
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))   #7,256   144,72

        self.dropout = nn.Dropout(0.5)#0.5
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)#512,48   144 2

    def forward(self, x):
        w = self.GCN(x)
        x=torch.cat(w,x,dim=-1)
        seq_last = x[:, -1:, :].detach()#使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变

        x = x - seq_last
        # print(x.shape)
        # torch.Size([256, 720, 7])
        # torch.Size([256, 1, 7])
        # torch.Size([256, 720, 7])

        B, L, C = x.shape#256,720,7
        N = self.seq_len // self.patch_len#15
        M = self.pred_len // self.patch_len#2
        W = self.patch_len#48
        d = self.d_model#512

        # print(x.shape)
        # print(N,M,W,d)#5,2,2,144




        xw = x.permute(0, 2, 1).reshape(B * C, N, -1) 
        # print('xw--------')
        # print(xw.shape) # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d      256*1,15,48  --      256*1, 15 ,512
        enc_in = self.relu(xd)

        #enc_in, _ = self.self_attention(enc_in, enc_in, enc_in)
        # print((self.gru(enc_in)[1]).shape)
#gru:第一个元素是 GRU 层的输出（通常是所有时间步的隐藏状态），第二个元素是 GRU 层在最后一个时间步的隐藏状态。
        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model)

        



        
        # .view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
#这里的repeat(1,1,m)表示重塑成（1，原本0维度的，一维度复制m次）
#这里的view里表示第一个维度是1，最后一个是512，中间-1表示自动调整
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d
#平铺和升维
        dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W         256*1*2,1,512    256*1*2,1,48

        #yw1=self.mlp(yw)
        

        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H        256,1,96  --256,96,1

        y = y + seq_last
        y=self.linear1(y)
        # print(y.shape)
        #([256, 96, 7])
        return y

# class MLP(nn.Module):
#     def __init__(self,input_dim,hide_dim,output_dim):
#         super(MLP, self).__init__()
#         self.fc1=nn.Linear(input_dim,hide_dim)
#         self.fc2=nn.Linear(hide_dim,output_dim)
#     def forward(self, x):
#         inter=F.relu(self.fc1(x))
#         out=self.fc2(inter)
#         return out



class GCN(nn.Module):
    def __init__(self, adj, input_dim,input_size,hidden):
        super(GCN, self).__init__()
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(
                torch.FloatTensor(adj))
        )  # 引入邻接矩阵
        self._num_nodes = adj.shape[0]
        self.input_dim = input_dim  # 要预测的句子的长度  12        44                                  # 时间序列的长度
        self.feature_dim = input_size  #144          self.input_size =                                  # 特征的个数
        self.out_dim = hidden  # 输出的隐层的长度  16
        self.weights = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x):
        # ba，seq,inputdim
        # 64,12,144
        batch_size = x.shape[0]

        x = x.transpose(0, 1)
        inputs = x.reshape((self._num_nodes, batch_size *
                            self.feature_dim ))
        # print(self.laplacian.shape)
        # print(inputs.shape)
        ax = self.laplacian @ inputs

        outputs = ax.reshape((self._num_nodes, batch_size,
                         self.feature_dim))
        outputs=outputs.transpose(0,1)
        return outputs


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian