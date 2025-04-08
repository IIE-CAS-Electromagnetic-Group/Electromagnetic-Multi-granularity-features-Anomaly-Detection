import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        """
        LSTM model for time series prediction.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each LSTM layer.
            output_dim (int): Number of output features.
            num_layers (int): Number of LSTM layers.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # self.label_len = configs.label_len
        # self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        input_dim=configs.enc_in
        hidden_dim = configs.d_model
        num_layers= configs.e_layers
        self.output_dim=configs.c_out
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim*configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding

        out, _ = self.lstm(x_enc)
        out = self.fc(out[:, -1, :])  # (batch_size, output_dim * forecast_steps)
        return out.view(out.size(0), -1, self.output_dim)  # 转换为 (batch_size, forecast_steps, output_dim)


        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=None)
        #
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        # return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None


