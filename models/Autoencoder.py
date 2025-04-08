import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len


        hidden_dim = configs.d_model
        e_num_layers=configs.e_layers
        input_dim = configs.enc_in
        d_num_layers=configs.d_layers
        latent_dim=configs.d_ff
        # Encoder
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=e_num_layers, batch_first=True)
        self.fc_encoder = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                               num_layers=d_num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)



        if self.task_name == 'anomaly_detection':
            # self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.projection = nn.Linear(128, configs.c_out, bias=True)

    def anomaly_detection(self, x_enc):
        _, (h_n, _) = self.encoder(x_enc)
        latent = self.fc_encoder(h_n[-1])  # Use the last hidden state from LSTM encoder

        # Decoding
        latent_rep = latent.unsqueeze(1).repeat(1, x_enc.size(1), 1)  # Repeat latent vector
        latent_rep = self.fc_decoder(latent_rep)
        decoded, _ = self.decoder(latent_rep)
        reconstructed = self.output_layer(decoded)
        dec_out=reconstructed
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None