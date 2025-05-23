import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.patch_length = configs.slice_len
        self.middle_size = configs.middle_len
        self.hidden_size = configs.hidden_len
        self.slice_stride = configs.slice_stride
        self.encoder_dropout = configs.encoder_dropout

        self.encoder = nn.Sequential(
            nn.Linear(self.patch_length, self.middle_size),
            nn.LeakyReLU(),
            nn.Dropout(self.encoder_dropout), 
            nn.Linear(self.middle_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.middle_size),
            nn.LeakyReLU(),
            nn.Dropout(self.encoder_dropout), 
            nn.Linear(self.middle_size, self.patch_length),
        )
        self.num_patches = int(self.seq_len/self.patch_length)
        self.num_patches_p = int(self.pred_len/self.patch_length)
        # self.fc_predictor = nn.Linear(self.hidden_size*self.num_patches, self.hidden_size *self.num_patches_p)
        self.fc_predictor = nn.Sequential(
            nn.Linear(self.hidden_size*self.num_patches, configs.d_ff),
            nn.LeakyReLU(),
            nn.Dropout(self.encoder_dropout), 
            nn.Linear(configs.d_ff, self.hidden_size *self.num_patches_p),
        )

        

    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        # x: [Batch, Input length, Channel] 
        num_patches = int(self.seq_len/self.patch_length)
        num_patches_p = int(self.pred_len/self.patch_length)
        x = x.permute(0,2,1)

        for_enc = x.unfold(-1,self.patch_length,self.slice_stride)
        # print(for_enc.shape,x.shape)
        slices = [for_enc[:,:, i,:] for i in range(for_enc.shape[-2])]
        slice = torch.cat(slices, dim=-1)
        encoded_slices = [self.encoder(patch) for patch in slices]
        encoded_slice = torch.cat(encoded_slices, dim=-1)
        decoded_slices = [self.decoder(encoded_patch) for encoded_patch in encoded_slices]
        decoded_slice = torch.cat(decoded_slices, dim=-1)

        data = x.chunk(num_patches, dim=-1)
        #TODO
        encoded_patche = [self.encoder(patch) for patch in data]
        encoded_patches = torch.cat(encoded_patche, dim=-1)
        prediction = self.fc_predictor(encoded_patches)
        prediction_patchs = prediction.chunk(num_patches_p, dim=-1)
        decoded_prediction_patches = [self.decoder(prediction_patch) for prediction_patch in prediction_patchs]
        predictions = torch.cat(decoded_prediction_patches, dim=-1)

        return predictions.permute(0,2,1)+ seq_last,slice,decoded_slice # [Batch, Output length, Channel]