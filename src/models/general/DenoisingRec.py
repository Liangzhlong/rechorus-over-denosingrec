# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from models.BaseModel import GeneralModel

class DenoisingRec(GeneralModel):
    reader = 'DenoisingReader'
    runner = 'Denoising_Runner'
    extra_log_args = ['factor_num', 'num_layers', 'drop_rate', 'num_gradual', 'exponent']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--factor_num', type=int, default=32,
                            help='predictive factors numbers in the model')
        parser.add_argument('--num_layers', type=int, default=3,
                            help='number of layers in MLP model')
        parser.add_argument('--drop_rate', type=float, default=0.2,
                            help='drop rate')
        parser.add_argument('--num_gradual', type=int, default=30000,
                            help='how many epochs to linearly increase drop_rate')
        parser.add_argument('--exponent', type=float, default=1,
                            help='exponent of the drop rate {0.5, 1, 2}')
        parser.add_argument('--model_type', type=str, default='NeuMF-end',
                            help='model used for training. options: GMF, NeuMF-end, MLP')
        # parser.add_argument('--model_type', type=str, default='GMF',
        #                     help='model used for training. options: GMF, NeuMF-end, MLP')
        parser.add_argument('--alpha', type=float, default=0.2,
                            help='alpha for the drop rate')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.factor_num = args.factor_num
        self.num_layers = args.num_layers
        self.model_type = args.model_type
        self.alpha = args.alpha
        self.drop_rate = args.drop_rate
        self.current_drop_rate = 0.0
        
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        # Ported from DenoisingRec/model.py NCF class
        user_num = self.user_num
        item_num = self.item_num
        factor_num = self.factor_num
        num_layers = self.num_layers
        dropout = self.dropout

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
                user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
                item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model_type in ['MLP', 'GMF']:
            predict_size = factor_num 
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

    def forward(self, feed_dict):
        user = feed_dict['user_id']
        item = feed_dict['item_id']
        
        # Flatten input if necessary (ReChorus passes [batch, 1+neg])
        # But NCF expects [batch]
        # We will process flattened and then reshape if needed, or just return flattened
        
        original_shape = item.shape
        user = user.unsqueeze(-1).expand_as(item) # Broadcast user to match item shape
        
        user_flat = user.reshape(-1)
        item_flat = item.reshape(-1)

        if not self.model_type == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user_flat)
            embed_item_GMF = self.embed_item_GMF(item_flat)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model_type == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user_flat)
            embed_item_MLP = self.embed_item_MLP(item_flat)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model_type == 'GMF':
            concat = output_GMF
        elif self.model_type == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat).view(-1)
        
        # Reshape back to [batch, 1+neg] for evaluation compatibility
        prediction_reshaped = prediction.view(original_shape)
        
        return {'prediction': prediction_reshaped, 'label': feed_dict['label']}

    def loss(self, out_dict):
        # prediction = out_dict['prediction'].reshape(-1)
        # label = out_dict['label'].float().reshape(-1)
        
        # return loss_function(prediction, label, self.current_drop_rate)
        pass

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            # Override to provide labels suitable for BCE
            user_id = self.data['user_id'][index]
            target_item = self.data['item_id'][index]
            
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            elif 'neg_items' in self.data and len(self.data['neg_items'][index]) > 0:
                neg_items = self.data['neg_items'][index]
            else:
                if self.phase != 'train':
                    neg_items = np.random.randint(1, self.corpus.n_items, size=self.model.num_neg)
                    clicked_set = self.corpus.train_clicked_set[user_id]
                    for j in range(len(neg_items)):
                        while neg_items[j] in clicked_set:
                            neg_items[j] = np.random.randint(1, self.corpus.n_items)
                else:
                    if 'neg_items' in self.data:
                        neg_items = self.data['neg_items'][index]
                    else:
                        neg_items = np.array([], dtype=int)
            
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            labels = np.concatenate([[1], [0] * len(neg_items)]).astype(int)
            
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids,
                'label': labels
            }
            return feed_dict
