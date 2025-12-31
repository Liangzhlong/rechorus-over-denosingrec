# -*- coding: UTF-8 -*-
import logging
import pandas as pd
import os
import numpy as np
from helpers.BaseReader import BaseReader
from utils import utils

class DenoisingReader(BaseReader):
    def _read_data(self):
        # Check if data exists in default path, if not try parent directory
        train_file = os.path.join(self.prefix, self.dataset, self.dataset + '.train.rating')
        if not os.path.exists(train_file):
            # Try ../data/
            if self.prefix == 'data/':
                alt_prefix = '../data/'
                alt_train_file = os.path.join(alt_prefix, self.dataset, self.dataset + '.train.rating')
                if os.path.exists(alt_train_file):
                    logging.info("Data not found in 'data/', switching to '../data/'")
                    self.prefix = alt_prefix
                    train_file = alt_train_file

        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        
        # Try to load .rating files (DenoisingRec format)
        if os.path.exists(train_file):
            logging.info("Detected .rating files, loading in DenoisingRec format.")
            self.data_df['train'] = pd.read_csv(
                train_file, 
                sep='\t', header=None, names=['user_id', 'item_id', 'noisy_or_not'], 
                usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32}
            )
            self.data_df['train']['time'] = 0 # Dummy time

            valid_file = os.path.join(self.prefix, self.dataset, self.dataset + '.valid.rating')
            if os.path.exists(valid_file):
                self.data_df['dev'] = pd.read_csv(
                    valid_file, 
                    sep='\t', header=None, names=['user_id', 'item_id', 'noisy_or_not'], 
                    usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32}
                )
                self.data_df['dev']['time'] = 0

                # Sample validation set for faster evaluation
                if len(self.data_df['dev']) > 10000:
                    logging.info(f"Sampling validation set: {len(self.data_df['dev'])} -> 10000 entries")
                    self.data_df['dev'] = self.data_df['dev'].sample(n=10000, random_state=2023)
            else:
                self.data_df['dev'] = pd.DataFrame(columns=['user_id', 'item_id', 'time', 'noisy_or_not'])

            test_file = os.path.join(self.prefix, self.dataset, self.dataset + '.test.negative')
            if os.path.exists(test_file):
                user_list, item_list, neg_items_list = [], [], []
                with open(test_file, 'r') as fd:
                    for line in fd:
                        arr = line.strip().split('\t')
                        try:
                            # Format: (user, item) \t neg1 \t neg2 ...
                            u, i = eval(arr[0])
                            neg_items = [int(x) for x in arr[1:]]
                        except:
                            # Fallback or other format
                            u, i = int(arr[0]), int(arr[1])
                            neg_items = [int(x) for x in arr[2:]]
                        user_list.append(u)
                        item_list.append(i)
                        neg_items_list.append(neg_items)
                
                self.data_df['test'] = pd.DataFrame({
                    'user_id': user_list,
                    'item_id': item_list,
                    'time': [0]*len(user_list),
                    'neg_items': neg_items_list
                })

                # Sample test set for faster evaluation
                if len(self.data_df['test']) > 10000:
                    logging.info(f"Sampling test set: {len(self.data_df['test'])} -> 10000 entries")
                    self.data_df['test'] = self.data_df['test'].sample(n=10000, random_state=2023)
            else:
                self.data_df['test'] = pd.DataFrame(columns=['user_id', 'item_id', 'time', 'neg_items'])

        else:
            # Fallback to standard ReChorus csv format
            logging.info("No .rating files found, falling back to standard CSV format.")
            super()._read_data()
            return

        # Post-processing similar to BaseReader
        self.all_df = pd.concat([self.data_df[key][['user_id', 'item_id']] for key in self.data_df if not self.data_df[key].empty])
        self.n_users = self.all_df['user_id'].max() + 1
        self.n_items = self.all_df['item_id'].max() + 1
        
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))

        # Initialize clicked sets
        self.train_clicked_set = dict()
        self.residual_clicked_set = dict()
        for key in ['train', 'dev', 'test']:
            if key not in self.data_df: continue
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)
