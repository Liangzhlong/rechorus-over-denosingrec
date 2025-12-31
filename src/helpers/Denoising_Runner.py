# -*- coding: UTF-8 -*-

import logging
import numpy as np
from helpers.BaseRunner import BaseRunner
from models.BaseModel import BaseModel

class Denoising_Runner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self.drop_rate = args.drop_rate
        self.num_gradual = args.num_gradual
        self.exponent = args.exponent
        self.iteration_count = 0

    def _drop_rate_schedule(self, iteration):
        drop_rate = np.linspace(0, self.drop_rate**self.exponent, self.num_gradual)
        if iteration < self.num_gradual:
            return drop_rate[iteration]
        else:
            return self.drop_rate

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()

        model.train()
        loss_lst = list()
        
        # Use BaseRunner's DataLoader creation logic
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from utils import utils
        import torch

        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        
        # pbar = tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), dynamic_ncols=True, mininterval=1)
        for batch in dl:
            batch = utils.batch_to_gpu(batch, model.device)

            # Update drop rate
            self.iteration_count += 1
            current_drop_rate = self._drop_rate_schedule(self.iteration_count)
            model.current_drop_rate = current_drop_rate

            # Shuffle items (optional, but good for BPR, maybe less critical for BCE but harmless)
            item_ids = batch['item_id']
            indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)						
            batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]
            # Also shuffle labels!
            batch['label'] = batch['label'][torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

            model.optimizer.zero_grad()
            out_dict = model(batch)

            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
            
        return np.mean(loss_lst).item()

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> dict:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(dataset)
        
        # If predictions are flattened (e.g. from T_CE model), reshape them
        if predictions.ndim == 1:
            n_items = dataset.corpus.n_items
            predictions = predictions.reshape(-1, n_items)
            
        return self.evaluate_method(predictions, topks, metrics)

