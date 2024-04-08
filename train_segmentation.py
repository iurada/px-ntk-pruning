import torch
import torch.nn as nn
import torch.jit
import os
import logging
from tqdm import tqdm
import pandas as pd
import wandb
import copy
import math

from lib.pruners import Rand, SNIP, GraSP, SynFlow, SynFlowL2, NTKSAP, Mag, PX
from lib.generator import masked_parameters, parameters, prunable

from lib.models.segmentation.deeplabv3 import deeplabv3plus_resnet50

import lib.metrics as metrics
import lib.layers as layers

import datasets.VOC2012.dataset as VOC2012

from globals import CONFIG

def kaiming_normal_init(model):
    for m in model.modules():
        if isinstance(m, (layers.Conv2d, layers.Linear)):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, (layers.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Experiment:

    def __init__(self):
        assert CONFIG.dataset in ['VOC2012'], f'"{CONFIG.dataset}" dataset not available!'
        assert CONFIG.pruner in ['Dense', 'Rand', 'SNIP', 'GraSP', 'SynFlow', 'SynFlowL2',
                                 'NTKSAP', 'Mag', 'PX', 'IMP'], f'"{CONFIG.pruner}" pruning strategy not available!'
        assert CONFIG.arch in ['deeplabv3plus_resnet50'], f'"{CONFIG.arch}" architecture not available!'


        # Load data
        self.data = eval(CONFIG.dataset).load_data()
        
        self.model = eval(CONFIG.arch)(num_classes=CONFIG.num_classes)
        
        self.model = self.model.to(CONFIG.device)

        
        # Optimizers, schedulers & losses
        self._init_optimizers()
        
        # Meters
        self._init_meters()


        # Pruning strategy
        if CONFIG.pruner in ['Rand', 'Mag', 'SNIP', 'GraSP', 'SynFlow', 'SynFlowL2', 'NTKSAP', 'PX']: #! Pruning-at-init         
            ROUNDS = CONFIG.experiment_args['rounds']
            sparsity = CONFIG.experiment_args['weight_remaining_ratio']

            self.pruner = eval(CONFIG.pruner)(masked_parameters(self.model))

            if CONFIG.pruner in ['SynFlow', 'SynFlowL2', 'PX']:
                self.model.eval()
            
            for round in range(ROUNDS):
                sparse = sparsity**((round + 1) / ROUNDS)

                self.pruner.score(self.model, self.loss_fn, self.data['train'], CONFIG.device)

                self.pruner.mask(sparse, 'global')
                remaining_params, total_params = self.pruner.stats()
                logging.info(f'{int(remaining_params)} / {int(total_params)} | {remaining_params / total_params}')

        elif CONFIG.pruner in ['IMP']: #! Iterative pruning
            
            ROUNDS = CONFIG.experiment_args['rounds']
            sparsity = CONFIG.experiment_args['weight_remaining_ratio']

            self.pruner = eval(CONFIG.pruner)(masked_parameters(self.model))
            
            initial_state = copy.deepcopy(self.model.state_dict())

            for round in range(ROUNDS):
                sparse = sparsity**((round + 1) / ROUNDS)

                self.model = self.fit(save_checkpoint=False)
                self.pruner.score(self.model, self.loss_fn, self.data['train'], CONFIG.device)
                self.pruner.mask(sparse, 'global')
                remaining_params, total_params = self.pruner.stats()
                logging.info(f'{int(remaining_params)} / {int(total_params)} | {remaining_params / total_params}')

                db = {}
                for k in initial_state:
                    if 'mask' not in k:
                        db[k] = initial_state[k]
                self.model.load_state_dict(db)

                self._init_optimizers()
                self._init_meters()
            logging.info('Retraining after Iterative Pruning...')

        if CONFIG.pruner != 'Dense':

            if CONFIG.reshuffle_mask:
                self.pruner.shuffle()
        
            if CONFIG.reinit_weights:
                kaiming_normal_init(self.model)

            if CONFIG.pruner:
                self.model.eval()
                prune_result = metrics.summary(self.model, 
                                    self.pruner.scores,
                                    metrics.flop(self.model, CONFIG.data_input_size, CONFIG.device),
                                    lambda p: prunable(p, False, False))
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    logging.info(prune_result)
    

    def _init_optimizers(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        if CONFIG.dataset in ['VOC2012']:
            self.optimizer = torch.optim.SGD(params=[
                {'params': parameters(self.model.backbone), 'lr': 0.001},
                {'params': parameters(self.model.classifier), 'lr': 0.01},
            ], lr=0.01, momentum=0.9, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 60], gamma=0.1)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def _init_meters(self):
        self.meter = metrics.StreamSegMetrics(CONFIG.num_classes)

    def fit(self, save_checkpoint=True):
        best_model = None

        # Load Checkpoint
        current_epoch = 0
        if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
            ckpt = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
            current_epoch = ckpt['current_epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])

        # Train loop
        for epoch in range(current_epoch, CONFIG.epochs):
            self.model.train()

            # Train epoch
            total_loss = 0
            for batch_idx, data_tuple in tqdm(enumerate(self.data['train'])):

                if CONFIG.dataset in ['VOC2012']:
                    x, y = data_tuple
                    x = x.to(CONFIG.device)
                    y = y.to(CONFIG.device).long()

                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y) / CONFIG.grad_accum_steps
                
                self.scaler.scale(loss).backward()

                if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(self.data['train'])):
                    self.scaler.step(self.optimizer)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()

                    if CONFIG.use_wandb:
                        wandb.log({'train_loss': loss.item()})
            
            self.scheduler.step()

            # Validation
            logging.info(f'[VAL @ Epoch={epoch}]')

            if CONFIG.dataset in ['VOC2012']:
                metrics = self.evaluate(self.data['test'])

                # Model selection & State management
                if save_checkpoint:
                    ckpt = {}
                    ckpt['current_epoch'] = epoch + 1
                    ckpt['model'] = self.model.state_dict()
                    ckpt['optimizer'] = self.optimizer.state_dict()
                    ckpt['scheduler'] = self.scheduler.state_dict()
                    torch.save(ckpt, os.path.join('record', CONFIG.experiment_name, 'last.pth'))
                else:
                    best_model = copy.deepcopy(self.model)

        return best_model


    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        # Reset meters
        if CONFIG.dataset in ['VOC2012']:
            self.meter.reset()

        # Validation loop
        loss = [0.0, 0]
        for data_tuple in tqdm(loader):

            if CONFIG.dataset in ['VOC2012']:
                x, y = data_tuple
                x = x.to(CONFIG.device)
                y = y.to(CONFIG.device).long()

            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                logits = self.model(x)
                loss[0] += self.loss_fn(logits, y).item()
                loss[1] += x.size(0)
                _, preds = torch.max(logits, 1)
                
            if CONFIG.dataset in ['VOC2012']:
                self.meter.update(y.cpu().numpy(), preds.cpu().numpy())

        # Compute metrics
        if CONFIG.dataset in ['VOC2012']:
            miou = self.meter.get_results()['Mean IoU']

            metrics = {
                'Mean IoU': miou,
                'Loss': loss[0] / loss[1]
            }

        logging.info(metrics)
        return metrics
    