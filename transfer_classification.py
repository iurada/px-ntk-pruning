import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import os
import logging
from torchmetrics import Accuracy
from tqdm import tqdm
import pandas as pd
import wandb
import copy

from lib.pruners import Rand, SNIP, GraSP, SynFlow, SynFlowL2, NTKSAP, Mag, PX, PXact
from lib.generator import masked_parameters, parameters, prunable

from lib.models.imagenet_resnet import resnet50
from lib.models.heads import get_ridge_classification_head

import lib.metrics as metrics
import lib.layers as layers

import datasets.ImageNetK.dataset as ImageNet10

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
        assert CONFIG.dataset in ['ImageNet10'], f'"{CONFIG.dataset}" dataset not available!'
        assert CONFIG.pruner in ['Dense', 'Rand', 'SNIP', 'GraSP', 'SynFlow', 'SynFlowL2',
                                 'NTKSAP', 'Mag', 'PX', 'IMP', 'PXact'], f'"{CONFIG.pruner}" pruning strategy not available!'
        assert CONFIG.arch in ['resnet50'], f'"{CONFIG.arch}" architecture not available!'


        # Load data
        self.data = eval(CONFIG.dataset).load_data(CONFIG.dataset_args['split_nr'])

        # Initialize model
        self.model = eval(CONFIG.arch)(num_classes=CONFIG.num_classes)
        self.model = self.model.to(CONFIG.device)

        # Fit ridge classifier
        self.model.fc = nn.Identity()
        self.model.fc = get_ridge_classification_head(self.model, 
                                                      self.data['test'], 
                                                      CONFIG.num_classes,
                                                      CONFIG.device)
        self.model.fc.requires_grad_(False)
        
        # Optimizers, schedulers & losses
        self._init_optimizers()

        # Meters
        self._init_meters()

        # Pruning strategy
        if CONFIG.pruner in ['Rand', 'Mag', 'SNIP', 'GraSP', 'SynFlow', 'SynFlowL2', 'NTKSAP', 'PX', 'PXact']: # Pruning-at-init         
            ROUNDS = CONFIG.experiment_args['rounds']
            sparsity = CONFIG.experiment_args['weight_remaining_ratio']

            self.pruner = eval(CONFIG.pruner)(masked_parameters(self.model))

            if CONFIG.pruner in ['SynFlow', 'SynFlowL2', 'PX', 'PXact']:
                self.model.eval()
            
            for round in range(ROUNDS):
                sparse = sparsity**((round + 1) / ROUNDS)

                self.pruner.score(self.model, self.loss_fn, self.data['train'], CONFIG.device)

                self.pruner.mask(sparse, 'global')
                remaining_params, total_params = self.pruner.stats()
                logging.info(f'{int(remaining_params)} / {int(total_params)} | {remaining_params / total_params}')

        elif CONFIG.pruner in ['IMP']: # Iterative pruning
            
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

        self.can_eval_transfer = True
        # Zero-shot Eval
        logging.info('Zero-shot Post-pruning Transfer')
        self.evaluate_transfer()


    def _init_optimizers(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        if CONFIG.dataset == 'ImageNet10':
            # imagenet-pt: lr=5e-5
            self.optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=5e-5, momentum=0.9, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=0.1)
            self.loss_fn = lambda input, target: F.mse_loss(input, F.one_hot(target, num_classes=CONFIG.num_classes).float())
            self.loss_fn_eval = lambda input, target: F.mse_loss(input, F.one_hot(target, num_classes=CONFIG.num_classes).float(), reduction='sum')


    def _init_meters(self):
        if CONFIG.dataset in ['ImageNet10']:
            self.acc_tot = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
            self.acc_tot = self.acc_tot.to(CONFIG.device)


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
            if CONFIG.experiment_args['freeze_bn_fit']:
                self.model.eval()

            # Train epoch
            for batch_idx, data_tuple in tqdm(enumerate(self.data['train'])):

                if CONFIG.dataset in ['ImageNet10']:
                    x, y = data_tuple
                    x = x.to(CONFIG.device)
                    y = y.to(CONFIG.device)

                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    logits = self.model(x).squeeze()
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

            if CONFIG.dataset in ['ImageNet10']:
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

        if self.can_eval_transfer:
            logging.info('Post-retraining Transfer')
            self.evaluate_transfer()

        return best_model


    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        # Reset meters
        if CONFIG.dataset in ['ImageNet10']:
            self.acc_tot.reset()

        # Validation loop
        loss = [0.0, 0]
        for data_tuple in tqdm(loader):

            if CONFIG.dataset in ['ImageNet10']:
                x, y = data_tuple
                x = x.to(CONFIG.device)
                y = y.to(CONFIG.device)

            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                logits = self.model(x).squeeze()
                loss[0] += self.loss_fn_eval(logits, y).item()
                loss[1] += x.size(0)
                
            if CONFIG.dataset in ['ImageNet10']:
                self.acc_tot.update(logits, y)

        # Compute metrics
        if CONFIG.dataset in ['ImageNet10']:
            acc_tot = self.acc_tot.compute()

            metrics = {
                'Acc': acc_tot.item(),
                'Loss': loss[0] / loss[1]
            }

        logging.info(metrics)
        return metrics
    

    @torch.no_grad()
    def evaluate_transfer(self):
        self.model.eval()
        
        num_splits = {
            'ImageNet10': 10
        }

        for split_nr in range(num_splits[CONFIG.dataset]):

            # Load data
            data = eval(CONFIG.dataset).load_data(split_nr)

            # Initialize model
            clean_model = eval(CONFIG.arch)(num_classes=CONFIG.num_classes)
            clean_model.fc = nn.Identity()
            clean_model = clean_model.to(CONFIG.device)
            
            # Fit ridge classifier
            copy_model = copy.deepcopy(self.model)
            copy_model.fc = get_ridge_classification_head(clean_model, 
                                                          data['test'], 
                                                          CONFIG.num_classes,
                                                          CONFIG.device)
            clean_model = clean_model.to('cpu')
            del clean_model
            torch.cuda.empty_cache()

            # Reset meters
            if CONFIG.dataset in ['ImageNet10']:
                self.acc_tot.reset()

            # Validation loop
            loss = [0.0, 0]
            for data_tuple in tqdm(data['test']):

                if CONFIG.dataset in ['ImageNet10']:
                    x, y = data_tuple
                    x = x.to(CONFIG.device)
                    y = y.to(CONFIG.device)

                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    logits = copy_model(x).squeeze()
                    loss[0] += self.loss_fn_eval(logits, y).item()
                    loss[1] += x.size(0)
                    
                if CONFIG.dataset in ['ImageNet10']:
                    self.acc_tot.update(logits, y)

            # Compute metrics
            if CONFIG.dataset in ['ImageNet10']:
                acc_tot = self.acc_tot.compute()

                metrics = {
                    'Split': split_nr,
                    'Acc': acc_tot.item(),
                    'Loss': loss[0] / loss[1]
                }

            logging.info(metrics)

            copy_model = copy_model.to('cpu')
            del copy_model
            torch.cuda.empty_cache()

        return metrics
