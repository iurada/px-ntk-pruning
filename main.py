import os
import logging
import warnings
import torch
import random
import numpy as np
import torch.backends.cudnn
import wandb
from parse_args import parse_arguments

from train_classification import Experiment
from train_segmentation import Experiment as SegmentationExperiment
from transfer_classification import Experiment as TransferClassificationExperiment

from globals import CONFIG

def main():
    # Select whether to use deterministic behavior
    if not CONFIG.use_nondeterministic:
        torch.manual_seed(CONFIG.seed)
        random.seed(CONFIG.seed)
        np.random.seed(CONFIG.seed)
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

    if CONFIG.task == 'segmentation':
        experiment = SegmentationExperiment()
    elif CONFIG.task == 'transfer_classification':
        experiment = TransferClassificationExperiment()
    else:
        experiment = Experiment()

    experiment.fit(save_checkpoint=CONFIG.save_checkpoint)
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    args = parse_arguments()
    CONFIG.update(vars(args))
    
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )
    
    if CONFIG.use_wandb:
        wandb.init(
            project='foresight-pruning',
            name=CONFIG.experiment_name,
            config=CONFIG
        )

    main()