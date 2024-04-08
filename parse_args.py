from argparse import ArgumentParser

def _clear_args(parsed_args):
    parsed_args.experiment_args = eval(parsed_args.experiment_args)
    parsed_args.dataset_args = eval(parsed_args.dataset_args)
    return parsed_args

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--use_nondeterministic', action='store_true', help='Whether to run non-deterministic experiments')
    parser.add_argument('--seed', type=int, default=42, help='Seed used for deterministic behavior')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb to log losses')
    parser.add_argument('--cpu', action='store_true', help='Whether to force the usage of CPU')
    parser.add_argument('--save_checkpoint', action='store_true', help='If set, it will store the last checkpoint at each epoch during training.')

    parser.add_argument('--reshuffle_mask', action='store_true', help='Whether to reshuffle pruning mask.')
    parser.add_argument('--reinit_weights', action='store_true', help='Whether to reinitialize weights after pruning.')

    parser.add_argument('--experiment', type=str, default='Baseline')
    parser.add_argument('--experiment_name', type=str, default='Baseline')
    parser.add_argument('--experiment_args', type=str, default='{}')
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--dataset_args', type=str, default='{}')
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--pruner', type=str, default='Dense')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=1)

    return _clear_args(parser.parse_args())