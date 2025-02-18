import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from globals import CONFIG
import lib.generator as generator
import lib.layers as layers
from lib.generator import masked_parameters
from torch.utils.data import DataLoader, Subset

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
 
    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask.copy_(mask.reshape(-1)[perm].reshape(shape))

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, data_tuple in enumerate(dataloader):
            if batch_idx == CONFIG.experiment_args['batch_limit']: break
            
            if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
                data, y = data_tuple
                data, y = data.to(device), y.to(device)
        
                output = model(data).squeeze()
        
                loss(output, y).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            if m.grad is None:
                self.scores[id(p)] = torch.zeros_like(p)
                m.requires_grad = False
                p.requires_grad = True
                if p.grad is not None:
                    p.grad.data.zero_()
                continue
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, data_tuple in enumerate(dataloader):
            if batch_idx == CONFIG.experiment_args['batch_limit']: break
        
            if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
                data, y = data_tuple
                data, y = data.to(device), y.to(device)
        
                output = model(data).squeeze() / self.temp
                L = loss(output, y)
        
            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads
        
        # second gradient vector with computational graph
        for batch_idx, data_tuple in enumerate(dataloader):
            if batch_idx == CONFIG.experiment_args['batch_limit']: break
        
            if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
                data, y = data_tuple
                
                data, y = data.to(device), y.to(device)
        
                output = model(data).squeeze() / self.temp
                L = loss(output, y)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            
            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            if p.grad is None:
                self.scores[id(p)] = torch.zeros_like(p)
                continue
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        data_tuple = next(iter(dataloader))
        if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
            data, y = data_tuple

        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)

        output = model(input)

        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            if p.grad is None:
                self.scores[id(p)] = torch.zeros_like(p)
                continue
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

class SynFlowL2(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlowL2, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        lin_model = copy.deepcopy(model)
        lin_model.eval()

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.pow_(2)
            return signs
        
        linearize(lin_model)

        data_tuple = next(iter(dataloader))
        if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
            data, y = data_tuple

        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)

        output = lin_model(input)

        torch.sum(output).backward()
        
        for (_, p), (_, lp) in zip(self.masked_parameters, masked_parameters(lin_model)):
            if lp.grad is None:
                self.scores[id(p)] = torch.zeros_like(p)
                continue
            self.scores[id(p)] = torch.clone(lp.grad * p).detach().abs_()

def reset_BN(net):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.reset_running_stats()

def copy_BN(from_model, to_model):
    with torch.no_grad():
        for fm, tm in zip(from_model.modules(), to_model.modules()):
            if isinstance(tm, (nn.BatchNorm2d, nn.GroupNorm)):
                tm.running_mean.copy_(fm.running_mean)
                tm.running_var.copy_(fm.running_var)
                tm.num_batches_tracked.copy_(fm.num_batches_tracked)

class NTKSAP(Pruner):
    def __init__(self, masked_parameters, *args, **kwargs):
        super(NTKSAP, self).__init__(masked_parameters)
        self.epsilon = CONFIG.experiment_args['ntksap_epsilon']
        self.R = CONFIG.experiment_args['ntksap_R']

    def score(self, model, loss, dataloader, device, *args, **kwargs):
        init_model = copy.deepcopy(model)

        def perturb(model_orig, model_copy):
            with torch.no_grad():
                for (m_orig, p_orig), (m_copy, p_copy) in zip(generator.masked_parameters(model_orig), generator.masked_parameters(model_copy)):
                    p_copy.data = p_orig.data + self.epsilon * torch.randn_like(p_orig.data)

            for module, module_mod in zip(model_orig.modules(), model_copy.modules()):
                if isinstance(module, nn.BatchNorm2d):
                    with torch.no_grad():
                        module_mod.running_mean = module.running_mean
                        module_mod.running_var = module.running_var
                        module_mod.num_batches_tracked = module.num_batches_tracked

        for m, p in self.masked_parameters:
            m.requires_grad = True
            p.requires_grad = False

        # Copy a same model
        model_mod = copy.deepcopy(model)

        # Set model mod to evaluation mode
        model_mod.eval()

        # Make two models share the same weight masks
        for module, module_mod in zip(model.modules(), model_mod.modules()):
            if hasattr(module, 'weight_mask'):
                module_mod.weight_mask = module.weight_mask
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 1.0
                module_mod.momentum = 1.0

        for _ in range(self.R):
            for index, data_tuple in tqdm(enumerate(dataloader)):
                if index == CONFIG.experiment_args['batch_limit']: break

                if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
                    data, y = data_tuple

                if isinstance(model, nn.DataParallel):
                    model.module._initialize_weights()
                else:
                    # model._initialize_weights()
                    for m in model.modules():
                        if isinstance(m, (layers.Conv2d, layers.Linear)):
                            nn.init.kaiming_normal_(m.weight)
                        elif isinstance(m, (layers.BatchNorm2d, nn.GroupNorm)):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)
                input = torch.randn_like(data).to(device)

                reset_BN(model)
                with torch.no_grad():
                    output_orig = model(input)

                model.eval()
                # Compute the true graph using eval mode
                output_orig = model(input)
                perturb(model, model_mod)
                output_mod = model_mod(input)
                jac_approx = (torch.norm(output_orig-output_mod,dim=-1)**2).sum()
                jac_approx.backward()
                model.train()

        for m, p in self.masked_parameters:
            if m.grad is None:
                self.scores[id(p)] = torch.zeros_like(p)
                m.requires_grad = False
                p.requires_grad = True
                continue
            self.scores[id(p)] = torch.clone(m.grad * (m!=0)).detach().abs_()
            m.grad.data.zero_()
            m.requires_grad = False
            p.requires_grad = True

        # Reset momentum of BatchNorm2d
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.1

        del model_mod
        model = init_model


class PX(Pruner):
    def __init__(self, masked_parameters):
        super(PX, self).__init__(masked_parameters)
        self.orig_relu = F.relu
        self.orig_leakyrelu = F.leaky_relu

    def get_balanced_subset(self, dataset, num_samples_per_class):
        class_labels = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
        class_to_indices = {}
        for idx, label in enumerate(class_labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        sampled_indices = []
        for label in class_to_indices:
            indices = class_to_indices[label]
            if len(indices) >= num_samples_per_class:
                sampled_indices.extend(indices[:num_samples_per_class])
            else:
                raise ValueError(f"Class {label} has fewer than {num_samples_per_class} samples.")
        return Subset(dataset, sampled_indices)

    def score(self, model, loss, dataloader, device):

        if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet']:
            balanced_dset = DataLoader(self.get_balanced_subset(dataloader.dataset, 
                                    CONFIG.experiment_args['examples_per_class']),
                                    batch_size=CONFIG.batch_size, shuffle=False)
        else:
            balanced_dset = dataloader

        def hook_activation(input, inplace=False): # overrides F.relu
            self.activation_maps.append(input.clone().detach())
            return self.orig_relu(input, inplace)

        def apply_activation(input, inplace=False): # overrides F.relu
            map = torch.where(self.activation_maps.pop(0) > 0, 1.0, 0.0)
            return input * map

        # Path Kernel Estimator
        pk_model = copy.deepcopy(model).train().to(device)
        with torch.no_grad():
            skip_layers = ['weight_mask', 'bias_mask', 'bn', 'fc']
            for name, param in pk_model.state_dict().items():
                if any([n in name for n in skip_layers]):
                    continue
                param.pow_(2)

        # Path Activation Matrix Estimator
        jvf_model = copy.deepcopy(model).train().to(device)
        with torch.no_grad():
            skip_layers = ['weight_mask', 'bias_mask', 'bn', 'fc']
            for name, param in jvf_model.state_dict().items():
                if any([n in name for n in skip_layers]):
                    continue
                param.fill_(1.0)

        # Auxiliary Activation Maps Extractor
        activation_model = copy.deepcopy(model).eval().to(device)

        for data_tuple in tqdm(balanced_dset):     
            if CONFIG.dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'VOC2012', 'ImageNet10']:
                data, y = data_tuple
            data = data.to(device)

            with torch.no_grad():
                F.relu = hook_activation
                F.leaky_relu = hook_activation
                self.activation_maps = []
                activation_model(data)

                F.relu = apply_activation
                z1 = jvf_model(data.pow(2))
            
            F.relu = lambda input, *args, **kwargs: input
            F.leaky_relu = lambda input, *args, **kwargs: input
            z2 = pk_model(torch.ones_like(data))
            (z1 * z2).sum().backward()
            
            F.relu = self.orig_relu
            F.leaky_relu = self.orig_leakyrelu
        
        with torch.no_grad():
            for (m, p), (_, p1), (_, p2) in zip(self.masked_parameters, masked_parameters(pk_model), masked_parameters(jvf_model)):
                if p1.grad is None:
                    self.scores[id(p)] = torch.zeros_like(p)
                    continue
                self.scores[id(p)] = torch.clone(p1.grad.abs() * p.pow(2))
