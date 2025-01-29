arch=resnet50
pretrain=imagenet
wrr=0.018
s=42
pruner=PX

python main.py \
--experiment=${pruner} \
--experiment_name=CIFAR100/${arch}/${wrr}/${pruner}-${pretrain}/run${s} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 4, 'aux_model': '${arch}_no_act', 'pretrain': '${pretrain}', 'pretrain_path': 'path/to/pth'}" \
--dataset=CIFAR100 \
--dataset_args="{'root': 'data/CIFAR100'}" \
--arch=${arch} \
--batch_size=256 \
--epochs=182 \
--num_workers=4 \
--seed=${s} \
--pruner=${pruner} \
--grad_accum_steps=1
