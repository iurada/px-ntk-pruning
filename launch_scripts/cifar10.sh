arch=resnet20
wrr=0.018
seed=42

python main.py \
--experiment=PX \
--experiment_name=CIFAR10/${arch}/${wrr}/PX/run${s} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 2, 'aux_model': '${arch}_no_act'}" \
--dataset=CIFAR10 \
--dataset_args="{'root': 'data/CIFAR10'}" \
--arch=${arch} \
--batch_size=128 \
--epochs=160 \
--num_workers=4 \
--seed=${s} \
--pruner=PX \
--grad_accum_steps=1
