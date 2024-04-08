arch=tinyimagenet_resnet18
wrr=0.018
seed=42

python main.py \
--experiment=PX \
--experiment_name=TinyImageNet/${arch}/${wrr}/PX/run${s} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 8, 'aux_model': '${arch}_no_act'}" \
--dataset=TinyImageNet \
--dataset_args="{'root': 'data/TinyImageNet'}" \
--arch=${arch} \
--batch_size=256 \
--epochs=200 \
--num_workers=4 \
--seed=${s} \
--pruner=PX \
--grad_accum_steps=1
