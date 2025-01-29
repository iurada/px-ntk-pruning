arch=resnet50
pretrain=mocov2
wrr=0.018
s=42
pruner=PX
split_nr=0

python main.py \
--experiment=${pruner} \
--experiment_name=ImageNet10/${arch}-${pretrain}/${wrr}/${pruner}/${split_nr}/run${s} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 2, 'aux_model': '${arch}_no_act', 'pretrain': '${pretrain}', 'pretrain_path': 'path/to/pth', 'freeze_bn_fit': True}" \
--dataset=ImageNet10 \
--dataset_args="{'root': '/data/datasets/ImageNet/', 'split_nr': ${split_nr}}" \
--arch=${arch} \
--batch_size=64 \
--epochs=90 \
--num_workers=4 \
--seed=${s} \
--pruner=${pruner} \
--grad_accum_steps=1 \
--task=transfer_classification
