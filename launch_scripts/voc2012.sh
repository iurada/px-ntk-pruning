arch=deeplabv3plus_resnet50
pretrain=dino
wrr=0.018
seed=42

python main.py \
--experiment=PX \
--experiment_name=VOC2012/${arch}/${wrr}/PX-${pretrain}/run${s} \
--experiment_args="{'weight_remaining_ratio': ${wrr}, 'rounds': 100, 'batch_limit': 53, 'aux_model': '${arch}_no_act', 'pretrain': '${pretrain}', 'pretrain_path': 'path/to/pth'}" \
--dataset=VOC2012 \
--dataset_args="{'root': 'data/VOC2012'}" \
--arch=${arch} \
--batch_size=4 \
--epochs=80 \
--num_workers=4 \
--seed=${s} \
--pruner=PX \
--grad_accum_steps=1 \
--task=segmentation
