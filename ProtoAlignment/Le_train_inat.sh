#!/bin/bash

set -x
set -e


others=(
# "--backbone resnet18 --optimizer=adam --learning_rate=1e-5" # --notes=using_comm_name_filtering_on_cub"
# "--dataset inat17"
# "--dataset inat21"
# "--dataset inat21mini"
""
)

others1=(
# "--skip_target_instance_bank --conv_lr_ratio=0.1 --momentum_target_centroids=0.1 --batch=128"
# "--skip_target_instance_bank --conv_lr_ratio=0.01 --batch 128  --num_workers=4 --load_inat_in_memory --notes=update_centroids_w_mmult"

# "--skip_target_instance_bank --conv_lr_ratio=0.1 --batch 128  --num_workers=8 --load_inat_in_memory --notes=collate_in_main_process --momentum_target_centroids=0.9" # 3 cores x 64 each
# "--skip_target_instance_bank --conv_lr_ratio=0.1 --batch 128  --num_workers=8 --notes=update_centroids_w_mmult --momentum_target_centroids=0.9"
# "--skip_target_instance_bank --conv_lr_ratio=0.1 --batch 128  --num_workers=8 --notes=update_centroids_w_mmult --momentum_target_centroids=0.9 --momentum_target_centroids=0.9"

# "--skip_target_instance_bank --conv_lr_ratio=0.1 --batch 128  --num_workers=8 --notes=update_centroids_w_mmult --momentum_source_instances=0.5 --momentum_target_centroids=0.5" # 3 cores x 32000  each
# "--skip_target_instance_bank --conv_lr_ratio=1.0 --batch 128  --num_workers=8 --notes=update_centroids_w_mmult --momentum_source_instances=0.5 --momentum_target_centroids=0.5"

# "--skip_target_instance_bank --conv_lr_ratio=1.0 --batch 128  --num_workers=4 --load_inat_in_memory --notes=update_centroids_w_mmult --momentum_target_centroids=0.9"

# "--unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=mlp --skip_target_instance_bank"
# "--is_pool_source_keypoints"

# "--momentum_source_instances=0 --momentum_target_centroids=0 --skip_target_instance_bank"
# "--momentum_source_instances=0.1 --momentum_target_centroids=0.1 --skip_target_instance_bank"
# "--momentum_source_instances=0.1 --momentum_target_centroids=0.1 --skip_target_instance_bank --unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=mlp"

# "--unseen_centroid_on_target=0.1 --unseen_prediction=gan-gp --optimizer=adam --learning_rate=1e-5"
# "--unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=gan-gp --learning_rate=1e-5"

# "--unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=gan-gp  --optimizer=adam --learning_rate=1e-5"

# "--optimizer=adam --learning_rate=5e-5"
# "--optimizer=adam --learning_rate=1e-5"

# "--update_target_unseen_weights=iter --unseen_centroid_on_target=0.1 --update_target_cls_weights"
# "--update_target_unseen_weights=iter --unseen_centroid_on_target=0.1"
# "--update_target_unseen_weights=epoch --unseen_centroid_on_target=0.1 --update_target_cls_weights"
# "--update_target_unseen_weights=epoch --unseen_centroid_on_target=0.1"


# settings after backbone_test (from CUB)
# "--conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros"

# settings after backbone_test (from CUB) april 2022
# "--weight_decay=0.2 --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros"

# "--backbone resnet101" "--backbone resnet50" "--backbone resnet18"
""

)

others2=(
# '--proto_on_source=0.1 --proto_on_target=0.1 --cross_on_source_instances=0.1 --cross_on_target_instances=0.1'
# '--proto_on_source=0.1 --proto_on_target=0.1'

# '--proto_on_source=0.1 --proto_on_target=1.0'
# '--proto_on_source=0.1 --proto_on_target=1.0 --use_source_as_cls_head'

# exp with cls_source_as_head

# '--superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=0.1'
# '--superv_on_target=0.1'
# '--superv_on_target=1.0'
# '--superv_on_target=0.1 --proto_cross_on_target_instances=0.1'
# '--superv_on_target=0.1 --proto_cross_on_target_instances=0.1 --proto_cross_on_source_instances=0.1'

# "--manualSeed -1" "--manualSeed -1" "--manualSeed -1" "--manualSeed -1" "--manualSeed -1"
# # experiments may
# "--epochs 100 --max_iterations 40000 --backbone resnet101 --batch 16 --accum_grad_iters=2"
# "--epochs 100 --max_iterations 40000 --backbone resnet50 --batch 16 --accum_grad_iters=2"
# "--epochs 100 --max_iterations 40000 --backbone resnet18 --batch 16 --accum_grad_iters=2"

# "--epochs 100 --max_iterations 40000 --backbone resnet18 --batch 32 --accum_grad_iters=1"
# "--epochs 100 --max_iterations 40000 --backbone resnet18 --batch 128 --accum_grad_iters=1"

# "" "" "" "" "" 

# missing replicas
"--backbone resnet101 --dataset inat17"
"--backbone resnet101 --dataset inat17"
"--backbone resnet18 --dataset inat17"
"--backbone resnet50 --dataset inat21"

)

# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb5/  --epochs 200 --billow_norm=basic_trainmean --use_source_as_cls_head  --dataset inat17"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb5/  --epochs 200 --billow_norm=basic_trainmean --use_source_as_cls_head  --dataset inat17"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 100 --billow_norm=basic_trainmean --filter_using_comm_names --use_source_as_cls_head --backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --notes=replicas_for_paper"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 100 --billow_norm=basic_trainmean --filter_using_comm_names --use_source_as_cls_head --backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --notes=saving_model --copy_checkpoint_freq=10"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_april_replicas/  --epochs 50 --billow_norm=basic_trainmean --filter_using_comm_names --use_source_as_cls_head --backbone resnet50 --batch 16 --accum_grad_iters=2 --optimizer=adam --learning_rate=1e-5 --notes=replicas_april"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_april_replicas/ --billow_norm=basic_trainmean --filter_using_comm_names --use_source_as_cls_head --optimizer=adam --learning_rate=1e-5 --notes=replicas_may6"

# settings after backbone_test (from CUB) april 2022 and 40k
EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_april_replicas/ --billow_norm=basic_trainmean --filter_using_comm_names --use_source_as_cls_head --optimizer=adam --learning_rate=1e-5 --weight_decay=0.2 --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros --epochs 100 --max_iterations 40000 --batch 16 --accum_grad_iters=2 --notes=replicas_may13"


#BSUB -W 120:00
#BSUB -o /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -R "rusage[mem=16000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 8
#BSUB -N
#BSUB -J DAinat[1-4]
##BSUB -u andresro@ethz.ch
##BSUB -w 'numended(3227583,*)'
#### BEGIN #####

index=$((LSB_JOBINDEX-1))

set +x

i=0
for oth in "${others[@]}"; do
   for oth1 in "${others1[@]}"; do
       for oth2 in "${others2[@]}"; do
		   if [ "$index" -eq "$i" ]; then
	            set -x
			    OTHER=$oth' '$oth1' '$oth2
	                set +x
	        fi
	        ((i+=1))
done;done;done;
echo $i combinations


module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1 eth_proxy || echo 'modules not loaded'

set -x

# to avoid having the same folder if several jobs start at the same time
# sleep $[ ( $RANDOM % 5 )  + $index ]s

python3 -u main_memorybank.py $OTHER $EXTRAARG --run_id $index

# python3 -u main.py


#
#
#
#### END #####
