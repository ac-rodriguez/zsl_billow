#!/bin/bash

set -x
set -e


others=(

# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow "
# "--backbone resnet50 --batch 32 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow "
# "--backbone resnet101 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow "

# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-4 --dataset CUB_billow "
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=5e-5 --dataset CUB_billow "
# "--backbone resnet50 --batch 32 --optimizer=adam --learning_rate=5e-5 --dataset CUB_billow "

# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow "
# "--backbone resnet18 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow"
# "--backbone resnet18 --batch 32 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow"
# "--backbone resnet50 --batch 32 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --accum_grad_iters=4"
# "--backbone resnet101 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --accum_grad_iters=8"


# "--backbone resnet18 --batch 32 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow"
# "--backbone resnet50 --batch 32 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow"
# "--backbone resnet101 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --accum_grad_iters=2"


# "--backbone resnet18 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --accum_grad_iters=2"
# "--backbone resnet50 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --accum_grad_iters=2"
# "--backbone resnet101 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --accum_grad_iters=2"

# "--backbone resnet50 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_dna_billow --accum_grad_iters=2"
# "--backbone resnet101 --batch 16 --optimizer=adam --learning_rate=1e-5 --dataset CUB_dna_billow --accum_grad_iters=2"

# "--backbone resnet18 --batch 16 --optimizer=adam --learning_rate=5e-6 --dataset CUB_billow --accum_grad_iters=2"
"--backbone resnet50 --batch 16 --optimizer=adam --learning_rate=5e-6 --dataset CUB_billow --accum_grad_iters=2"
"--backbone resnet101 --batch 16 --optimizer=adam --learning_rate=5e-6 --dataset CUB_billow --accum_grad_iters=2"

# test smalller learning rate?

# "--dataset CUB_dna_billow"
# "--dataset CUB_billow"

)

others1=(
# "--conv_lr_ratio=0.1 --momentum_source_instances=0.0 --momentum_target_centroids=0.5 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0"

# settings for wdecay tests
# "--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros"
# "--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.03  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros"
# "--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.01  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros"
# settings after backbone_test
# "--conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0  --init_type_target_centroids=zeros"


#### ablations (may)
# final config
# "--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head"
# # all ones
"--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head"
# # no clshead supervision
"--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=0.0 --use_source_as_cls_head"
# # no proto losses
"--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=0.0 --proto_on_target=0.0 --superv_on_both_target_cls_head=1.0  --use_source_as_cls_head"
# # learned cls_head
"--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0"
# contrastive head 
"--momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head --contrastive_head"


#### ablations (march)
# final config
# "--superv_on_target=0.1 --superv_on_source=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head"
# # all ones
# "--superv_on_target=1.0 --superv_on_source=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head"
# # no clshead supervision
# "--superv_on_target=0.1 --superv_on_source=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=0.0 --use_source_as_cls_head"
# # no proto losses
# "--superv_on_target=0.1 --superv_on_source=1.0  --proto_on_source=0.0 --proto_on_target=0.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head"
# # learned cls_head
# "--superv_on_target=0.1 --superv_on_source=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=0.0"
# contrastive head 
# "--superv_on_target=0.1 --superv_on_source=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --use_source_as_cls_head --contrastive_head"

# " --conv_lr_ratio=0.1 --momentum_source_instances=0.5 --momentum_target_centroids=0.5"
# " --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9"
)

others2=(
    # "--manualSeed -1" "--manualSeed -1" "--manualSeed -1" "--manualSeed -1"  "--manualSeed -1"
	""
	
	### wdecay tests
	# "--weight_decay=5e-4"
	# "--weight_decay=0.001 --conv_lr_ratio=0.1"
	# "--weight_decay=0.005 --conv_lr_ratio=0.1"
	# "--weight_decay=0.01 --conv_lr_ratio=0.1"
	# "--weight_decay=0.001 --conv_lr_ratio=0.01"
	# "--weight_decay=0.005 --conv_lr_ratio=0.01"
	# "--weight_decay=0.01 --conv_lr_ratio=0.01"
	# "--weight_decay=5e-4 --conv_lr_ratio=0.01"
	# "--weight_decay=1e-4 --conv_lr_ratio=0.01"


	# "--weight_decay=0.05 --conv_lr_ratio=0.1"
	# "--weight_decay=0.05 --conv_lr_ratio=0.01"
	
	
	# "--weight_decay=0.1 --conv_lr_ratio=0.01"
	
	# "--weight_decay=0 --conv_lr_ratio=0.1"
	# "--weight_decay=0.1 --conv_lr_ratio=0.1"
	# "--weight_decay=0.2 --conv_lr_ratio=0.1"
	# "--weight_decay=0.3 --conv_lr_ratio=0.1"
	# "--weight_decay=0.4 --conv_lr_ratio=0.1"
	# "--weight_decay=0.5 --conv_lr_ratio=0.1"

	# "--weight_decay=0.01 --conv_lr_ratio=0.05"
	# "--weight_decay=0.1 --conv_lr_ratio=0.05"
	# "--weight_decay=0.2 --conv_lr_ratio=0.05"
	# "--weight_decay=0.3 --conv_lr_ratio=0.05"
	# "--weight_decay=0.4 --conv_lr_ratio=0.05"
	# "--weight_decay=0.5 --conv_lr_ratio=0.05"


	# "--conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --init_type_target_centroids=zeros"
# '--superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0'
# '--superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --init_type_target_centroids=zeros'

# '--superv_on_target=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0'
# '--superv_on_target=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --init_type_target_centroids=zeros'

)

# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_april8/  --epochs 50 --billow_norm=basic_trainmean --filter_using_comm_names --skip_target_instance_bank --use_source_as_cls_head --weight_decay=0.2 --conv_lr_ratio=0.1 --notes=replicas_april "
EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_april8/  --epochs 50 --billow_norm=basic_trainmean --filter_using_comm_names --skip_target_instance_bank --weight_decay=0.2 --conv_lr_ratio=0.1 --init_type_target_centroids=zeros --notes=ablations_may "
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_march22/  --epochs 50 --billow_norm=basic_trainmean --filter_using_comm_names --skip_target_instance_bank --use_source_as_cls_head --notes=backones_test_wdecay_50epochsb16x2"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_march22/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names --skip_target_instance_bank --use_source_as_cls_head --notes=backones_test_wdecay"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb24/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names --skip_target_instance_bank --use_source_as_cls_head --notes=backones_test_w_replicas"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names  --backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --notes=replicas_for_paper"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names  --backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --use_source_as_cls_head --notes=saving_model --copy_checkpoint_freq=50"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names  --backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --notes=ablations"



#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -R "rusage[mem=8000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 8
#BSUB -N
#BSUB -J DAreplicas[1-10]
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
