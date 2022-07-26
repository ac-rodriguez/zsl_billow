#!/bin/bash

set -x
set -e


others=(
# "--backbone resnet50 --batch 20 --superv_on_target=1.0 --superv_on_source=1.0"

#    "--backbone resnet50 --batch 20 --learning_rate=1e-3"
#    "--backbone resnet18 --batch 128 --superv_on_target=1.0 --superv_on_source=1.0"
#  "--backbone resnet18 --batch 128 --contrastive_head"

#  "--backbone resnet18 --batch 128 --superv_on_target=0.0"


# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5  --superv_on_both_target_cls_head=1.0"
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5"
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --conv_lr_ratio=1.0"

# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow --notes=using_comm_name_filtering_on_cub"
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB_dna_billow --notes=using_comm_name_filtering_on_cub"
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB --notes=using_old_code_and_no_in_memory_loading"
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB --filter_using_comm_names --notes=using_new_code_and_in_memory"

"--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB_billow "
"--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB_dna_billow"
# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --dataset CUB --filter_using_comm_names"

# "--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --contrastive_head"

# "--backbone resnet18 --batch 128 --superv_on_target=1.0 --optimizer=adam --learning_rate=1e-5"

# "--backbone resnet50 --batch 64 --superv_on_target=0.1 --freeze_backbone"
# "--backbone resnet101 --batch 64 --superv_on_target=0.1 --freeze_backbone --optimizer=adam --learning_rate=1e-5"

#  "--backbone resnet18 --batch 128 --superv_on_target=0.1 --contrastive_head"
)

others1=(
# "--update_target_cls_weights"
# ""
# "--unseen_centroid_on_target=0.1 --unseen_prediction=mlp"
# "--unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=mlp"

# "--unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=mlp "
# ""
# "--billow_from_numpy"
# "--skip_target_instance_bank"
# "--unseen_centroid_on_target=0.1 --superv_on_unseen_predicted_centroids=0.1 --unseen_prediction=mlp --is_pool_source_keypoints"
# "--is_pool_source_keypoints"

# "--momentum_target_centroids=0.1"
# "--momentum_target_centroids=0.1 --skip_target_instance_bank"

# "--momentum_target_centroids=0.01 --skip_target_instance_bank"
# "--momentum_target_centroids=0.1"

# "--momentum_target_centroids=0 --skip_target_instance_bank"

# "--init_type_target_centroids=zeros"
# "--momentum_target_centroids=0.5 --skip_target_instance_bank"
# "--momentum_target_centroids=0.9 --skip_target_instance_bank"

" --conv_lr_ratio=0.1 --momentum_source_instances=0.0 --momentum_target_centroids=0.5 --skip_target_instance_bank"
" --conv_lr_ratio=0.1 --momentum_source_instances=0.5 --momentum_target_centroids=0.5 --skip_target_instance_bank"
" --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank"

# " --conv_lr_ratio=0.1 --momentum_source_instances=0.5 --momentum_target_centroids=0.5 --skip_target_instance_bank"
# " --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank"
# " --conv_lr_ratio=1.0 --momentum_source_instances=0.9 --momentum_target_instances=0.9"
# " --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9"
# " --conv_lr_ratio=0.1 --momentum_source_instances=0.9 --momentum_target_centroids=0.9 --skip_target_instance_bank --proto_cross_on_target_instances=0.1 --proto_cross_on_source_instances=0.1"


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
)

others2=(
# '--proto_on_source=0.1 --proto_on_target=0.1 --cross_on_source_instances=0.1 --cross_on_target_instances=0.1'
# '--proto_on_source=0.1 --proto_on_target=0.1'

# '--proto_on_source=0.1 --proto_on_target=1.0'
# '--proto_on_source=0.1 --proto_on_target=1.0 --use_source_as_cls_head'

# exp with cls_source_as_head

'--superv_on_target=0.1  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0'
'--superv_on_target=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0'


# '--superv_on_target=1.0  --proto_on_source=0.0 --proto_on_target=0.0 --superv_on_both_target_cls_head=1.0'
# '--superv_on_target=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0'
# '--superv_on_target=1.0  --proto_on_source=1.0 --proto_on_target=1.0 --superv_on_both_target_cls_head=1.0 --proto_cross_on_target_instances=1.0'

# '--superv_on_target=0.1'
# '--superv_on_target=1.0'
# '--superv_on_target=0.1 --proto_cross_on_target_instances=0.1'
# '--superv_on_target=0.1 --proto_cross_on_target_instances=0.1 --proto_cross_on_source_instances=0.1'

# exp with momentum normal
# ''
# '--superv_on_target=0.1 --proto_cross_on_target_instances=0.05'


# '--proto_on_source=0.1 --proto_on_target=0.0 --proto_cross_on_target_instances=1.0'

# '--proto_on_source=0.1 --proto_on_target=1.0 --proto_cross_on_target_instances=1.0 --proto_cross_on_source_instances=1.0'
# '--proto_on_source=0.1 --proto_on_target=2.0'
# '--proto_on_source=1.0 --proto_on_target=1.0'

# '--proto_on_source=1.0 --proto_on_target=1.0'
# '--proto_on_source=1.0 --proto_on_target=1.0 --cross_on_source_instances=1.0 --cross_on_target_instances=1.0 --freeze_backbone'
# '--proto_on_source=1.0 --proto_on_target=1.0 --cross_on_source_instances=1.0 --cross_on_target_instances=1.0'

# '--proto_on_source=1.0 --proto_on_target=1.0 --proto_cross_on_source_instances=1.0 --proto_cross_on_target_instances=1.0 --freeze_backbone'
# '--proto_on_source=0.1 --proto_on_target=1.0 --proto_cross_on_source_instances=1.0 --proto_cross_on_target_instances=1.0'

# '--cross_on_source_instances=0.1 --cross_on_target_instances=0.1'
# '--proto_on_source=0.1 --proto_on_target=0.1 --cross_on_target_instances=0.1'
# ''
# '--superv_target=0.0 --proto_on_source=1.0 --proto_on_target=1.0 --proto_cross_on_source_instances=1.0 --proto_cross_on_target_instances=1.0'
# '--superv_target=0.1 --proto_on_source=1.0 --proto_on_target=1.0 --proto_cross_on_source_instances=1.0 --proto_cross_on_target_instances=1.0'

# '--proto_on_source=0.1 --proto_on_target=0.1 --proto_cross_on_source_instances=0.1 --proto_cross_on_target_instances=0.1'
# '--proto_on_source=0.1 --proto_on_target=0.1 --proto_cross_on_source_instances=0.1'
# '--proto_on_source=0.1 --proto_on_target=0.1 --proto_cross_on_target_instances=0.1'
# '--proto_on_source=1 --proto_on_target=0.5 --cross_on_target_instances=0.5 --cross_on_source_instances=0.5'
# '--proto_on_source=1 --proto_on_target=0.5 --cross_on_target_instances=0.5'

)

# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb5/  --epochs 200 --billow_norm=basic_trainmean --use_source_as_cls_head"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb5/  --epochs 200 --billow_norm=basic_trainmean"


EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb24_replicas/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names --notes=replicas_for_paper_wseeds"

# EXTRAARG="--checkpoints_dir=${SCRATCH}/birds/DA_baseline/logs_feb5/  --epochs 200 --billow_norm=basic_trainmean --use_source_as_cls_head"
# EXTRAARG="--checkpoints_dir=${SCRATCH}/birds/DA_baseline/logs_feb5/  --epochs 200 --billow_norm=basic_trainmean --proto_on_source=1.0 --proto_on_target=1.0"

#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -R "rusage[mem=32000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 1
##BSUB -N
#BSUB -J DAcub[1-2]
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
