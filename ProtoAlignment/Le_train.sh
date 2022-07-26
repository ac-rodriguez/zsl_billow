#!/bin/bash

set -x
set -e


others=(
#   "--backbone resnet50 --batch 50"
  #  "--backbone resnet18 --batch 128"
""
)

others1=(
# "--update_target_cls_weights"
"--superv_on_both_target_cls_head=1.0"
)

others2=(
#'--proto_on_source=0.1 --proto_on_target=0.1 --cross_on_source_instances=0.1 --cross_on_target_instances=0.1'
"--backbone resnet18 --batch 128 --optimizer=adam --learning_rate=1e-5 --superv_on_target=0.1 --proto_on_source=1.0 --proto_on_target=1.0 --conv_lr_ratio=0.1 --dataset CUB" #  --superv_on_both_target_cls_head=1.0"
# '--proto_on_source=0.1 --proto_on_target=0.1'
#'--cross_on_source_instances=0 --cross_on_target_instances=0'
#'--proto_on_source=0.1 --proto_on_target=0.1 --cross_on_target_instances=0.1'
)

EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb5/ --epochs 200 --billow_norm=basic_trainmean --use_source_as_cls_head"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb5/  --epochs 50"

#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -R "rusage[mem=32000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 1
##BSUB -N
#BSUB -J DA[1]
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
