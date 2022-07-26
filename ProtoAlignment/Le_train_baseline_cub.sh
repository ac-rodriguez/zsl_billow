#!/bin/bash

set -x
set -e


others=(
"--dataset CUB_billow"
# "--dataset CUB_dna_billow" 
)

others1=(
    "--backbone resnet18 --batch 128"
    "--backbone resnet50 --batch 32"
    "--backbone resnet101 --batch 16"
)

others2=(
    "--manualSeed -1"  "--manualSeed -1" "--manualSeed -1" "--manualSeed -1" #  "--manualSeed -1"	
)

# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb24/  --epochs 200 --billow_norm=basic_trainmean --filter_using_comm_names --skip_target_instance_bank --use_source_as_cls_head --notes=backones_test"
# EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 50 --billow_norm=basic_trainmean --filter_using_comm_names --notes=replicas_for_paper_adam --optimizer=adam --learning_rate=1e-5 --conv_lr_ratio=0.1"
EXTRAARG="--checkpoints_dir=${WORK}/birds/DA_baseline/logs_feb25_replicas/  --epochs 50 --billow_norm=basic_trainmean --filter_using_comm_names --notes=replicas_for_paper_adam1 --optimizer=adam --learning_rate=1e-5 --conv_lr_ratio=1.0"

#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output/DA_mbank_Le.%J.%I.txt
#BSUB -R "rusage[mem=8000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 8
#BSUB -N
#BSUB -J DAbaselinednabillow[1-12]
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

python3 -u main.py $OTHER $EXTRAARG --run_id $index

# python3 -u main.py


#
#
#
#### END #####
