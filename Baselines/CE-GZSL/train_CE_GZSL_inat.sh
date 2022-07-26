#!/bin/bash

set -x
set -e


others=(
    "--manualSeed -1"
    "--manualSeed -1"
    "--manualSeed -1"
    "--manualSeed -1"
    "--manualSeed -1"
)

others1=(


"--dataset inat2017 --nclass_all=895 --nclass_seen=381 --normalize_embedding l2 --lr 5e-5"
"--dataset inat2021 --nclass_all=1485 --nclass_seen=749 --normalize_embedding l2 --lr 5e-5"
"--dataset inat2021mini --nclass_all=1485 --nclass_seen=749 --normalize_embedding l2 --lr 5e-5"

)

others2=(
"--syn_num 100"

# "--syn_num 1000"
# "--syn_num 3000" # mem 64*3
# "--syn_num 5000" # mem 64*3
)



EXTRAARG="--outf $WORK/birds/CUB/checkpoints/ --nepoch 100 --batch_size 512 --valsplit=allhop --class_embedding billow --attSize 512 --code_path=${WORK}/birds/billow/codes.h5 --notes=replicas_for_paper_wseeds"


#BSUB -W 72:00
#BSUB -o /cluster/scratch/andresro/birds/output_lrs_gan/train_inat_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output_lrs_gan/train_inat_Le.%J.%I.txt
#BSUB -R "rusage[mem=64000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 3
#BSUB -N
#BSUB -J ceGZinat21[1-60]
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


python -u CE_GZSL.py --dataroot $WORK/birds/CUB/  \
    --nz 1024 --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 \
    --cls_weight 0.001 --ins_temp 0.1 --cls_temp 0.1 --manualSeed 3483  $OTHER $EXTRAARG

#
#
#
#### END #####







