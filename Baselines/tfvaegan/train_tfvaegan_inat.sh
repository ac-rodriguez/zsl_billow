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
"--dataset inat2017 --nclass_all=895"
"--dataset inat2021 --nclass_all=1485"
"--dataset inat2021mini --nclass_all=1485"
)

others2=(
    # "--syn_num 300"
    # "--syn_num 400"
    # "--syn_num 500"
    # "--syn_num 1000"

"--syn_num 100"
"--syn_num 1000"
"--syn_num 3000" # mem 64*3
"--syn_num 5000" # mem 64*3
)

EXTRAARG="--outf $WORK/birds/CUB/checkpoints_feb25_tfvaegan_replicas/ --nepoch 100 --batch_size 64 --dataroot $WORK/birds/CUB/ --valsplit=allhop --class_embedding billow --attSize 512 --code_path=${WORK}/billow_codes.h5 --notes=replicas_for_paper_wseeds"

#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/birds/output_lrs_gan/train_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output_lrs_gan/train_Le.%J.%I.txt
#BSUB -R "rusage[mem=32000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 8
#BSUB -N
##BSUB -J tfinat[40]
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

python -u train_images.py --gammaD 10 --gammaG 10 \
--gzsl --encoded_noise --preprocessing --cuda --image_embedding res101 \
--ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
--nz 312 --latent_size 312 --resSize 2048 --syn_num 300 \
--recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 $OTHER $EXTRAARG


#
#
#
#### END #####







