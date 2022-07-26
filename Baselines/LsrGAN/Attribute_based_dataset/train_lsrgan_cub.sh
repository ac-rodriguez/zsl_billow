#!/bin/bash

set -x
set -e


others=(

    "--dataset CUB_billow --nclass_all 196 --class_embedding billow --attSize 512 --code_path=${WORK}/birds/billow_codes.npz  --batch_size 2048"
    # "--dataset CUB_billow --nclass_all 196 --class_embedding sent --attSize 1024 --batch_size 2048"
    # "--dataset CUB_billow --nclass_all 196 --class_embedding att --attSize 312 --batch_size 2048"

    # "--dataset CUB_dna_billow --nclass_all 191 --class_embedding billow --attSize 512 --code_path=${WORK}/birds/billow_codes.npz  --batch_size 2048"
    # "--dataset CUB_dna_billow --nclass_all 191 --class_embedding sent --attSize 1024 --batch_size 2048"
    # "--dataset CUB_dna_billow --nclass_all 191 --class_embedding att --attSize 312 --batch_size 2048"
    # "--dataset CUB_dna_billow --nclass_all 191 --class_embedding att_w2v --attSize 400"
    # "--dataset CUB_dna_billow --nclass_all 191 --class_embedding att_dna --attSize 400"


)

others1=(

"--dataset=CUB_billow --nclass_all=196"
"--dataset=CUB_dna_billow --nclass_all=191"

)

others2=(

# "--manualSeed 3483"  # original
"--manualSeed -1"
"--manualSeed -1"
"--manualSeed -1"
"--manualSeed -1"
# "--manualSeed -1"

)


EXTRAARG="--outf $WORK/birds/CUB/checkpoints/ --correlation_penalty 20 --nepoch 40 --filter_using_comm_names --unseen_cls_weight 0.03 --normalize_embedding l2 --notes=replicas_for_paper_wseeds"



#BSUB -W 4:00
#BSUB -o /cluster/scratch/andresro/birds/output_lrs_gan/train_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output_lrs_gan/train_Le.%J.%I.txt
#BSUB -R "rusage[mem=32000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 1
#BSUB -N
#BSUB -J lrsRepl[1-24]
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


python -u clswgan.py --gzsl --val_every 1 --cls_weight 0.01 \
    --dataroot $WORK/birds/CUB/  \
    --preprocessing --cuda --image_embedding res101 \
    --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 \
    --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
    --batch_size 64 --nz 312 --resSize 2048 --syn_num 300 --outname cub \
    --no_classifier True --epsilon 0.1 --upper_epsilon 0.1 $OTHER $EXTRAARG

#
#
#
#### END #####







