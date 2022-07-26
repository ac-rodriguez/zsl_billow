#!/bin/bash

set -x
set -e


others=(

    # "--dataset CUB_billow --nclass_all 196 --nclass_seen 148 --class_embedding billow --attSize 512 --code_path=${WORK}/birds/billow_codes.npz  --batch_size 2048"
    # "--dataset CUB_billow --nclass_all 196 --nclass_seen 148 --class_embedding sent --attSize 1024 --batch_size 2048"
    # "--dataset CUB_billow --nclass_all 196 --nclass_seen 148 --class_embedding att --attSize 312 --batch_size 2048"

    "--dataset CUB_dna_billow --nclass_all 191 --nclass_seen 145 --class_embedding billow --attSize 512 --code_path=${WORK}/birds/billow_codes.npz  --batch_size 2048"
    # "--dataset CUB_dna_billow --nclass_all 191 --nclass_seen 145 --class_embedding sent --attSize 1024 --batch_size 2048"
    # "--dataset CUB_dna_billow --nclass_all 191 --nclass_seen 145 --class_embedding att --attSize 312 --batch_size 2048"
    # "--dataset CUB_dna_billow --nclass_all 191 --nclass_seen 145 --class_embedding att_w2v --attSize 400"
    # "--dataset CUB_dna_billow --nclass_all 191 --nclass_seen 145 --class_embedding att_dna --attSize 400"


)

others1=(
# ""
# "--normalize_embedding l2 --lr 1e-4"
# "--manualSeed 3483"  # original

"--manualSeed -1"
"--manualSeed -1"
"--manualSeed -1"
"--manualSeed -1"
# "--manualSeed -1"

)

others2=(

"--dataset CUB_billow --nclass_all 196 --nclass_seen 148"
"--dataset CUB_dna_billow --nclass_all 191 --nclass_seen 145"


)


EXTRAARG="--outf $WORK/birds/CUB/checkpoints/ --nepoch 900 --syn_num 100 --normalize_embedding l2 --lr 5e-5 --filter_using_comm_names --notes=replicas_for_paper_wseeds"


#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/birds/output_lrs_gan/train_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/birds/output_lrs_gan/train_Le.%J.%I.txt
#BSUB -R "rusage[mem=32000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 1
#BSUB -N
#BSUB -J ceCUBRepl[1-24]
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
    --cls_weight 0.001 --ins_temp 0.1 --cls_temp 0.1 $OTHER $EXTRAARG

#
#
#
#### END #####







