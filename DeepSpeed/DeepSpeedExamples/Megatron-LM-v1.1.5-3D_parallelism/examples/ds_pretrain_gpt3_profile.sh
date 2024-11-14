#! /bin/bash

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6009
export HOME=/root/AMP/AMP

DATA_PATH=/root/data/data/gpt/my-gpt2_text_document
VOCAB_PATH=$HOME/AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/gpt_data/gpt2-vocab.json
MERGE_PATH=$HOME/AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/gpt_data/gpt2-merges.txt
CHECKPOINT_PATH=$HOME/AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ckpts/gpt2_1542m_ds_distributed

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
#config_json="$script_dir/ds_zero_stage_2_config.json"
config_json="$script_dir/ds_config.json"

# Megatron Model Parallelism
mp_size=$4
# DeepSpeed Pipeline parallelism
pp_size=1

NLAYERS=$1
NHIDDEN=$2
BATCHSIZE=1
gas=1

exp_name=$3
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${pp_size}pp_${mp_size}mp_${BATCHSIZE}b_ds4"

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --gas ${gas} \
        --exp_name ${exp_name} \
        --num-layers ${NLAYERS} \
        --hidden-size ${NHIDDEN} \
        --num-attention-heads 32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --batch-size $BATCHSIZE \
        --train-iters 320000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 98,2,0 \
        --distributed-backend nccl \
        --lr 6.0e-5 \
        --lr-decay-style cosine \
        --min-lr 6.0e-6 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --log-interval 20 \
        --fp16 \
        --save-interval 1000 \
        --eval-interval 100000 \
        --eval-iters 10000 \
        --tensorboard-dir ${LOGDIR}
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
            "

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

# %@
# run_cmd="deepspeed --hostfile=/root/AMP/hostfile  --num_nodes 1 --num_gpus $4 pretrain_gpt2.py ${full_options}"
run_cmd="deepspeed   --num_nodes 1 --num_gpus $4 pretrain_gpt2.py ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x