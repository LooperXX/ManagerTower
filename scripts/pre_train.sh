date ; hostname ; pwd
ADDR_1=node-0
PORT_1=19800

MASTER_ADDR=$ADDR_1
MASTER_PORT=$PORT_1
NODE_RANK=$1 # input each node's rank

EXP_IS=288
EXP_PGB=8
EXP_PGEB=32
EXP_LR=1e-5
EXP_RGM=clip_pure

export MASTER_ADDR=node-0
export MASTER_PORT=19800
export NODE_RANK=$1

PREFIX_NAME="pt"

echo $MASTER_ADDR, $MASTER_PORT, $NODE_RANK, $EXP_IS, $EXP_PGB, $EXP_PGEB, $EXP_LR, $EXP_RGM

TIME=$(date "+%Y%m%d%H%M")

RUN_NAME=""$PREFIX_NAME"_"$EXP_IS"_"$EXP_PGB"_"$EXP_PGEB"_"$EXP_LR"_"$EXP_RGM"_"$TIME""

echo $RUN_NAME

python run.py with run_name=$RUN_NAME task_mlm_itm_clip_bert mt clip16 text_roberta $EXP_RGM data_root='./dataset/pre-train' num_gpus=8 num_nodes=8 image_size=$EXP_IS per_gpu_batchsize=$EXP_PGB per_gpu_eval_batchsize=$EXP_PGEB learning_rate=$EXP_LR

date

bash scripts/occupation.sh