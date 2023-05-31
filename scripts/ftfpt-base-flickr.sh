date ; hostname ; pwd
ADDR_1=node-0
PORT_1=19800

MASTER_ADDR=$ADDR_1
MASTER_PORT=$PORT_1
NODE_RANK=$1 # input each node's rank

EXP_NODES=$2
EXP_IS=384
EXP_PGB=8
EXP_PGEB=8
EXP_LR=6e-6
EXP_BS=512
EXP_ME=20
EXP_WS=0.1
EXP_WD=0.01
EXP_LMH=5
EXP_LMC=5
EXP_THL=2
EXP_HHS=1
EXP_RGM=albef_randaug_wc
EXP_PGEBT=256
EXP_PGEBI=128
EXP_GAI=False
EXP_GGN=True
EXP_GAII=False
EXP_AF=True
EXP_IC=1
EXP_LP=ManagerTower_pt_base.ckpt

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK

PREFIX_NAME="ftfpt"

echo $MASTER_ADDR, $MASTER_PORT, $NODE_RANK, $EXP_NODES, $EXP_IS, $EXP_PGB, $EXP_PGEB, $EXP_LR, $EXP_BS, $EXP_ME, $EXP_WS, $EXP_WD, $EXP_LMH, $EXP_LMC, $EXP_THL, $EXP_HHS, $EXP_RGM

TIME=$(date "+%Y%m%d%H%M")
RUN_NAME=""$PREFIX_NAME"-"$EXP_PGB"-"$EXP_PGEB"-"$EXP_IS"-"$EXP_NL"-"$EXP_GGN"-"$EXP_RGM"-"$EXP_LR"-"$EXP_BS"-"$EXP_ME"-"$EXP_WS"-"$EXP_WD"-"$EXP_HHS"-"$TIME""
echo $RUN_NAME
python run.py with run_name=$RUN_NAME task_finetune_irtr_itm_itc_f30k_clip_bert mt clip16 text_roberta $EXP_RGM num_gpus=8 num_nodes=$EXP_NODES load_path=./best_checkpoints/$EXP_LP image_size=$EXP_IS per_gpu_batchsize=$EXP_PGB per_gpu_eval_batchsize=$EXP_PGEB learning_rate=$EXP_LR batch_size=$EXP_BS max_epoch=$EXP_ME warmup_steps=$EXP_WS weight_decay=$EXP_WD lr_mult_head=$EXP_LMH lr_mult_cross_modal=$EXP_LMC task_head_layers=$EXP_THL head_hidden_scale=$EXP_HHS per_gpu_eval_batchsize_text=$EXP_PGEBT per_gpu_eval_batchsize_image=$EXP_PGEBI gather_all_inputs=$EXP_GAI gather_global_negative=$EXP_GGN gather_all_image_inputs=$EXP_GAII amp_flag=$EXP_AF image_chunks=$EXP_IC 

date

bash scripts/occupation.sh