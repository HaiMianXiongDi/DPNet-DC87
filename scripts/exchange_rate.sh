#!/usr/bin/env bash
# Exchange rate dataset, fixed patch_len/stride (pred_len‑specific hyperparams)

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=DPNet
data_name=custom
root_path=./dataset
data_path=exchange_rate.csv
features=M
enc_in=8
seq_len=96
train_epochs=20
stage_num=4
itr=1
des=Exp


pred_lens=(96)
bss=(256)
hds=(0.2)
lrs=(0.0001)

patch_len=16
stride=8


n=${#pred_lens[@]}
if [ $n -ne ${#bss[@]} ] || [ $n -ne ${#hds[@]} ] || [ $n -ne ${#lrs[@]} ]; then
    echo "[ERROR] array length mismatch"
    exit 1
fi


for ((i=0;i<$n;i++)); do
    pl=${pred_lens[$i]}
    bs=${bss[$i]}
    hd=${hds[$i]}
    lr=${lrs[$i]}

    model_id="exchange_sl${seq_len}_pl${pl}_bs${bs}_hd${hd}_lr${lr}_patch${patch_len}_str${stride}_DPNet"
    log_path="./logs/LongForecasting/${model_id}.log"

    echo ">>> Running model_id=$model_id"
    python -u run_longExp.py \
        --is_training 1 \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$model_id" \
        --model "$model_name" \
        --data "$data_name" \
        --features "$features" \
        --seq_len "$seq_len" \
        --pred_len "$pl" \
        --enc_in "$enc_in" \
        --des "$des" \
        --stage_num "$stage_num" \
        --head_dropout "$hd" \
        --train_epochs "$train_epochs" \
        --itr "$itr" \
        --batch_size "$bs" \
        --learning_rate "$lr" \
        --stage_pool_kernel 2 \
        --stage_pool_stride 2 \
        --stage_pool_padding 1 \
        --patch_len "$patch_len" \
        --stride "$stride" \
    | tee "$log_path"

    echo ">>> Finished model_id=$model_id"
done

echo "=== All Exchange experiments (pred_len‑specific hyperparams) finished! ==="
