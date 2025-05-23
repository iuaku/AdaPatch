if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=AdaPatch
root_path=./dataset/
data_path=ETTh1.csv
data_name=ETTh1
features=M
enc_in=7
des=Exp
itr=1
batch_size=32


python -u run_longExp.py \
    --is_training 1 \
    --alpha 5 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 64 \
    --slice_stride 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTh1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in $enc_in \
    --gpu 1\
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_ETTh1_96_96.log


python -u run_longExp.py \
    --is_training 1 \
    --alpha 5 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 64 \
    --slice_stride 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTh1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in $enc_in \
    --des $des \
    --gpu 1\
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0001 > logs/LongForecasting/${model_name}_ETTh1_96_192.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 5 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 64 \
    --slice_stride 4 \
    --gpu 1\
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTh1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0001 > logs/LongForecasting/${model_name}_ETTh1_96_336.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 5 \
    --slice_len 8 \
    --middle_len 512 \
    --gpu 1\
    --hidden_len 64 \
    --slice_stride 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTh1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0001 > logs/LongForecasting/${model_name}_ETTh1_96_720.log
