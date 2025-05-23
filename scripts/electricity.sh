if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=AdaPatch
root_path=./dataset/
data_path=electricity.csv
data_name=custom
features=M
enc_in=321
des=Exp
itr=1
batch_size=16



python -u run_longExp.py \
    --is_training 1 \
    --alpha 10 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 8 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 96 \
    --gpu 0 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001  > logs/LongForecasting/${model_name}_electricity_96_96.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 3 \
    --gpu 0 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 8 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_electricity_96_192.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 512 \
    --gpu 0 \
    --slice_stride 4 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_electricity_96_336.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 8 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 4 \
    --gpu 0 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_electricity_96_720.log
