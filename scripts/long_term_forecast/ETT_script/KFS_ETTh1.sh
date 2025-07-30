export CUDA_VISIBLE_DEVICES=0

model_name=KFS


##lbd_list=(0 0.1 0.2 0.3  0.4 0.5 0.6 0.7 0.8 0.9 1.0)
##for lambda in ${lbd_list[@]}; do
#python -u run.py \
# --task_name long_term_forecast \
# --is_training 1 \
# --root_path ./dataset/ETT-small/ \
# --data_path ETTh1.csv \
# --model_id ETTh1_96_96 \
# --model $model_name \
# --data ETTh1 \
# --features M \
# --seq_len 96 \
# --label_len 48 \
# --pred_len 96 \
# --e_layers 2 \
# --d_layers 1 \
# --factor 3 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --d_ff 256 \
# --d_model 512 \
# --des 'Exp' \
# --fft_alpha 0.6 \
# --percent 90 \
# --down_sampling_layers 2 \
# --down_sampling_window 2 \
# --down_sampling_method "avg" \
# --batch_size 32 \
# --itr 1
##  done
#
#
##lbd_list=(0 0.1 0.2 0.3  0.4 0.5 0.6 0.7 0.8 0.9 1.0)
##for lambda in ${lbd_list[@]}; do
#python -u run.py "$@" \
# --task_name long_term_forecast \
# --is_training 1 \
# --root_path ./dataset/ETT-small/ \
# --data_path ETTh1.csv \
# --model_id ETTh1_96_192 \
# --model $model_name \
# --data ETTh1 \
# --features M \
# --seq_len 96 \
# --label_len 48 \
# --pred_len 192 \
# --e_layers 2 \
# --d_layers 1 \
# --factor 3 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des 'Exp' \
# --d_model 16 \
# --d_ff 64 \
# --fft_alpha 0.9 \
# --percent 90 \
# --learning_rate 0.01 \
# --down_sampling_layers 3 \
# --down_sampling_window 2 \
# --down_sampling_method "avg" \
# --batch_size 128 \
# --itr 1
##done
## #
## lbd_list=(0 0.1 0.2 0.3  0.4 0.5 0.6 0.7 0.8 0.9 1.0)
## for lambda in ${lbd_list[@]}; do
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --fft_alpha 0.9 \
#  --d_model 16 \
#  --d_ff 32 \
#  --percent 95 \
#  --learning_rate 0.005 \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --batch_size 128 \
#  --itr 1
#
#
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 256 \
#  --percent 95 \
#  --fft_alpha 0.3 \
#  --learning_rate 0.0001 \
#  --d_ff 1024 \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --batch_size 32 \
#  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_ff 256 \
  --d_model 512 \
  --des 'Exp' \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 512 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 2048 \
  --learning_rate 0.0001 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "conv" \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 1024 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "conv" \
  --batch_size 32 \
  --itr 1