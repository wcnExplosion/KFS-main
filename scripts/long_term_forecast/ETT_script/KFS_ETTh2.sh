export CUDA_VISIBLE_DEVICES=0

model_name=KFS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --d_model 1024 \
  --d_ff 256 \
  --percent 95 \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --d_model 1024 \
  --k 1 \
  --percent 95 \
  --d_ff 256 \
  --batch_size 16 \
  --itr 1

#
##lbd_list=(0 0.1 0.2 0.3  0.4 0.5 0.6 0.7 0.8 0.9 1.0)
##sl_list=(48 96 192 336 512 720)
##for lambda in ${lbd_list[@]}; do
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_336 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --fft_alpha 0.8 \
#  --percent 100 \
#  --d_model 1024 \
#  --d_ff 512 \
#  --itr 1
##done
#
##lbd_list=(0 0.1 0.2 0.3  0.4 0.5 0.6 0.7 0.8 0.9 1.0)
##sl_list=(48 96 192 336 512 720)
##for lambda in ${lbd_list[@]}; do
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_720 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --fft_alpha 0.7 \
#  --d_model 1024 \
#  --percent 100 \
#  --d_ff 2048 \
#  --batch_size 8 \
#  --itr 1
##  done

#python -u run.py  \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_96 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --fft_alpha  0.9 \
#  --d_model 1024 \
#  --d_ff 2048 \
#  --batch_size 32 \
#  --itr 1
#
##sl_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
##for lambda in ${sl_list[@]}; do
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_192 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --fft_alpha 0.9 \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --d_model 1024 \
#  --d_ff 256 \
#  --batch_size 32 \
#  --itr 1

#done
#


python -u run.py "$@" \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --fft_alpha 0.8 \
  --percent 95 \
  --d_model 1024 \
  --d_ff 512 \
  --itr 1
#
#
python -u run.py "$@" \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --d_model 1024 \
  --fft_alpha 0.7 \
  --percent 95 \
  --d_ff 2048 \
  --batch_size 8 \
  --itr 1
