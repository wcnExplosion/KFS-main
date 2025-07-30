export CUDA_VISIBLE_DEVICES=0

model_name=KFS


#lbd_list=(80 85 90 95)
#for lambda in ${lbd_list[@]}; do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_96 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --fft_alpha 0.6 \
#  --percent ${lambda} \
#  --down_sampling_layers 1 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --d_model 128 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --des 'Exp' \
#  --itr 1
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_192 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 256 \
#  --d_ff 256 \
#  --fft_alpha 0.6 \
#  --percent ${lambda} \
#  --down_sampling_layers 1 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --batch_size 32 \
#  --itr 1
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_336 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 128 \
#  --d_ff 256 \
#  --percent ${lambda} \
#  --fft_alpha 0.8 \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --batch_size 32 \
#  --itr 1
##
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_720 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 128 \
#  --d_ff 512 \
#  --percent ${lambda} \
#  --fft_alpha 0.4 \
#  --down_sampling_layers 1 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --batch_size 8 \
#  --itr 1
#
#done

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --d_model 128 \
  --d_ff 512 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 256 \
  --fft_alpha 0.7 \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 512 \
  --fft_alpha 0.4 \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --batch_size 8 \
  --itr 1
