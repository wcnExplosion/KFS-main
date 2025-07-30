export CUDA_VISIBLE_DEVICES=0

model_name=KFS
#lbd_list=(0.5 0.6 0.7 0.8 0.9 1.0)
#sl_list=(48 96 192 336 512 720)
#for lambda in ${sl_list[@]}; do${lambda}
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_720 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96  \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 128 \
#  --d_ff 256 \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --itr 1
#
#done


#lbd_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#for lambda in ${lbd_list[@]}; do
# python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_96 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 512 \
#  --d_ff 256 \
#  --fft_alpha ${lambda} \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --batch_size 16 \
#  --itr 1
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_192 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 256 \
#  --d_ff 512 \
#  --fft_alpha ${lambda} \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --itr 1
#
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_336 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 128 \
#  --d_ff 256 \
#  --fft_alpha ${lambda} \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --itr 1
#
#
#python -u run.py "$@" \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_720 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --d_model 1024 \
#  --d_ff 512 \
#  --batch_size 8 \
#  --fft_alpha ${lambda} \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --des 'Exp' \
#  --itr 1
#
#done

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 256 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --batch_size 16 \
  --itr 1
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 512 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 1024 \
  --d_ff 512 \
  --batch_size 8 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --des 'Exp' \
  --itr 1