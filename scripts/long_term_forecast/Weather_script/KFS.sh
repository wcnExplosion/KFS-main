export CUDA_VISIBLE_DEVICES=0

model_name=KFS

#sl_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#for lambda in ${sl_list[@]}; do
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_96_96 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --fft_alpha ${lambda} \
#  --des 'Exp' \
#  --down_sampling_layers 1 \
#  --down_sampling_window 2 \
#  --down_sampling_method "avg" \
#  --d_model 128 \
#  --d_ff 512 \
#  --batch_size 32 \
#  --itr 1
#
#done

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --d_model 128 \
  --d_ff 512 \
  --batch_size 32 \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --down_sampling_layers 1 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --d_model 256 \
  --d_ff 1024 \
  --k 16 \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method "avg" \
  --d_model 1024 \
  --d_ff 512 \
  --batch_size 16 \
  --itr 1
