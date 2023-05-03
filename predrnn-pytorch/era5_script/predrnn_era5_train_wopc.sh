export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
base_path="/scratch/09012/haoli1/ERA5/dataset/"
train_data_paths=""
for file in /scratch/09012/haoli1/ERA5/dataset/*; do
    train_data_paths+="${file},"
done

# Remove the trailing comma
train_data_paths=${train_data_paths%,}

python -u run1.py \
    --is_training 1 \
    --device cuda:0 \
    --dataset_name mnist \
    --train_data_paths ${train_data_paths} \
    --valid_data_paths /scratch/09012/haoli1/ERA5/val_dataset/era5_train_09012020_3_24hr.npz \
    --save_dir /work/09012/haoli1/ls6/PredRNN_checkpoints/ \
    --gen_frm_dir /work/09012/haoli1/ls6/PredRNN_checkpoints/ \
    --model_name predrnn_v2 \
    --reverse_input 0 \
    --is_WV 0 \
    --press_constraint 0 \
    --center_enhance 0 \
    --patch_size 40 \
    --weighted_loss 1 \
    --upload_run 1 \
    --layer_need_enhance 1 \
    --find_max False \
    --multiply 2 \
    --img_height 720 \
    --img_width 1440 \
    --use_weight 0 \
    --layer_weight 20 \
    --img_channel 3 \
    --img_layers 0,1,2 \
    --input_length 24 \
    --total_length 48 \
    --num_hidden 512,512,512,512 \
    --skip_time 1 \
    --wavelet db1 \
    --filter_size 5 \
    --stride 1 \
    --layer_norm 1 \
    --decouple_beta 0.05 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 1e-4 \
    --batch_size 3 \
    --test_batch_size 9 \
    --max_iterations 20000 \
    --display_interval 1000 \
    --test_interval 10 \
    --snapshot_interval 2000 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --curr_best_mse 0.027 \
    --save_best_name test_schedule_sampling_DynaMean \
    # --pretrained_model /work/09012/haoli1/ls6/PredRNN_checkpoints/ \
    # --pretrained_model_name model_test_schedule_sampling_DynaMean.ckpt \

#cp /scratch/network/hvtran/era5/checkpoints/era5_predrnn/model.ckpt-1000 /home/hvtran/
#,/work/09012/haoli1/ls6/ERA5/era5_train_1001002016_3_24hr.npz,/work/09012/haoli1/ls6/ERA5/era5_train_0827002021_3_24hr.npz,/work/09012/haoli1/ls6/ERA5/era5_train_0921002022_3_24hr.npz
