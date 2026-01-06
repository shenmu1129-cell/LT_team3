CUDA_VISIBLE_DEVICES=5 nohup python run_federated_qwenvl.py \
    --num_clients 2 \
    --num_rounds 100 \
    --batch_size 8 \
    --temperature 0.4 \
    --dataroot "/home/sutongtong/LanTu_team3/dataset/nuScenes/train" \
    --enable_server_update \
    --num_clean_clients 1 \
    --model_path "/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct" \
    > logs/$(date +"%Y%m%d_%H%M%S").log 2>&1 &