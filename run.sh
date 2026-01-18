CUDA_VISIBLE_DEVICES=5 nohup python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 100 \
    --batch_size 4 \
    --lr 1e-5 \
    --tau 5.0 \
    --beta_divergence 0.5 \
    --temperature 2 \
    --partition_mode non-iid-dirichlet \
    --dirichlet_alpha 0.5 \
    --dataroot "/home/sutongtong/LanTu_team3/dataset/nuScenes/train" \
    --enable_server_update \
    --num_clean_clients 1 \
    --model_path "/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct" \
    > logs/$(date +"%Y%m%d_%H%M%S").log 2>&1 &
