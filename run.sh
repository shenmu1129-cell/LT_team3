CUDA_VISIBLE_DEVICES=4 nohup python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 100 \
    --batch_size 1 \
    --temperature 0.4 \
    --dataroot "/home/sutongtong/LanTu_team3/dataset/nuScenes/train" \
    --enable_server_update \
    > logs/$(date +"%Y%m%d_%H%M%S").log 2>&1 &