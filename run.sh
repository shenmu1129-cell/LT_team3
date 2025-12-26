CUDA_VISIBLE_DEVICES=4 nohup python run_federated_qwenvl.py \
    --num_clients 5 \
    --num_rounds 100 \
    --batch_size 32 \
    --max_batches 20     \
    --temperature 0.4 \
    --data_partition non_iid \
    --enable_server_update \
    --server_max_batches 2 \
    --malicious_client_ratio 0.4 \
    > logs/$(date +"%Y%m%d_%H%M%S").log 2>&1 &