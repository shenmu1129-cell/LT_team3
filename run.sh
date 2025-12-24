CUDA_VISIBLE_DEVICES=5 nohup python run_federated_qwenvl.py \
    --num_clients 5 \
    --num_rounds 100 \
    --batch_size 32 \
    --max_batches 20     \
    --temperature 0.4 \
    --data_partition non_iid \
    --enable_server_update \
    --server_max_batches 2 \
    2>/dev/null &