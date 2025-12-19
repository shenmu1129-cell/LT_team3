CUDA_VISIBLE_DEVICES=4 python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 10 \
    --batch_size 16 \
    --max_batches 2 \
    --data_partition non_iid 