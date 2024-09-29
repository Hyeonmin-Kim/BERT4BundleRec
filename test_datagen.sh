dataset_name="test_data"
max_seq_length=50
max_predictions_per_seq=20
mask_prob=1.0
dupe_factor=10
pool_size=10
masked_lm_prob=0.4
prop_sliding_window=0.5
signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mppq${max_predictions_per_seq}-msl${max_seq_length}"

./.conda/python.exe gen_data.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --pool_size=${pool_size} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \