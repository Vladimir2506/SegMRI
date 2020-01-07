python tools/make_mri_hybrid_dataset.py \
    --raw_data_root /mnt/data1/MRIData \
    --record_root /mnt/data1/MRIRecHybrid \
    --phase portal \
    --nb_samples 50000 --samples_per_record 1000 \
    --resample_size 320 256 72 \
    --patch_size 3 \
    --test_rate 0.2
