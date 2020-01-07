python tools/make_mri_dataset.py \
    --raw_data_root /mnt/data1/MRIData \
    --record_root /mnt/data1/MRIRec \
    --phase artery \
    --nb_samples 80000 --samples_per_record 1000 \
    --resample_size 320 256 72 \
    --patch_size 64 \
    --test_rate 0.2
