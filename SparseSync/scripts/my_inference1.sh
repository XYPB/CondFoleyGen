set -x

python predict_best_sync.py -d 0 --split 0 --dest_dir logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_tolerance --tolerance 0.2
python predict_best_sync.py -d 0 --split 1 --dest_dir logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_tolerance --tolerance 0.2
python predict_best_sync.py -d 0 --split 2 --dest_dir logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_tolerance --tolerance 0.2
python predict_best_sync.py -d 0 --split 3 --dest_dir logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_tolerance --tolerance 0.2
python predict_best_sync.py -d 0 --split 4 --dest_dir logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_tolerance --tolerance 0.2