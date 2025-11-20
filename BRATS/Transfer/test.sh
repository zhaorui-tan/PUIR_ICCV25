export CUDA_VISIBLE_DEVICES=0
python test_test.py \
--json_list='data_split.json' --data_dir='path/to/data/' \
--feature_size=48 --in_channels 4 --out_channels 4 --exp_name cropped_autopet_nii  --crop_foreground \
--roi_x=96 --roi_y=96 --roi_z=96 \
--pretrained_dir runs/path --pretrained_model_name model.pt