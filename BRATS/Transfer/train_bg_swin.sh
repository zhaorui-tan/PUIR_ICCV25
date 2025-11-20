
/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/tanzhaorui/test/swinunetr/bin/python main_bg_swin.py \
--json_list='/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/tanzhaorui/test/data_process/BRATS23_GLI.json' --world_size 1 --crop_foreground --distributed --port 12101 \
--data_dir='/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/dataplatform/lifescience/BRATS23'  \
--feature_size=48 --noamp --val_every 10 --in_channels 4 --out_channels 4 --optim_lr 0.0002 \
--logdir 'train_recon_2bs_4cuda_vis_fixed_bg_swin/' --save_every 10 \
--roi_x=96 --roi_y=96 --roi_z=96 --batch_size=1 --max_epochs=2000 --save_checkpoint 