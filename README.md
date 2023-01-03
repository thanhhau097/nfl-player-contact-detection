# Training tutorials

1. Download data and save to "./data" folder
2. K fold split
```
python kfold.py
```
3. Train
```
export NEPTUNE_PROJECT="thanhhau097/nfl"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTRjM2ExOC1lYTA5LTQwODctODMxNi1jZjEzMjdlMjkxYTgifQ=="
```

```
python -m torch.distributed.launch --nproc_per_node 2 train.py --logging_strategy steps --logging_steps 20 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --output_dir ./outputs/ --num_train_epochs 20 --do_train --do_eval --metric_for_best_model eval_matthews_corrcoef --overwrite_output_dir --per_device_train_batch_size 32 --model_name resnet50 --fold 0 --dataloader_num_workers 32 --evaluation_strategy epoch --save_strategy epoch --learning_rate 1e-3 --save_total_limit 3 --fp16 --load_best_model_at_end --num_cache 120000
```

4. Find best threshold
```
CUDA_VISIBLE_DEVICES=0 python train.py --output_dir ./outputs_infer/ --do_eval --per_device_eval_batch_size 64 --model_name resnet50 --fold 0 --dataloader_num_workers 32 --resume /home/thanh/shared_disk/thanh/nfl-player-contact-detection/outputs/checkpoint-185000/pytorch_model.bin
```


5. Debug
```
CUDA_VISIBLE_DEVICES=1 python train.py --output_dir ./outputs_frame_features/ --evaluation_strategy epoch --save_strategy epoch --save_total_limit 2  --logging_strategy steps --logging_steps 20 --fp16 --warmup_ratio 0.1 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_matthews_corrcoef  --model_name resnet50 --fold 0 --dataloader_num_workers 24 --learning_rate 2e-4  --num_train_epochs 20 --num_frames 5 --frame_steps 4 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --remove_unused_columns False --overwrite_output_dir --report_to neptune --load_best_model_at_end
```

### Another command


# Requirements
```
pip install PyTurboJPEG
sudo apt-get install libturbojpeg
```

# TODO: 
- [ ] Verify preprocessing time of training on whole image (check resize/augmentation/model-forward time), because if it costs too much time, we can't inference: using turbojpeg, need to update in inference kernel
- [x] Add more features: https://www.kaggle.com/code/ahmedelfazouan/nfl-player-contact-detection-helmet-track-ftrs#Helmet-track-Features: training
- [ ] Add more center frames to the image input
- [ ] Add (attention) layer to focus more on center layer:
    - Idea from @nhan
        - hoặc 2.5d + conv3d block ở sau feature map cuối
        - non-local block: chèn thêm block đấy sau last feature map, attention cross frames cross views luôn
- [ ] MCC loss function: https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/370723
- [ ] Full image with heatmap: in code but didn't train