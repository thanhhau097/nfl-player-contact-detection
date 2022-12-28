# Training tutorials

1. Download data and save to "./data" folder
2. K fold split
```
python kfold.py
```
3. Train
```
python -m torch.distributed.launch --nproc_per_node 2 train.py --output_dir ./outputs/ --num_train_epochs 10 --do_train --do_eval --metric_for_best_model eval_matthews_corrcoef --overwrite_output_dir --per_device_train_batch_size 16 --model_name resnet50 --fold 0 --dataloader_num_workers 64 --evaluation_strategy steps  --eval_steps 5000 --save_strategy steps --save_steps 5000
```

4. Find best threshold
```
CUDA_VSSIBLE_DEVICES=1 python train.py --output_dir ./outputs_infer/ --do_eval --per_device_eval_batch_size 64 --model_name resnet50 --fold 0 --dataloader_num_workers 32 --resume /home/thanh/shared_disk/thanh/nfl-player-contact-detection/outputs/checkpoint-185000/pytorch_model.bin
```

### Another command
```
python -m torch.distributed.launch --nproc_per_node 2 train.py --output_dir ./outputs_heatmap/ --num_train_epochs 10 --do_train --do_eval --metric_for_best_model eval_matthews_corrcoef --overwrite_output_dir --per_device_train_batch_size 8 --model_name resnet50 --fold 0 --dataloader_num_workers 1 --evaluation_strategy steps  --eval_steps 5000 --save_strategy steps --save_steps 5000 --learning_rate 1e-3
```