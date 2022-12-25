# Training tutorials

1. Download data and save to "./data" folder
2. K fold split
```
python kfold.py
```
3. Train
```
python train.py --output_dir ./outputs/ --num_train_epochs 10 --do_train --do_eval --metric_for_best_model eval_AUC --overwrite_output_dir --per_device_train_batch_size 16 --model_name resnet50 --fold 0 --dataloader_num_workers 64 --evaluation_strategy steps  --eval_steps 5000 --save_strategy steps --save_steps 5000
```