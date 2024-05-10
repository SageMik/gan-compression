@REM 获取图像的FID评分
@REM python .\get_real_stat.py ^
@REM --dataroot ..\..\Dataset\IC ^
@REM --dataset_mode single ^
@REM --output_path .\IC-FID.npz ^
@REM --gpu_ids -1

@REM 训练主模型，模型colorization默认dataset_mode为colorization
@REM python train.py --dataroot ../../Dataset/IC ^
@REM --model colorization ^
@REM --real_stat_path IC-FID.npz ^
@REM --batch_size 32 ^
@REM --gpu_ids -1

@REM 训练超网
python train_supernet.py ^
--dataroot ..\..\Dataset\COCO2017\val ^
--dataset_mode colorization ^
--supernet resnet ^
--batch_size 4 ^
--teacher_netG resnet_9blocks ^
--restore_teacher_G_path ../checkpoints/coco/latest_net_G.pth ^
--real_stat_path IC-FID.npz ^
--norm instance ^
--teacher_dropout_rate 0.5 ^
--nepochs 10 --nepochs_decay 30 ^
--teacher_ngf 64 --student_ngf 64 ^
--config_set channels-64-pix2pix ^
--gpu_ids -1