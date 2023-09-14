### News

**2023-09-14** 为服务器运行程序创建分支

### 复现指标
|scale|PSNR|SSIM|SCC|SAM|
|:---:|:---:|:---:|:---:|:---:|
|UCx2|34.64|0.9349|0.6503|0.0478|
|UCx3|30.30|0.8487|0.4257|0.0779|
|UCx4|28.01|0.771|0.2900|0.0999|

和原论文中的指标进行对比，整体来说复现差距很小，复现成功。


### Train
```bash
# x4
CUDA_VISIBLE_DEVICES=0 python demo_train.py --model=HAUNET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=HAUNETx4_UCMerced 
# x2
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=HAUNETx2_UCMerced
```
输入LR的大小被裁剪为:48*48，同时有一个数据预处理（包括随机水平、垂直翻转、随机旋转90°，以及添加噪声）。

添加wandb后的Train
```bash
# x4
CUDA_VISIBLE_DEVICES=0 python demo_train.py --project_name=SRx4 --model=HAUNET_V1 --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=HAUNETV1x4_UCMerced 
```


### Test
```bash
python demo_deploy.py --scale=2 --model=HAUNET --patch_size=128 --test_block --pre_train=/home/wjq/wjqHD/RSISR/model-zoo/HAUNet_RSISR/experiment/HAUNETx2_UCMerced/model/model_best.pt --dir_data=/home/wjq/wjqHD/RSISR/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/home/wjq/wjqHD/RSISR/HAUNet-wjq/experiment/HAUNETx4_UCMerced_debug/results
```
以64x64为block进行测试。