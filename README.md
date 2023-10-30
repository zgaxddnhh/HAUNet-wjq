# 复现实验
2023-10-23更新整理！！！

## 原文指标
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.46|0.9333|0.6437|0.0488|论文|
|UCx3|HAUNet|30.34|0.8476|0.4236|0.0779|论文|
|UCx4|HAUNet|28.06|0.7726|0.2932|0.0997|论文|

## 137复现结果（无随机种子，batch_size=4）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.64|0.9349|0.6503|0.0478|137-HAUNETx2_UCMerced|
|UCx3|HAUNet|30.30|0.8487|0.4257|0.0779|137-HAUNETx3_UCMerced|
|UCx4|HAUNet|28.01|0.771|0.2900|0.0999|137-HAUNETx4_UCMerced|

## auto复现结果（无随机种子，batch_size=8）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.49|0.9335|0.6452|0.0485|auto-HAUNETx2_UCMerced|
|UCx3|HAUNet|30.20|0.8469|0.4192|0.0788|auto-HAUNETx3_UCMerced|
|UCx4|HAUNet|28.00|0.770|0.2890|0.10045|auto-HAUNETx4_UCMerced|

## auto复现结果（无随机种子，batch_size=4）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx3|HAUNet|30.36|0.8485|0.4275|0.0777|auto-HAUNETx3_UCMerced_b4|

## auto复现结果（随机种子=1，batch_size=8，lr=0.0008）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.56|0.9340|0.6465|0.0483|auto-HAUNETx2_UCMerced_1|
|UCx3|HAUNet|30.25|0.8471|0.4221|0.0786|auto-HAUNETx3_UCMerced_1|
|UCx4|HAUNet|27.97|0.7708|0.2883|0.1007|auto-HAUNETx4_UCMerced_1|

## auto复现结果（随机种子=1，batch_size=8，lr=0.0011）\
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.503664|0.933529|0.645193|0.048577|auto-HAUNETx2_UCMerced_s1_b8|
|UCx3|HAUNet|30.372846|0.848352|0.426527|0.077667|auto-HAUNETx3_UCMerced_s1_b8|
|UCx4|HAUNet|28.033755|0.772385|0.292654|0.100051|auto-HAUNETx4_UCMerced_s1_b8|

### lr=0.0016 4倍超分 HAUNETx4_UCMerced_s1_b8_lr16
Average: PSNR: 27.910537 dB, SSIM: 0.768060, SAM: 0.101240, QI: 0.990916, SCC: 0.283339


> *按照各种复现实验的结果来看，在接下来的实验中，设置随机种子=1，batch_size=8，lr=0.0011进行实验。*
# Train
```bash
# x4
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=HAUNETx4_UCMerced 
# x3
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=HAUNETx3_UCMerced
# x2
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=HAUNETx2_UCMerced
```
输入LR的大小被裁剪为:48*48，同时有一个数据预处理（包括随机水平、垂直翻转、随机旋转90°，以及添加噪声）。

# Test
```bash
# debug模式
python demo_deploy.py --scale=2 --model=HAUNET --patch_size=128 --test_block --pre_train=/home/wjq/wjqHD/RSISR/model-zoo/HAUNet_RSISR/experiment/HAUNETx2_UCMerced/model/model_best.pt --dir_data=/home/wjq/wjqHD/RSISR/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/home/wjq/wjqHD/RSISR/HAUNet-wjq/experiment/HAUNETx2_UCMerced_debug/results
# x2
python demo_deploy.py --scale=2 --model=HAUNET_WJQ --patch_size=128 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
# x3
python demo_deploy.py --scale=3 --model=HAUNET --patch_size=192 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
# x4
python demo_deploy.py --scale=4 --model=HAUNET_WJQ --patch_size=256 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
```
以`64x64`为block进行测试。2倍时`pathch_size=128`，3倍时`patch_size=192`，4倍时`patch_size=256`。

### 评估指标
```bash
cd metric_scripts 
python calculate_metric.py
```
