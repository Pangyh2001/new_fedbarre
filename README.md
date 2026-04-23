这个代码有DP,fedavg,fednfl,fedRPF,barre算法，fedRPF,barre是我加的，你可以帮我看下逻辑有没有问题。



# 运行FedAvg算法
```bash
python main.py 
```


# 运行RPF算法+优化噪声
```bash
python main.py --use_rp True  --nfl "eps=0.1,privacy=rpf,distort=rpf," --out_dir "./runs_RPF" --global_epoch 100 --noise_type 2
```
```bash
python main.py --use_rp True  --nfl "eps=1,privacy=rpf,distort=rpf" --out_dir "./runs_RPF&noise2" --global_epoch 100 --noise_type 2 --rp_ratio 0.3 --rp_eps 1
```

代码中的fedRPF0.sh是跑仅随机投影滤波器的实验，fedRPF1是跑随机投影滤波器+随机噪声的实验，fedRPF2是随机投影滤波器+优化噪声的实验。

# 运行BARRE算法
```bash
python main.py --nfl "eps=0.7,privacy=barre,distort=barre,barre_noise_type=4" --out_dir "./runs_barre2"  --use_rp False
```

代码中的fedbarre1.sh是跑多分类器（3个分类器）+随机噪声的实验，fedbarre4_2.sh是跑多分类器（5个分类器）+优化噪声的实验，fedbarre4_3.sh是跑多分类器（9个分类器）+随机噪声的实验的实验。fedbarre4_2.sh是跑多分类器（3个分类器）+优化噪声的实验

# 运行DP算法
```bash
python main.py --nfl "eps=0.3,privacy=dp,distort=dp-laplace,clipDP=1.0" --out_dir "./runs_dp" --local_epoch 1  --global_epoch 100 --use_rp False
```


DP.sh是DP算法的代码




# 运行NFL算法
```bash
python main.py --nfl "eps=0.1, privacy=nfl, distort=nfl,clipDP=1.0" --out_dir "./runs_nfl" --local_epoch 1 --global_epoch 100 --use_rp False
```
fednfl.sh是fednfl的代码。