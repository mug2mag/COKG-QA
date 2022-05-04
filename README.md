## Data
下载知识图谱数据、问答数据和RoBERTa模型：  
链接: https://pan.baidu.com/s/1OPoJlsm4Wjt7Aio4xxSOMQ 提取码: 592x 

## KG Embedding Module

进入COKG/OpenKE，训练实体嵌入和schema嵌入
```
cd OpenKE
bash prepare_embedding.sh
```
## Question Embedding Module
训练COKG-QA模型
```
cd COKG
python main.py --mode train --relation_dim 200 --do_batch_norm 1 --gpu 0 --freeze 1 --batch_size 90 --validate_every 1 --lr 0.00005 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 --decay 0.9 --model ComplEx --patience 1 --ls 0.05 --l3_reg 0.00005 --nb_epochs 200 --outfile split --dataset gethered
```
