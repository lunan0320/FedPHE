# Efficient and Straggler-Resistant Homomorphic Encryption for Heterogeneous Federated Learning [[paper](https://liyuqingwhu.github.io/lyq/papers/INFOCOM2024-yan.pdf)]

> Nan Yan, Yuqing Li, Jing Chen, Xiong Wang, Jianan Hong, Kun He, and Wei Wang. *IEEE INFOCOM 2024* 

![](https://cdn.jsdelivr.net/gh/lunan0320/pics@main/images/202403/image-20240320201923223.png)

## News 

- [2024.03.20] FedPHE source code has been released

## Abstract

Cross-silo federated learning (FL) enables multiple institutions (clients) to collaboratively build a global model without sharing their private data. To prevent privacy leakage during aggregation, homomorphic encryption (HE) is widely used to encrypt model updates, yet incurs high computation and communication overheads. To reduce these overheads, packed HE (PHE) has been proposed to encrypt multiple plaintexts into a single ciphertext. However, the original design of PHE does not consider the heterogeneity among different clients, an intrinsic problem in cross-silo FL, often resulting in undermined training efficiency with slow convergence and stragglers. In this work, we propose FedPHE, an efficiently packed homomorphically encrypted FL framework with secure weighted aggregation and client selection to tackle the heterogeneity problem. Specifically, using CKKS with sparsification, FedPHE can achieve efficient encrypted weighted aggregation by accounting for contributions of local updates to the global model. To mitigate the straggler effect, we devise a sketching-based client selection scheme to cherry-pick representative clients with heterogeneous models and computing capabilities. We show, through rigorous security analysis and extensive experiments, that FedPHE can efficiently safeguard clients’ privacy, achieve a training speedup of 1.85 − 4.44×, cut the communication overhead by 1.24 − 22.62×, and reduce the straggler effect by up to 1.71 − 2.39×.

## Citation

> If you find FedPHE useful or relevant to your research, please kindly cite our paper using the following bibtex.

```
@InProceedings{Yan2024FedPHE,
  author  = {Yan, Nan and Li, Yuqing and Chen, Jing and Wang, Xiong and Hong, Jianan and He, Kun and Wang, Wei},
  booktitle = {Proc. IEEE INFOCOM},
  title   = {Efficient and Straggler-Resistant Homomorphic Encryption for Heterogeneous Federated Learning},
  year    = {2024},
}
```

## Folder Structure

```
├── workspace  
│   └── data
│   └── log
│   └── encryption  
|   |   └── paillier,bfv,ckks
│   ├── models  
│   ├── utils  
│   │   └── dataset 
│   │   └── min_hash 
│   ├── client  
│   ├── server    
│   ├── fed
│   ├── main
```

## Usage

### Installation

```
# create an environment called "FedPHE"
conda create -n FedPHE python
conda activate FedPHE

# git clone the repo first
git clone git@github.com:lunan0320/FedPHE.git

cd FedPHE
mkdir data, log, data_dir

# install the correct packages required
pip install -r requirements.txt
```

### Run

```
python main.py --dataset {dataset_name} --epochs {epoch_num} --lr {learing_rate} --n_clients {client_num} --topk {sparse_rate} --algorithm {HE_algo} --enc_batch_size {pack_size} --sim_len {hash_func_num} 
```

### Reproduce our results

In this repository, you will find all the necessary components to reproduce the results from our research. The example instruction is outlined below:

```
python main.py --dataset MNIST --epochs 100 --lr 0.001 --n_clients 8 --topk 0.1 --algorithm ckks --enc_batch_size 4096 --sim_len 200 --enc True --isSelection True --isSpars topk
```

1. The package size for `paillier`, `bfv` and `ckks` are different. We always set `80` for paillier and `4096` for bfv and ckks. 
2. You can adaptively choose whether to `encrypt`, whether to `select`, whether to `sparse`, etc.
3. You can choose which encryption method to use. If you want to calculate a more accurate time cost, you can set `--cipher_count False` (there is also a certain time cost for counting cipher traffic)
4. Please make sure your `GPU` has enough memory, `inter-process communication` ciphertext may cause `memory overflow`

> Note: In order to communicate and synchronize between processes, the code uses the `torch.multiprocessing` library. We have set up more `locks`, `events`, `values`, `pipes`, and `queues`. If you are a novice in this area, do not modify these switches, as it may cause running failures.

## Results

- We observe that FedPHE reduces the network footprint for MNIST, FashionMNIST, and CIFAR-10 by up to 16.49×, 16.89×, and 3.31×, respectively, compared to PackedCKKS. Moreover, it outperforms PackedBFV for 4.28−22.62× across three datasets. It is worth noting that the ciphertext size is only 0.81×, 0.77×, and 4.11× compared to the BatchCrypt. This indicates the efficiency of FedPHE in reducing the ciphertext generated by CKKS to the level of BatchCrypt encryption with Paillier. This achievement is truly remarkable. Additionally, the ciphertext size, which is previously in “memory out” state as shown in Table I, has been reduced to only 2.07 − 9.88× larger than the plaintext baseline, making FedPHE applicable to FL in practice. In conclusion, FedPHE achieves communication overhead reduction ranging from 1.24× to 22.62× compared to these baselines.

-  As shown in Table III, BatchCrypt requires 4.02 − 7.01× more training time compared to plaintext. In contrast, FedPHE incurs only 1.58 − 2.17× training time of the plaintext baseline, greatly enhancing the efficiency of model training. Furthermore, leveraging sparsification and client selection, FedPHE achieves a training acceleration of 1.85−4.44×. With an apt sparsification threshold, FedPHE does not adversely affect the trained model quality. Instead, it achieves significant compression while maintaining high performance.

![](https://cdn.jsdelivr.net/gh/lunan0320/pics@main/images/202403/image-20240320222216046.png)





