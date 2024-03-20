# Efficient and Straggler-Resistant Homomorphic Encryption for Heterogeneous Federated Learning

> Nan Yan, Yuqing Li, Jing Chen, Xiong Wang, Jianan Hong, Kun He, and Wei Wang. *IEEE INFOCOM 2024* ([link](https://liyuqingwhu.github.io/lyq/papers/INFOCOM2024-yan.pdf))

![](https://cdn.jsdelivr.net/gh/lunan0320/pics@main/images/202403/image-20240320201923223.png)

## News 

- [2024.03.28] FedPHE source code has released

## Abstract

Cross-silo federated learning (FL) enables multiple institutions (clients) to collaboratively build a global model without sharing their private data. To prevent privacy leakage during aggregation, homomorphic encryption (HE) is widely used to encrypt model updates, yet incurs high computation and communication overheads. To reduce these overheads, packed HE (PHE) has been proposed to encrypt multiple plaintexts into a single ciphertext. However, the original design of PHE does not consider the heterogeneity among different clients, an intrinsic problem in cross-silo FL, often resulting in undermined training efficiency with slow convergence and stragglers. In this work, we propose FedPHE, an efficiently packed homomorphically encrypted FL framework with secure weighted aggregation and client selection to tackle the heterogeneity problem. Specifically, using CKKS with sparsification, FedPHE can achieve efficient encrypted weighted aggregation by accounting for contributions of local updates to the global model. To mitigate the straggler effect, we devise a sketching-based client selection scheme to cherry-pick representative clients with heterogeneous models and computing capabilities. We show, through rigorous security analysis and extensive experiments, that FedPHE can efficiently safeguard clients’ privacy, achieve a training speedup of 1.85 − 4.44×, cut the communication overhead by 1.24 − 22.62×, and reduce the straggler effect by up to 1.71 − 2.39×.

## Citation

> You can cite the paper using the following bibtex.

```
@InProceedings{Yan2024FedPHE,
  author  = {Yan, Nan and Li, Yuqing and Chen, Jing and Wang, Xiong and Hong, Jianan and He, Kun and Wang, Wei},
  booktitle = {Proc. IEEE INFOCOM},
  title   = {Efficient and Straggler-Resistant Homomorphic Encryption for Heterogeneous Federated Learning},
  year    = {2024},
}
```

## How to install

```
# git clone the repo first
# create an environment called "FedPHE"

# install the correct packages required
pip install -r requirements.txt
```



## How to use FedPHE





