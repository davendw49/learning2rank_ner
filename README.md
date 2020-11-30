# learning2rank_ner

This repository is used to share some L2R algorithms implemted by **Python**.

So far, this repository contains **RankNet** , **LambdaRank** and **LambdaMART**

## RankNet

I utilize **Pytorch** to implement the network structure.

In order to use the interface, you should input following parameters:

- `n_feaure`: int, features numble
- `h1_units`: int,  the unit numbers of hidden layer1
- `h2_units`: int, the unit numbers of hidden layer2
- `epoch`: int, iteration times
- `learning_rate`: float, learning rate
- `plot`: boolean, whether plot the loss.

## LambdaRank

The usage is similar with RankNet.

## Dataset

store in `train.npy` and `test.npy`. You can directly used `np.load()` to import dataset.

The first column is `label`, the second column is `qid`, and the following columns are features (total 46 features).

