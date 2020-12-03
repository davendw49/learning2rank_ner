# learning2rank_ner

L2R algorithms implemted by **Python**.

this repository contains **RankNet** , **LambdaRank** and **LambdaMART**

## RankNet

**Pytorch** implementation

- para:
  - `n_feaure`: int, features numble
  - `h1_units`: int,  the unit numbers of hidden layer1
  - `h2_units`: int, the unit numbers of hidden layer2
  - `epoch`: int, iteration times
  - `learning_rate`: float, learning rate
  - `plot`: boolean, whether plot the loss.

## LambdaRank

The usage is similar with RankNet.

## Dataset

store in `train.npy` and `test.npy`. Used `np.load()` to import dataset.

The first column is `label`, the second column is `qid`, and the following columns are features (total 6 features).

## REFERENCE

- [L2R](https://github.com/houchenyu/L2R)
- [PTRanking](https://github.com/wildltr/ptranking)