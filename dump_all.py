import torch
from l2r import Rank
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True)

model = "Lambdarank"
train = np.load('./train.npy')
test = np.load('./test.npy')
benchmark = np.load('./benchmark.npy')
alldata = np.load('./alldata.npy')
n_feature = 6
h1_units = 512
h2_units = 256
epoch = 10
lr = 0.0001
nt = 20
k = 1

if __name__ == '__main__':
    rank = Rank(rank_model=model, training_data=train, n_feature=n_feature, h1_units=h1_units, h2_units=h2_units, epoch=epoch, lr=lr, number_of_trees=nt)
    rank.handler.model.load_state_dict(torch.load('model.pth'))
    predict_result_score, predict_result_pair = rank.predict(alldata, 1)
    print(predict_result_pair)
