from l2r import Rank
from sklearn import preprocessing
import numpy as np

model = "Lambdarank"
train = np.load('./train.npy')
test = np.load('./test.npy')
n_feature = 6
h1_units = 512
h2_units = 256
epoch = 10000
lr = 0.0001
nt = 20

def process_StandardScal(x):
    standard_scaler = preprocessing.StandardScaler()
    x_train_standard = standard_scaler.fit_transform(x)
    return x_train_standard

train_len = train.shape[0]
data = np.concatenate((train,test),axis=0)
data = process_StandardScal(data)
train = data[:train_len,]
test = data[train_len:,]
ndcg3_list =[]

rank = Rank(rank_model=model, training_data=train, n_feature=n_feature, h1_units=h1_units, h2_units=h2_units, epoch=epoch, lr=lr, number_of_trees=nt)
rank.handler.fit(3,ndcg3_list)
ndcg_list = rank.handler.validate(test, 3)
print(ndcg_list)
print(max(ndcg3_list,key=itemgetter(-1))[-1])
print(rank.handler.predict(test))
