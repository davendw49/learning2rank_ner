from l2r import Rank

model = "LambdaMART"
train = './train.npy'
test = './test.npy'
n_feature = 6
h1_units = 512
h2_units = 256
epoch = 10
lr = 0.0001
nt = 20

rank = Rank(rank_model=model, training_data=train, n_feature=n_feature, h1_units=h1_units, h2_units=h2_units, epoch=epoch, lr=lr, number_of_trees=nt)
rank.handler.fit(3)
# ndcg_list = rank.handler.validate(test, 3)
# print(ndcg_list)

print(rank.handler.predict(test))