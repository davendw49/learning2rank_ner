from l2r import Rank
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import ujson

model = "Lambdarank"
train = np.load('./train.npy')
test = np.load('./test.npy')
benchmark = np.load('./benchmark.npy')
n_feature = 6
h1_units = 512
h2_units = 256
epoch = 2
lr = 0.0001
nt = 20
k = 1

def process_StandardScal(x):
    standard_scaler = preprocessing.StandardScaler().fit(x)
    x_train_mean =standard_scaler.mean_
    x_train_std = standard_scaler.scale_
    x_train_standard = standard_scaler.transform(x)
    return x_train_standard, x_train_mean, x_train_std

def process_StandardScal_test(x,mean,std):
    standard_scaler = preprocessing.StandardScaler().fit(x)
    standard_scaler.mean_ = mean
    standard_scaler.std_ = std
    x_test = standard_scaler.transform(x)
    return x_test

def evaluate(pred_result,true_result):
    pred_data = {}
    for val in pred_result.values():
        for item in val:
            pred_data[item[0]] = item[1]
    true_data = {}
    for res in true_result:
        true_data[res[0]] = res[1]
    return (factor(pred_data, true_data))

def factor(pred_data,true_data):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for key in pred_data.keys():
        if true_data[key]==1:
            tp += 1
        else:
            fp += 1
    for key in true_data.keys():
        if key not in pred_data.keys():
            if true_data[key]==1:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score

def plot(ndcg_record):
    x = np.array(list(ndcg_record.keys()))
    y = np.array(list(ndcg_record.values()))
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    axes.plot(x, y, 'r')
    plt.show()

train_len = train.shape[0]
data = np.concatenate((train,test),axis=0)
x_data,x_train_mean, x_train_std = process_StandardScal(data[:,3:])
data = np.hstack((data[:,:3],x_data))
train = data[:train_len,]
test = data[train_len:,]
x_benchmark = process_StandardScal_test(benchmark[:,3:],x_train_mean, x_train_std)
benchmark = np.hstack((benchmark[:,:3],x_benchmark))


ndcg_record = {}

rank = Rank(rank_model=model, training_data=train, n_feature=n_feature, h1_units=h1_units, h2_units=h2_units, epoch=epoch, lr=lr, number_of_trees=nt)
rank.handler.fit(k, ndcg_record)
with open('ndcg_record.json','w')as f:
    ujson.dump(ndcg_record,f)


test_pred_result = rank.handler.predict(test,k)
print(test_pred_result)
test_true_result = test[:, [2, 0]].astype(np.int32).tolist()
test_evaluate = evaluate(test_pred_result,test_true_result)
print("test evaluate: ",test_evaluate)


benchmark_pred_result = rank.handler.predict(benchmark,k)
benchmark_true_result = benchmark[:,[2,0]].astype(np.int32).tolist()
benchmark_evaluate = evaluate(benchmark_pred_result,benchmark_true_result)
print("benchmark evaluate: ",benchmark_evaluate)

