from .utils import *
from .Net import Net
from . import BaseModel
from torch.utils.data import DataLoader


class LambdaRank(BaseModel):

    def __init__(self, training_data, n_feature, h1_units, h2_units, epoch, lr=0.001, number_of_trees=10, plot=True):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.epoch = epoch
        self.lr = lr
        self.trees = []
        self.model = Net(n_feature, h1_units, h2_units)

    def get_parameters(self):
        for para in self.model.parameters():
            print(para[0])

    def fit(self, k, ndcg_record):
        """
        train the model to fit the train dataset
        """
        qid_doc_map = group_by(self.training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        true_scores = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        sample_num = len(self.training_data)
        print('Training .....\n')
        for i in range(self.epoch):
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 3:].astype(np.float32)))
            predicted_scores_numpy = predicted_scores.data.numpy()
            lambdas = np.zeros(sample_num)
            # w = np.zeros(sample_num)

            pred_score = [predicted_scores_numpy[qid_doc_map[qid]] for qid in query_idx]

            zip_parameters = zip(true_scores, pred_score, order_paris, query_idx)
            for ts, ps, op, qi in zip_parameters:
                sub_lambda, sub_w, qid = compute_lambda(ts, ps, op, qi)
                lambdas[qid_doc_map[qid]] = sub_lambda
                # w[qid_doc_map[qid]] = sub_w
            # update parameters
            self.model.zero_grad()
            lambdas_torch = torch.Tensor(lambdas).view((len(lambdas), 1))
            predicted_scores.backward(lambdas_torch,
                                      retain_graph=True)  # This is very important. Please understand why?
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data.add_(param.grad.data * self.lr)

            if i % 1 == 0:
                qid_doc_map = group_by(self.training_data, 1)
                ndcg_list = []
                for qid in qid_doc_map.keys():
                    subset = qid_doc_map[qid]

                    X_subset = torch.from_numpy(self.training_data[subset, 3:].astype(np.float32))
                    sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()

                    # calculate the predicted NDCG
                    true_label = self.training_data[qid_doc_map[qid], 0]
                    # k = len(true_label)
                    pred_sort_index = np.argsort(sub_pred_score)[::-1]
                    true_label = true_label[pred_sort_index]
                    ndcg_val = ndcg_k(true_label, k)
                    ndcg_list.append(ndcg_val)
                print('Epoch:{}, Average NDCG@{} : {}'.format(i, k, np.nanmean(ndcg_list)))
                ndcg_record[i] = np.nanmean(ndcg_list)

                torch.save(self.model.state_dict() ,'model.pth')

    def predict(self, data, k):
        """
        predict the score for each document in testset
        :param data: given testset
        :return:
        """
        #data = np.load(data)
        qid_top3_dict = {}
        qid_doc_map = group_by(data, 1)
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 3:].astype(np.float32))
            sub_pred_score = self.model(X_subset).data.numpy().reshape(1,len(X_subset)).flatten()
            sub_ae_id  = data[subset,2]
            sub_ae_pre_score = [(int(x[0]),x[1]) for x in zip(sub_ae_id,sub_pred_score)]
            qid_top3_dict[qid] = sorted(sub_ae_pre_score,key=lambda x:x[1])[::-1][:k]
            predicted_scores[qid_doc_map[qid]] = sub_pred_score
        predicted_scores_aeid= {}
        data_ae_id = data[:,2]
        for i in range(0,len(predicted_scores)):
            predicted_scores_aeid[int(data_ae_id[i])] = predicted_scores[i]
        return qid_top3_dict, predicted_scores, predicted_scores_aeid

    def validate(self, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        #data = np.load(data)
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 3:].astype(np.float32))
            sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()

            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]
            # k = len(true_label)
            pred_sort_index = np.argsort(sub_pred_score)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        return ndcg_list


"""
if __name__ == '__main__':
    training_data = np.load('./dataset/train.npy')
    n_feature = training_data.shape[1] - 2
    h1_units = 512
    h2_units = 256
    epoch = 10
    learning_rate = 0.0001
    model = LambdaRank(training_data, n_feature, h1_units, h2_units, epoch, learning_rate)
    model.fit()
    k = 4
    test_data = np.load('./dataset/test.npy')
    ndcg = model.validate(test_data, k)
    print(ndcg)
"""
