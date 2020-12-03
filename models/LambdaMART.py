from .utils import *

from . import BaseModel


class LambdaMART(BaseModel):

    def __init__(self, training_data, n_feature, h1_units, h2_units, epoch, lr, number_of_trees=10, plot=True):
        self.training_data = np.load(training_data)
        self.number_of_trees = number_of_trees
        self.lr = lr
        self.trees = []

    def fit(self, topk):
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
        predicted_scores = np.zeros(sample_num)
        for k in range(self.number_of_trees):
            print('Tree %d' % k)
            lambdas = np.zeros(sample_num)
            w = np.zeros(sample_num)

            temp_score = [predicted_scores[qid_doc_map[qid]] for qid in query_idx]
            zip_parameters = zip(true_scores, temp_score, order_paris, query_idx)
            for ts, temps, op, qi in zip_parameters:
                sub_lambda, sub_w, qid = compute_lambda(ts, temps, op, qi)
                lambdas[qid_doc_map[qid]] = sub_lambda
                w[qid_doc_map[qid]] = sub_w
            tree = DecisionTreeRegressor(max_depth=50)
            tree.fit(self.training_data[:, 2:], lambdas)
            self.trees.append(tree)
            pred = tree.predict(self.training_data[:, 2:])
            predicted_scores += self.lr * pred

            # print NDCG
            qid_doc_map = group_by(self.training_data, 1)
            ndcg_list = []
            for qid in qid_doc_map.keys():
                subset = qid_doc_map[qid]
                sub_pred_score = predicted_scores[subset]

                # calculate the predicted NDCG
                true_label = self.training_data[qid_doc_map[qid], 0]
                # topk = len(true_label)
                pred_sort_index = np.argsort(sub_pred_score)[::-1]
                true_label = true_label[pred_sort_index]
                ndcg_val = ndcg_k(true_label, topk)
                ndcg_list.append(ndcg_val)
            print('Epoch:{}, Average NDCG@{} : {}'.format(k, topk, np.nanmean(ndcg_list)))

    def predict(self, data):
        """
        predict the score for each document in testset
        :param data: given testset
        :return:
        """
        data = np.load(data)
        qid_doc_map = group_by(data, 1)
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            sub_result = np.zeros(len(qid_doc_map[qid]))
            for tree in self.trees:
                sub_result += self.lr * tree.predict(data[qid_doc_map[qid], 2:])
            predicted_scores[qid_doc_map[qid]] = sub_result
        return predicted_scores

    def validate(self, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        data = np.load(data)
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        for qid in qid_doc_map.keys():
            sub_pred_result = np.zeros(len(qid_doc_map[qid]))
            for tree in self.trees:
                sub_pred_result += self.lr * tree.predict(data[qid_doc_map[qid], 2:])
            predicted_scores[qid_doc_map[qid]] = sub_pred_result
            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]
            pred_sort_index = np.argsort(sub_pred_result)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        return ndcg_list


"""
if __name__ == '__main__':
    training_data = np.load('./dataset/train.npy')
    model = LambdaMART(training_data, 20, 0.01)
    model.fit()

    k = 4
    test_data = np.load('./dataset/test.npy')
    ndcg = model.validate(test_data, k)
    print(ndcg)
"""
