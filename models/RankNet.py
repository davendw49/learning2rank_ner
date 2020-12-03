from .utils import *
from . import BaseModel


class Model(torch.nn.Module):
    """
    construct the RankNet
    """

    def __init__(self, n_feature, h1_units, h2_units):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            # h_1
            torch.nn.Linear(n_feature, h1_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            # h_2
            torch.nn.Linear(h1_units, h2_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            # output
            torch.nn.Linear(h2_units, 1),
        )
        self.output_sig = torch.nn.Sigmoid()

    def forward(self, input_1, input_2):
        s1 = self.model(input_1)
        s2 = self.model(input_2)
        out = self.output_sig(s1 - s2)
        return out

    def predict(self, input_):
        s = self.model(input_)
        n = s.data.numpy()[0]
        return n


class RankNet(BaseModel):
    """
    user interface
    """

    def __init__(self, training_data, n_feature, h1_units, h2_units, epoch, lr, number_of_trees=10, plot=True):
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.model = Model(n_feature, h1_units, h2_units)
        self.epoch = epoch
        self.plot = plot
        self.learning_rate = lr
        self.training_data = np.load(training_data)

    def fit(self, k=0):
        training_data = self.training_data
        """
        train the RankNet based on training data.
        After training, save the parameters of RankNet, named 'parameters.pkl'
        :param training_data:
        """
        net = self.model
        qid_doc_map = group_by(training_data, 1)
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        true_scores = [training_data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        relevant_doc, irrelevant_doc = split_pairs(order_paris, true_scores)
        relevant_doc = training_data[relevant_doc]
        irrelevant_doc = training_data[irrelevant_doc]

        X1 = relevant_doc[:, 2:]
        X2 = irrelevant_doc[:, 2:]
        y = np.ones((X1.shape[0], 1))

        # training......
        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
        y = torch.Tensor(y)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        loss_fun = torch.nn.BCELoss()

        loss_list = []

        if self.plot:
            plt.ion()

        print('Traning………………\n')
        for i in range(self.epoch):
            decay_learning_rate(optimizer, i, 0.95)

            net.zero_grad()
            y_pred = net(X1, X2)
            loss = loss_fun(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.numpy())
            if self.plot:
                plt.cla()
                plt.plot(range(i + 1), loss_list, 'r-', lw=5)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.pause(1)
            if i % 10 == 0:
                print('Epoch:{}, loss : {}'.format(i, loss.item()))

        if self.plot:
            plt.ioff()
            plt.show()

        # save model parameters
        torch.save(net.state_dict(), 'parameters.pkl')

    def validate(self, data, k):
        """
        compute the average NDCG@k for the given test data.
        :param test_data: test data
        :param k: used to compute NDCG@k
        :return:
        """
        data = np.load(data)
        # load model parameters
        net = Model(self.n_feature, self.h1_units, self.h2_units)
        net.load_state_dict(torch.load('parameters.pkl'))

        qid_doc_map = group_by(data, 1)
        query_idx = qid_doc_map.keys()
        ndcg_k_list = []

        for q in query_idx:
            true_scores = data[qid_doc_map[q], 0]
            if sum(true_scores) == 0:
                continue
            docs = data[qid_doc_map[q]]
            X_test = docs[:, 2:]

            pred_scores = [net.predict(torch.Tensor(test_x).data) for test_x in X_test]
            pred_rank = np.argsort(pred_scores)[::-1]
            pred_rank_score = true_scores[pred_rank]
            ndcg_val = ndcg_k(pred_rank_score, k)
            ndcg_k_list.append(ndcg_val)
        print("Average NDCG@{} is {}".format(k, np.mean(ndcg_k_list)))

        return ndcg_k_list


"""
if __name__ == '__main__':
    print('Load training data...')
    training_data = np.load('./dataset/train.npy')
    print('Load done.\n\n')

    model1 = RankNet(46, 512, 256, 100, 0.01, True)
    model1.fit(training_data)

    print('Validate...')
    test_data = np.load('./dataset/test.npy')
    model1.validate(test_data)
"""
