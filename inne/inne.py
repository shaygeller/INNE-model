from sklearn.neighbors import NearestNeighbors
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool


class INNE:
    def __init__(self, n_samples=5, sample_size=10000, n_jobs=1, random_state=1234, verbose=False):
        """
        :param n_samples: int. Number of samples to draw from the fitted dataset.
        :param sample_size: int. Size of each sample.
        :param n_jobs: int. Number of parallel jobs. The number of jobs to run in parallel. fit, predict, are all
        parallelized over the samples.
        :param random_state: Controls both the randomness of the bootstrapping of the samples used
        when building samples.
        :param verbose: True/False. Controls the verbosity when fitting and predicting.
        """
        self.nn_clfs = []
        self.sample_nn_radiuses_li = []
        self.sample_nn_indexes_li = []
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.map = Pool(nodes=n_jobs).map
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X):
        self.samples = self.create_samples(X, self.n_samples, self.sample_size, self.random_state)

        results = self.map(self.do_fit_parallel, range(self.n_samples))

        for result in results:
            nn_clf, sample_nn_radiuses, sample_nn_indexes = result
            self.nn_clfs.append(nn_clf)
            self.sample_nn_radiuses_li.append(sample_nn_radiuses)
            self.sample_nn_indexes_li.append(sample_nn_indexes)

    def predict(self, pred_samples):
        self.pred_samples = pred_samples

        # get them to work in parallel
        sample_scores = self.map(self.do_predict_parallel, range(self.n_samples))

        # Average results over all the samples
        anomaly_scores = np.stack(sample_scores).mean(axis=0)
        return anomaly_scores

    @staticmethod
    def create_samples(X_data, n_samples, sample_size, random_seed):
        # Make things reproducible
        np.random.seed(random_seed)

        samples = []
        for i in range(n_samples):
            sample = X_data[np.random.choice(range(X_data.shape[0]), sample_size, replace=False)]
            samples.append(sample)
        samples = np.stack(samples, axis=0)
        return samples

    def do_fit_parallel(self, sample_index):
        if self.verbose:
            print(f"Creating sample: {sample_index}")

        sample = self.samples[sample_index, :]
        nn_clf = NearestNeighbors(n_neighbors=2)
        nn_clf.fit(sample)

        # Get radius of all the instances in the sample. Instance's radius = distance to its NN. First NN is self.
        nns = nn_clf.kneighbors(sample, 2, return_distance=True)
        sample_nn_distances, sample_nn_indexes = nns[0][:, 1], nns[1][:, 1]

        return nn_clf, sample_nn_distances, sample_nn_indexes

    def do_predict_parallel(self, sample_index):
        if self.verbose:
            print(f"Predicting using sample: {sample_index}")

        curr_nn_clf = self.nn_clfs[sample_index]
        curr_sample_nn_radiuses = self.sample_nn_radiuses_li[sample_index]
        curr_sample_nn_indexes = self.sample_nn_indexes_li[sample_index]

        # Second find NN of test instances
        nns = curr_nn_clf.kneighbors(self.pred_samples, 2, return_distance=True)
        pred_nn_radiuses, pred_nn_index = nns[0][:, 1], nns[1][:, 1]

        # Calculate anomaly scores - Closes to 1, more anomalous
        # Get radius of NN of the NN of the pred instances
        nn_of_pred_nn_indexes = curr_sample_nn_indexes[pred_nn_index]
        sample_nn_nn_radiuses = curr_sample_nn_radiuses[nn_of_pred_nn_indexes]

        # Case 1 - pred instances are inside their NN radius - calc score 1-(pred-NN-NN-radius/pred-NN-radius)
        # Here we calculate if for all of the instances, we will later recalculate case 2 scores are edit their score
        score = np.ones(self.pred_samples.shape[0]) - (sample_nn_nn_radiuses / curr_sample_nn_radiuses[pred_nn_index])

        # Case 2 - All pred instances that are farther than their nn radius, get max score of 1.
        score[pred_nn_radiuses > curr_sample_nn_radiuses[pred_nn_index]] = 1

        # Deal with cases where pred-NN-radius is 0, so we divided by 0.
        # Following the alg, these cases should be set to 0
        score = np.nan_to_num(score, nan=0.0)
        return score
