import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from pandas import *


def model_score(model, samples, true_labels):
    return Model(model, model.training_amount).score(samples, true_labels)


class Model:
    model = None
    training_amount = 0

    def __init__(self, model, training_amount):
        self.model = model
        self.training_amount = training_amount

    def predict(self, samples):
        return self.model.predict(samples)

    def score(self, samples, true_labels):
        samples_amount = len(samples.transpose())
        predictions = self.predict(samples)
        compare_mat = predictions - true_labels
        accuracy_amount = samples_amount - np.count_nonzero(compare_mat)
        accuracy_rate = accuracy_amount / samples_amount
        error_amount = samples_amount - accuracy_amount
        error_rate = error_amount / samples_amount

        false_positive_mat = (compare_mat > 0)
        false_positive_amount = np.sum(false_positive_mat)
        false_positive_rate = false_positive_amount / samples_amount

        predicted_positive = (predictions == 1)
        labeled_positive = (true_labels == 1)
        true_positive_mat = (predicted_positive == labeled_positive)
        true_positive_amount = np.sum(true_positive_mat)
        true_positive_rate = true_positive_amount / samples_amount
        predicted_true_amount = np.sum(predicted_positive)
        precision = true_positive_amount / predicted_true_amount
        labeled_true_amount = np.sum(labeled_positive)
        recall = true_positive_amount / labeled_true_amount

        scores = dict()
        scores['num_samples'] = samples_amount
        scores['error'] = error_rate
        scores['accuracy'] = accuracy_rate
        scores['FPR'] = false_positive_rate
        scores['TPR'] = true_positive_rate
        scores['precision'] = precision
        scores['recall'] = recall

        return scores


def get_conditions_mat(samples, labels):
    samples = add_ones_col(samples)
    conditions_mat = np.array(
        [samples[i] * labels[i] for i in range(len(labels))])
    return conditions_mat


def add_ones_col(samples):
    samples_amount = len(samples)
    new_row = np.ones((samples_amount, 1))
    samples = np.concatenate((new_row, samples), axis=1)
    return samples


class Perceptron:
    model = None
    training_amount = 0

    def __init__(self, samples, labels):
        self.training_amount = len(samples)
        self.fit(samples, labels)

    def fit(self, samples, labels):
        conditions_mat = get_conditions_mat(samples, labels)
        row_len = conditions_mat.shape[1]
        weights = np.zeros((row_len,))
        min_val = 0
        while min_val <= 0:
            current_result = np.matmul(conditions_mat, weights)
            min_val_index = np.argmin(current_result)
            min_val = current_result[min_val_index]
            if min_val <= 0:
                min_vec = conditions_mat[min_val_index]
                weights += min_vec
                continue
        self.model = weights

    def predict(self, samples):
        samples = add_ones_col(samples)
        return np.sign(np.matmul(samples, self.model))

    def score(self, samples, true_labels):
        return model_score(self, samples, true_labels)


class LDA:
    model = None
    training_amount = 0

    def __init__(self, samples, labels):
        self.training_amount = len(samples)
        self.fit(samples, labels)

    def fit(self, samples, labels):

        labeled_positive_indices = list((labels == 1).nonzero())[0]
        labeled_negative_indices = list((labels == -1).nonzero())[0]

        positive_samples = samples[labeled_positive_indices, :]
        negative_samples = samples[labeled_negative_indices, :]

        positive_mean = np.mean(positive_samples, axis=0)
        negative_mean = np.mean(negative_samples, axis=0)

        samples_amount = len(samples)
        total_mean = np.mean(samples, axis=1)
        samples_centered = (samples.transpose() - total_mean).transpose()
        positive_samples = samples_centered[labeled_positive_indices, :]
        negative_samples = samples_centered[labeled_negative_indices, :]

        positive_cov = positive_samples.transpose() @ positive_samples
        negative_cov = negative_samples.transpose() @ negative_samples

        general_scalar = 1 / samples_amount
        general_cov = general_scalar * (positive_cov + negative_cov)
        general_cov_inverse = np.linalg.pinv(general_cov)

        positive_amount = len(positive_samples)
        negative_amount = len(negative_samples)
        positive_prob = positive_amount / samples_amount
        negative_prob = negative_amount / samples_amount

        def delta_label(sample, label_mean, label_prob):
            first = sample @ general_cov_inverse @ label_mean \
                   - 0.5 * label_mean.transpose() @ general_cov_inverse @ \
                   label_mean
            return first + np.log(label_prob)

        def delta_positive(sample):
            return delta_label(sample, positive_mean, positive_prob)

        def delta_negative(sample):
            return delta_label(sample, negative_mean, negative_prob)

        def max_delta(sample):
            return max(delta_positive(sample), delta_negative(sample))

        def get_label(sample):
            if max_delta(sample) == delta_positive(sample):
                return 1
            else:
                return -1

        def get_predictions(new_samples):
            return np.array([get_label(sample) for sample in new_samples])

        self.model = get_predictions

    def predict(self, samples):
        return self.model(samples)

    def score(self, samples, true_labels):
        return model_score(self, samples, true_labels)


class SVM:
    model = SVC(C=1e10, kernel='linear')
    training_amount = 0

    def __init__(self, samples, labels):
        self.training_amount = len(samples)
        self.fit(samples, labels)

    def fit(self, samples, labels):
        self.model.fit(samples, labels)

    def predict(self, samples):
        return self.model.predict(samples)


class Logistic:
    model = LogisticRegression(solver='liblinear')
    training_amount = 0

    def __init__(self, samples, labels):
        self.training_amount = len(samples)
        self.fit(samples, labels)

    def fit(self, samples, labels):
        self.model.fit(samples, labels)

    def predict(self, samples):
        return self.model.predict(samples)

    def score(self, samples, true_labels):
        return model_score(self, samples, true_labels)


class DecisionTree:
    model = None
    training_amount = 0

    def __init__(self, samples, labels):
        self.training_amount = len(samples)
        self.model = DecisionTreeClassifier(max_depth=5)
        self.fit(samples, labels)

    def fit(self, samples, labels):
        self.model.fit(samples, labels)

    def predict(self, samples):
        return self.model.predict(samples)

    def score(self, samples, true_labels):
        return model_score(self, samples, true_labels)
