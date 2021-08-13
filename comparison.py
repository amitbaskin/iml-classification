import numpy as np
from pandas import DataFrame
from models import Model, Perceptron, SVM, LDA
import matplotlib.pyplot as plt
import time


WEIGHTS = [0.3, -0.5]
WEIGHTS_VEC = np.array([WEIGHTS])
INTERCEPT = 0.1
SAMPLES_AMOUNTS = [5, 10, 15, 25, 70]
TEST_AMOUNT = 10000
RUNS_AMOUNT = 500
MODELS = [Perceptron, SVM, LDA]


def draw_points_helper(samples_amount):

    def true_hypothesis(sample):
        return np.sign(WEIGHTS_VEC @ sample + INTERCEPT)

    identity_mat = np.identity(2)
    zero_vec = np.zeros((2,))

    samples = np.random.multivariate_normal(
        zero_vec, identity_mat, size=samples_amount).transpose()

    true_labels = np.array(
        [true_hypothesis(sample)[0] for sample in samples.transpose()])

    return samples, true_labels


def draw_points(samples_amount, helper):
    samples, true_labels = helper(samples_amount)
    max_sum = len(true_labels)
    current_sum = np.sum(true_labels)

    while current_sum == max_sum or current_sum == -max_sum:
        samples, true_labels = helper(samples_amount)
        current_sum = np.sum(true_labels)

    return samples.transpose(), true_labels


def q9_helper(samples_amount):
    samples, true_labels = \
        draw_points(samples_amount, draw_points_helper(samples_amount))
    xx = np.linspace(np.min(samples), np.max(samples))

    def get_yy(weights, intercept):
        slope = -weights[0] / weights[1]
        return slope * xx - intercept / weights[1]

    yy_f = get_yy(WEIGHTS, INTERCEPT)
    perceptron = Perceptron(samples, true_labels).model
    yy_perceptron = get_yy(perceptron[1:], perceptron[0])
    svm = SVM(samples, true_labels).model
    svm_weights = svm.coef_[0]
    yy_svm = get_yy(svm_weights, svm.intercept_[0])

    df = DataFrame({'x1': samples[:, 0], 'x2': samples[:, 1],
                    'class': true_labels})

    plt.scatter(df.loc[df['class'] == 1]['x1'],
                df.loc[df['class'] == 1]['x2'],
                color='blue', label='labeled positive')

    plt.scatter(df.loc[df['class'] == -1]['x1'],
                df.loc[df['class'] == -1]['x2'],
                color='orange', label='labeled negative')

    plt.plot(
        xx, yy_f, 'm', linewidth=2, alpha=0.7, label='true hypothesis')

    plt.plot(
        xx, yy_perceptron, '--r', linewidth=2, alpha=0.7,
        label='perceptron')

    plt.plot(
        xx, yy_svm, 'g', linewidth=2, alpha=0.7, label='svm')

    plt.title('comparing hyperplane classifiers for {} samples'
              '\nfrom a normal distribution'
              '\nwith the zero vector as the mean vector'
              '\nand the identity as the covariance matrix'
              .format(samples_amount))

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def q9():
    for amount in SAMPLES_AMOUNTS:
        q9_helper(amount)


# q9()


def get_accuracy(model, test_samples, test_true_labels):
    test_samples_amount = len(test_samples)
    pred = model.predict(test_samples)
    compare_mat = pred - test_true_labels
    accuracy_amount = test_samples_amount - np.count_nonzero(
        compare_mat)
    return accuracy_amount / test_samples_amount


def get_mean_accuracy(runs_amount, model, test_samples, test_true_labels):
    start_time = time.time()
    accuracy = 0
    for i in range(runs_amount):
        current_acc = get_accuracy(model, test_samples, test_true_labels)
        accuracy += current_acc
    print('  {}, {} training samples: {} seconds'.
          format(model.model.__class__.__name__,
                 model.training_amount,
                 round(time.time() - start_time, 2)))
    return accuracy / runs_amount


def get_train_samples_and_labels_dict(samples_amounts, draw_points_helper):
    samples_and_labels_dict = dict()
    for amount in samples_amounts:
        samples_and_labels_dict[amount] = \
            draw_points(amount, draw_points_helper)
    return samples_and_labels_dict


def get_model(model_class, train_samples, train_true_labels):
    return Model(model_class(train_samples, train_true_labels),
                 len(train_samples))


def get_model_versions(samples_amounts, model_class, samples_and_labels_dict):
    return [get_model(model_class, samples_and_labels_dict[amount][0],
                      samples_and_labels_dict[amount][1])
            for amount in samples_amounts]


def get_mean_accuracies(samples_amounts, runs_amount, model_class,
                        samples_and_labels_dict,
                        test_samples, test_true_labels):
    model_versions = get_model_versions(samples_amounts, model_class,
                                        samples_and_labels_dict)
    return [get_mean_accuracy(runs_amount, model, test_samples,
                              test_true_labels)
            for model in model_versions]


def question_helper(train_samples_and_labels_dict, test_samples,
                    test_true_labels, samples_amounts, models, runs_amount):
    for model_class in models:
        model_mean_accs = get_mean_accuracies(samples_amounts, runs_amount,
                     model_class, train_samples_and_labels_dict,
                     test_samples, test_true_labels)
        if model_class.__name__ == 'Logistic':
            plt.plot(samples_amounts, model_mean_accs, '--m', linewidth=4,
                     alpha=0.7, label='{}'.format(model_class.__name__))
        else:
            plt.plot(samples_amounts, model_mean_accs, linewidth=2,
                     label='{}'.format(model_class.__name__))
    plt.xlabel('Train Samples Amount')
    plt.ylabel('Mean Accuracies')
    plt.title('comparing the mean accuracies of classifying models'
              '\nas a function of the number of training samples')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def run_question(samples_amounts, models, runs_amount, draw_points_helper,
                 test_samples, test_true_labels):

    train_samples_and_labels_dict = \
        get_train_samples_and_labels_dict(samples_amounts, draw_points_helper)

    question_helper(train_samples_and_labels_dict, test_samples, test_true_labels,
                    samples_amounts, models, runs_amount)


def q10(samples_amounts, test_amount, models, runs_amount, draw_points_helper):

    test_samples, test_true_labels = draw_points(
        test_amount, draw_points_helper)

    run_question(samples_amounts, models, runs_amount,
                 draw_points_helper, test_samples, test_true_labels)


# q10(SAMPLES_AMOUNTS, TEST_AMOUNT, MODELS, RUNS_AMOUNT, draw_points_helper)
