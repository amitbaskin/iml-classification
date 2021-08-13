import matplotlib.pyplot as plt
from models import *
from comparison import run_question
from sklearn.neighbors import NearestCentroid


class NearestNeighbors:
    model = NearestCentroid()

    def __init__(self, samples, true_labels):
        self.model.fit(samples, true_labels)

    def predict(self, samples):
        return self.model.predict(samples)


SAMPLES_AMOUNTS = [50, 100, 300, 500]
RUNS_AMOUNT = 50
MODELS = [Logistic, SVM, DecisionTree, NearestNeighbors]


image_size = 28
no_of_different_labels = 2
image_pixels = image_size * image_size
data_path = r'C:\Users\amitb\PycharmProjects\IML\ex3\dataset\mldata'
train_data = np.loadtxt(data_path + r"\mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + r"\mnist_test.csv",
                       delimiter=",")

fac = 0.99 / 255
all_train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
all_test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

all_train_labels = np.asfarray(train_data[:, :1])
all_test_labels = np.asfarray(test_data[:, :1])

binary_train_indices = np.logical_or((all_train_labels == 0),
                                    (all_train_labels == 1))
binary_test_indices = np.logical_or((all_test_labels == 0),
                                    (all_test_labels == 1))

binary_train_indices = binary_train_indices.reshape(
     binary_train_indices.shape[0])
binary_test_indices = binary_test_indices.reshape(
    binary_test_indices.shape[0])

x_train, y_train = all_train_imgs[binary_train_indices], \
                   all_train_labels[binary_train_indices]
x_test, y_test = all_test_imgs[binary_test_indices], \
                 all_test_labels[binary_test_indices]

y_train = y_train.reshape((len(y_train),))
y_test = y_test.reshape((len(y_test),))
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1


def q12():
    positive_counter = 0
    negative_counter = 0
    while positive_counter < 3 or negative_counter < 3:
        for i in range(len(x_train)):
            img = x_train[i].reshape((image_size, image_size))
            if y_train[i][0] == 1 and positive_counter < 3:
                plt.imshow(img)
                plt.show()
                positive_counter += 1
            elif y_train[i][0] == -1 and negative_counter < 3:
                plt.imshow(img)
                plt.show()
                negative_counter += 1


# q12()


def q14_draw_points_helper(samples_amount):
    true_arr = np.ones(samples_amount, dtype=bool)
    false_arr = np.zeros(len(x_train) - samples_amount, dtype=bool)
    indices_arr = np.concatenate((true_arr, false_arr))
    np.random.shuffle(indices_arr)
    train_samples = x_train[indices_arr]
    true_labels = y_train[indices_arr]
    return train_samples.transpose(), true_labels


def q14():
    run_question(SAMPLES_AMOUNTS, MODELS, RUNS_AMOUNT,
                 q14_draw_points_helper, x_test, y_test)


# q14()
