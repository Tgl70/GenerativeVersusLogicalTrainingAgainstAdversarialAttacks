from keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np


def create_data(dataset):
    (X_train, y_train), (X_test, y_test) = eval(dataset).load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train_mod = []
    X_test_mod = []

    for x in X_train:
        if dataset == 'mnist' or dataset == 'fashion_mnist':
            x = x.reshape(28, 28, 1)
        X_train_mod.append(x)

    for x in X_test:
        if dataset == 'mnist' or dataset == 'fashion_mnist':
            x = x.reshape(28, 28, 1)
        X_test_mod.append(x)

    print(dataset)
    print(len(X_train_mod))
    print(X_train_mod[0].shape)
    print(len(X_test_mod))
    print(X_test_mod[0].shape)

    np.save(f"datasets/{dataset}/X_train.npy", X_train_mod)
    np.save(f"datasets/{dataset}/y_train.npy", y_train)
    np.save(f"datasets/{dataset}/X_test.npy", X_test_mod)
    np.save(f"datasets/{dataset}/y_test.npy", y_test)


if __name__ == "__main__":
    create_data("mnist")
    create_data("fashion_mnist")
    create_data("cifar10")
