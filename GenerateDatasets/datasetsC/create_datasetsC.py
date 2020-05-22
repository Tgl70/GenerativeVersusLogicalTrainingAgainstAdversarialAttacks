import numpy as np
import random


def mod_data(dataset, distance, X, Y, n):
    X_mod = []
    Y_mod = []

    for i in range(len(X)):
        x_mod = []
        y_mod = []
        if dataset == 'mnist' or dataset == 'fashion_mnist':
            # Create 5 new images per image 'x'
            while len(x_mod) < (n/2):
                x = X[i].copy()
                # Randomly modify numbers until a certain euclidean distance from the original image is reached
                while True:
                    # Random positive modification number
                    d = random.uniform(0, 0.1)
                    # Random pixel
                    px = random.randint(0, 28 - 1)
                    py = random.randint(0, 28 - 1)
                    # Modify the pixel if it stays between [-1, 1]
                    if x[px][py] + d >= 0 and x[px][py] + d <= 1:
                        x[px][py] += d
                    # Calculatin euclidean distance
                    euclidean_distance = 0
                    for j in range(len(X[i])):
                        for k in range(len(x)):
                            euclidean_distance += (X[i][j][k][0] - x[j][k][0]) ** 2
                    euclidean_distance = np.sqrt(euclidean_distance)
                    if euclidean_distance > distance:
                        x_mod.append(x)
                        y_mod.append(Y[i])
                        break

            # Create another 5 new images per image 'x'
            while len(x_mod) < n:
                x = X[i].copy()
                # Randomly modify numbers until a certain euclidean distance from the original image is reached
                while True:
                    # Random negative modification number
                    d = random.uniform(-0.1, 0)
                    # Random pixel
                    px = random.randint(0, 28 - 1)
                    py = random.randint(0, 28 - 1)
                    # Modify the pixel if it stays between [-1, 1]
                    if x[px][py] + d >= 0 and x[px][py] + d <= 1:
                        x[px][py] += d
                    # Calculatin euclidean distance
                    euclidean_distance = 0
                    for j in range(len(X[i])):
                        for k in range(len(x)):
                            euclidean_distance += (X[i][j][k][0] - x[j][k][0]) ** 2
                    euclidean_distance = np.sqrt(euclidean_distance)
                    if euclidean_distance > distance:
                        x_mod.append(x)
                        y_mod.append(Y[i])
                        break

        if dataset == 'cifar10':
            # Create 5 new images per image 'x'
            while len(x_mod) < (n/2):
                x = X[i].copy()
                # Randomly modify numbers until a certain euclidean distance from the original image is reached
                while True:
                    # Random positive modification number
                    d = random.uniform(0, 0.1)
                    # Random pixel
                    px = random.randint(0, 32 - 1)
                    py = random.randint(0, 32 - 1)
                    pz = random.randint(0, 3 - 1)
                    # Modify the pixel if it stays between [-1, 1]
                    if x[px][py][pz] + d >= 0 and x[px][py][pz] + d <= 1:
                        x[px][py][pz] += d
                    # Calculatin euclidean distance
                    euclidean_distance = 0
                    for j in range(len(X[i])):
                        for k in range(len(x)):
                            for z in range(3):
                                euclidean_distance += (X[i][j][k][z] - x[j][k][z]) ** 2
                    euclidean_distance = np.sqrt(euclidean_distance)
                    if euclidean_distance > distance:
                        x_mod.append(x)
                        y_mod.append(Y[i])
                        break

            # Create another 5 new images per image 'x'
            while len(x_mod) < n:
                x = X[i].copy()
                # Randomly modify numbers until a certain euclidean distance from the original image is reached
                while True:
                    # Random negative modification number
                    d = random.uniform(-0.1, 0)
                    # Random pixel
                    px = random.randint(0, 32 - 1)
                    py = random.randint(0, 32 - 1)
                    pz = random.randint(0, 3 - 1)
                    # Modify the pixel if it stays between [-1, 1]
                    if x[px][py][pz] + d >= 0 and x[px][py][pz] + d <= 1:
                        x[px][py][pz] += d
                    # Calculatin euclidean distance
                    euclidean_distance = 0
                    for j in range(len(X[i])):
                        for k in range(len(x)):
                            for z in range(3):
                                euclidean_distance += (X[i][j][k][z] - x[j][k][z]) ** 2
                    euclidean_distance = np.sqrt(euclidean_distance)
                    if euclidean_distance > distance:
                        x_mod.append(x)
                        y_mod.append(Y[i])
                        break

        for x in x_mod:
            X_mod.append(x)
        for y in y_mod:
            Y_mod.append(y)

        if i % 1000 == 0:
            print(i)

    return X_mod, Y_mod


def create_data(dataset, distance, n):
    print(f"\nLoading {dataset} dataset.")

    X_train = np.load(f"../datasetsA/datasets/{dataset}/X_train.npy")
    y_train = np.load(f"../datasetsA/datasets/{dataset}/y_train.npy")
    X_test = np.load(f"../datasetsA/datasets/{dataset}/X_test.npy")
    y_test = np.load(f"../datasetsA/datasets/{dataset}/y_test.npy")

    print(f"Creating new {dataset} dataset.")

    X_train_mod, y_train_mod = mod_data(dataset, distance, X_train, y_train, n)
    X_test_mod, y_test_mod = mod_data(dataset, distance, X_test, y_test, n)

    print(f"Saving new {dataset} dataset.")
    print(f"New {dataset} dataset X_train_mod len: {len(X_train_mod)}")
    print(f"New {dataset} dataset X_test_mod len: {len(X_test_mod)}")

    np.save(f"datasets/{dataset}/X_train.npy", X_train_mod)
    np.save(f"datasets/{dataset}/y_train.npy", y_train_mod)
    np.save(f"datasets/{dataset}/X_test.npy", X_test_mod)
    np.save(f"datasets/{dataset}/y_test.npy", y_test_mod)


if __name__ == "__main__":
    create_data('mnist', 0.2, n=2)
    create_data('fashion_mnist', 0.3, n=2)
    create_data('cifar10', 0.4, n=2)
