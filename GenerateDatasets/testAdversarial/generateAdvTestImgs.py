from keras.models import load_model
import numpy as np


def generateDatasets(model, dataset, n_labels, sample_per_label):
    for n in range(n_labels):
        noise = np.random.normal(0, 1, (sample_per_label, 100))
        labels = np.array([n for _ in range(sample_per_label)])
        gen_imgs = model.predict([noise, labels])
        np.save(f"datasets/{dataset}/{n}_images.npy", gen_imgs)


if __name__ == '__main__':
    
    # trainACGANs()

    n_labels = 10
    sample_per_label = 1000
    model_mnist = load_model("../datasetsB/models/mnist/generator_mnist_29800.model")
    model_fashion_mnist = load_model("../datasetsB/models/fashion_mnist/generator_fashion_mnist_28000.model")
    model_cifar10 = load_model("../datasetsB/models/cifar10/generator_cifar10_29800.model")
    generateDatasets(model=model_mnist, dataset="mnist", n_labels=n_labels, sample_per_label=sample_per_label)
    generateDatasets(model=model_fashion_mnist, dataset="fashion_mnist", n_labels=n_labels, sample_per_label=sample_per_label)
    generateDatasets(model=model_cifar10, dataset="cifar10", n_labels=n_labels, sample_per_label=sample_per_label)
