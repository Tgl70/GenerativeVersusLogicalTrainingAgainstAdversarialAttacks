from __future__ import print_function, division
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ACGAN:
    def __init__(self, rows=28, cols=28, channels=1):
        # Input shape
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        d1 = int(self.img_rows/4)
        d2 = int(self.img_cols/4)

        model.add(Dense(128 * d1 * d2, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((d1, d2, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, dataset, batch_size=128, sample_interval=50):

        if dataset == "gtsrb":
            x_train = np.load(f"../datasetsA/datasets/{dataset}/X_train.npy")
            y_train = np.load(f"../datasetsA/datasets/{dataset}/y_train.npy")
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
            # [0, 1] --> [-1, 1]
            x_train = (x_train.astype(np.float32) - 0.5) * 2

        else:
            (x_train, y_train), (_, _) = eval(dataset).load_data()
            # [0, 255] --> [-1, 1]
            x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        
        x_train = np.expand_dims(x_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of imagesACGAN
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))

            # Generate a half batch of new imagesACGAN
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9
            img_labels = y_train[idx]

            # Train the discriminator
            if dataset == "cifar10":
                imgs = imgs.reshape(32, 32, 32, 3)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]"
                  % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and epoch > 20000:
                self.save_model(epoch, dataset)
                self.sample_images(epoch, dataset)


    def sample_images(self, epoch, dataset):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale imagesACGAN 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if dataset == "cifar10":
                    axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
                else:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"imagesACGAN/{dataset}/{epoch}.png")
        plt.close()


    def save_model(self, epoch, dataset):
        self.generator.save(f"models/{dataset}/generator_{dataset}_{epoch}.model")
        self.discriminator.save(f"models/{dataset}/discriminator_{dataset}_{epoch}.model")


def trainACGANs():
    '''
    mnist_acgan = ACGAN(rows=28, cols=28, channels=1)
    mnist_acgan.train(epochs=30001, batch_size=32, sample_interval=200, dataset="mnist")

    fashion_mnist_acgan = ACGAN(rows=28, cols=28, channels=1)
    fashion_mnist_acgan.train(epochs=30001, batch_size=32, sample_interval=200, dataset="fashion_mnist")

    cifar10_acgan = ACGAN(rows=32, cols=32, channels=3)
    cifar10_acgan.train(epochs=30001, batch_size=32, sample_interval=200, dataset="cifar10")
    '''
    gtsrb_acgan = ACGAN(rows=48, cols=48, channels=1)
    gtsrb_acgan.train(epochs=30001, batch_size=32, sample_interval=200, dataset="gtsrb")


def generateDatasets(model, dataset, n_labels, sample_per_label):
    for n in range(n_labels):
        noise = np.random.normal(0, 1, (sample_per_label, 100))
        labels = np.array([n for _ in range(sample_per_label)])
        gen_imgs = model.predict([noise, labels])
        np.save(f"datasets/{dataset}/{n}_images.npy", gen_imgs)

        for i in range(len(gen_imgs)):
            if dataset == "cifar10":
                img = gen_imgs[i, :, :]
                img = Image.fromarray(np.uint8(img * 255), 'RGB')
            else:
                img = gen_imgs[i, :, :, 0]
                img = Image.fromarray(np.uint8(img * 255), 'L')
            img.save(f"datasets/{dataset}/{n}/{i}.png")


def showDatasets(mnist=False, fashion_mnist=False, cifar10=False, gtsrb=False):
    print("mnist")
    for n in range(10):
        imgs = np.load(f"datasets/mnist/{n}_images.npy")
        print(imgs.shape)
        if mnist:
            for img in imgs:
                plt.imshow(img[:, :, 0], cmap='gray')
                plt.show()

    print("fashion_mnist")
    for n in range(10):
        imgs = np.load(f"datasets/fashion_mnist/{n}_images.npy")
        print(imgs.shape)
        if fashion_mnist:
            for img in imgs:
                plt.imshow(img[:, :, 0], cmap='gray')
                plt.show()

    print("cifar10")
    for n in range(10):
        imgs = np.load(f"datasets/cifar10/{n}_images.npy")
        print(imgs.shape)
        if cifar10:
            for img in imgs:
                plt.imshow(img[:, :], cmap='gray')
                plt.show()

    print("gtsrb")
    for n in range(10):
        imgs = np.load(f"datasets/gtsrb/{n}_images.npy")
        print(imgs.shape)
        if gtsrb:
            for img in imgs:
                plt.imshow(img[:, :], cmap='gray')
                plt.show()


if __name__ == '__main__':
    
    trainACGANs()
    
    n_labels = 10
    sample_per_label = 10000
    model_mnist = load_model("models/mnist/generator_mnist_29600.model")
    model_fashion_mnist = load_model("models/fashion_mnist/generator_fashion_mnist_27800.model")
    model_cifar10 = load_model("models/cifar10/generator_cifar10_30000.model")
    model_gtsrb = load_model("models/gtsrb/generator_gtsrb_29800.model")
    generateDatasets(model=model_mnist, dataset="mnist", n_labels=n_labels, sample_per_label=sample_per_label)
    generateDatasets(model=model_fashion_mnist, dataset="fashion_mnist", n_labels=n_labels, sample_per_label=sample_per_label)
    generateDatasets(model=model_cifar10, dataset="cifar10", n_labels=n_labels, sample_per_label=sample_per_label)
    generateDatasets(model=model_gtsrb, dataset="gtsrb", n_labels=n_labels, sample_per_label=sample_per_label)

    showDatasets(mnist=False, fashion_mnist=False, cifar10=False, gtsrb=False)
    