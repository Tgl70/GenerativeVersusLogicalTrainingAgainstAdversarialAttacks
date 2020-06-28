#!/bin/bash

# Create datasets
python create_datasetsA.py
python acgan.py
python create_datasetsC.py
python generateAdvTestImgs.py

# DL2
python main.py --dl2-weight 0.2 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset mnist --dtype "datasetA"
python main.py --dl2-weight 0.2 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset fashion_mnist --dtype "datasetA"
python main.py --dl2-weight 0.1 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset gtsrb --dtype "datasetA"
python main.py --dl2-weight 0.1 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset cifar10 --dtype "datasetA"

# DatasetsA
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset mnist --dtype "datasetA"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset fashion_mnist --dtype "datasetA"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset gtsrb --dtype "datasetA"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset cifar10 --dtype "datasetA"

# DatasetsB
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset mnist --dtype "datasetB"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset fashion_mnist --dtype "datasetB"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset gtsrb --dtype "datasetB"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset cifar10 --dtype "datasetB"

# DatasetsC
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset mnist --dtype "datasetC"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset fashion_mnist --dtype "datasetC"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset gtsrb --dtype "datasetC"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.3, delta=0.52)" --dataset cifar10 --dtype "datasetC"

# Datasets A, B and C with eps = 0.01
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset mnist --dtype "datasetA"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset mnist --dtype "datasetB"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset mnist --dtype "datasetC"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset fashion_mnist --dtype "datasetA"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset fashion_mnist --dtype "datasetB"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset fashion_mnist --dtype "datasetC"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset gtsrb --dtype "datasetA"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset gtsrb --dtype "datasetB"
python main.py --dl2-weight 0 --constraint "RobustnessG(eps=0.01, delta=0.52)" --dataset gtsrb --dtype "datasetC"


# Attack the models
python ACGAN_Attack.py
python FGSM_Attack.py
