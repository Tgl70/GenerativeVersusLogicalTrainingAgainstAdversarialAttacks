#!/bin/bash

# DL2
python main.py --dl2-weight 0.2 --dataset mnist --dtype "datasetA"
python main.py --dl2-weight 0.2 --dataset fashion_mnist --dtype "datasetA"
python main.py --dl2-weight 0.1 --dataset cifar10 --dtype "datasetA"

# DatasetsA
python main.py --dl2-weight 0 --dataset mnist --dtype "datasetA"
python main.py --dl2-weight 0 --dataset fashion_mnist --dtype "datasetA"
python main.py --dl2-weight 0 --dataset cifar10 --dtype "datasetA"

# DatasetsB
python main.py --dl2-weight 0 --dataset mnist --dtype "datasetB"
python main.py --dl2-weight 0 --dataset fashion_mnist --dtype "datasetB"
python main.py --dl2-weight 0 --dataset cifar10 --dtype "datasetB"

# DatasetsC
python main.py --dl2-weight 0 --dataset mnist --dtype "datasetC"
python main.py --dl2-weight 0 --dataset fashion_mnist --dtype "datasetC"
python main.py --dl2-weight 0 --dataset cifar10 --dtype "datasetC"





# DL2
python main.py --num-epochs 2 --dl2-weight 0.2 --dataset mnist  --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --dtype "datasetA"
python main.py --num-epochs 2 --dl2-weight 0.2 --dataset fashion_mnist  --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --dtype "datasetA"
python main.py --num-epochs 2 --dl2-weight 0.04 --dataset cifar10  --constraint "RobustnessT(eps1=13.8, eps2=0.9)" --dtype "datasetA"

# Datasets
python main.py --num-epochs 2 --dl2-weight 0 --dataset PH --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --dtype "datasetA"
python main.py --num-epochs 2 --dl2-weight 0 --dataset PH --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --dtype "datasetB"
python main.py --num-epochs 2 --dl2-weight 0 --dataset PH --constraint "RobustnessT(eps1=7.8, eps2=2.9)" --dtype "datasetC"

python results.py --folder reports
