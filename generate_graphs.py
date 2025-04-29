import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from network import LeNet5
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dataset import MnistDataloader
from networkx.algorithms.community import greedy_modularity_communities
from visualize_network import Neural_Net_Graph
import re

def throughout_training_visualizations():
    model_paths = ["model_weights/untrained_8.pt", "model_weights/one_epoch_8.pt", "model_weights/fully_trained_8.pt"]

    for i in range(2):
        if i == 1:
            for model_path in model_paths:
                model = LeNet5(reduction_factor=8)
                model.load_model(model_path)
                mnist_loader = MnistDataloader('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte',
                                                'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')
                train_dataset, test_dataset = mnist_loader.load_dataset()

                # print(test_dataset)
                plt.figure(figsize=(18, 10))        
                LeNetGraph = Neural_Net_Graph(model)
                LeNetGraph.compute_activations(test_dataset)
                LeNetGraph.build_graph(weights="pearson_correlation")

                match = re.search(r'([^-/]+)\.pt$', model_path)
                LeNetGraph.draw_graph_page_rank(pathname= f"plots/PR_{match.group(1)}_full_dataset_8.png", h_spacing= 4, v_spacing= 8 , node_size_frac= 10)

                plt.clf()

        else:
             for model_path in model_paths:
                model = LeNet5(reduction_factor=8)
                model.load_model(model_path)
                mnist_loader = MnistDataloader('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte',
                                                'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')
                train_dataset, test_dataset = mnist_loader.load_dataset()

                # print(test_dataset)
                plt.figure(figsize=(18, 10))        
                LeNetGraph = Neural_Net_Graph(model)
                LeNetGraph.compute_activations(test_dataset)
                LeNetGraph.build_graph(weights="pearson_correlation")
                communities = list(greedy_modularity_communities(LeNetGraph.G, weight='weight'))


                match = re.search(r'([^-/]+)\.pt$', model_path)
                LeNetGraph.draw_graph_communities(pathname= f"plots/com_{match.group(1)}_full_dataset_8.png", communities=communities, h_spacing= 4, v_spacing= 8)

                plt.clf()


def pr_visualizations():
    model = LeNet5(reduction_factor=8)
    model.load_model("model_weights/fully_trained_8.pt")

    #Generate the page rank visualizations for fully trained network
    for i in range(11):
        mnist_loader = MnistDataloader('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte',
                                        'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')
        train_dataset, test_dataset = mnist_loader.load_dataset(include_only=[i])

        # print(test_dataset)
        plt.figure(figsize=(18, 10))        
        LeNetGraph = Neural_Net_Graph(model)
        LeNetGraph.compute_activations(test_dataset)
        LeNetGraph.build_graph(weights="pearson_correlation")
        LeNetGraph.draw_graph_page_rank(pathname= f"plots/PR_fullytrained_{i}dataset_8.png", h_spacing= 4, v_spacing= 8 , node_size_frac= 10)

        plt.clf()

def even_odd_visualizations():
    model_paths = ["model_weights/evenodd_fully_trained_8.pt", "model_weights/fully_trained_8.pt"]

    odd_vals = [1,3,5,7,9]
    even_vals = [0,2,4,6,8]

    for indx, model_path in enumerate(model_paths):
        num_classes = 10 if indx == 1 else 2

        model = LeNet5(reduction_factor=8, num_classes=num_classes )
        model.load_model(model_path)

        for i in range(2):
            mnist_loader = MnistDataloader('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte',
                                            'data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')
            include_arr = odd_vals if i == 0 else even_vals
            train_dataset, test_dataset = mnist_loader.load_dataset(include_only=include_arr)

            plt.figure(figsize=(18, 10))        
            LeNetGraph = Neural_Net_Graph(model)
            LeNetGraph.compute_activations(test_dataset)
            LeNetGraph.build_graph(weights="pearson_correlation")
            communities = list(greedy_modularity_communities(LeNetGraph.G, weight='weight'))

            LeNetGraph.draw_graph_communities(communities=communities, pathname=f"plots/com_fullytrained_{indx}{i}_8.png", h_spacing= 4, v_spacing= 8)

            plt.clf()


if __name__ == "__main__":
    throughout_training_visualizations()
    pr_visualizations()
    even_odd_visualizations()
    