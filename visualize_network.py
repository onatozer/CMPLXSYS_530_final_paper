import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from network import LeNet5
from pyvis_visualize import draw_interactive_graph
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dataset import MnistDataloader

class Neural_Net_Graph():
    def __init__(self, model):
        self.G = nx.DiGraph()
        self.layer_counter = 0
        # self.layers = []
        self.model = model
        self.added_input_layer = False
        self.prev_nodes = None
        
        self.prev_lin_layer = None
        self.prev_conv_layer = None

        self.activation_dict = {}
        self.activation_mat = []


    def add_edges(self, prev_nodes, curr_nodes, weights=None):
        for j, dst in enumerate(curr_nodes):
            for i, src in enumerate(prev_nodes):
                if weights is not None:
                    self.G.add_edge(src, dst, weight=float(weights[j][i]))
                else:
                    self.G.add_edge(src, dst, weight=1.0)

    #TODO: Change this func if you want to change the node names :-)
    def add_layer_nodes(self, num_nodes, role):
        nodes = [f"L{self.layer_counter}_N{i}" for i in range(num_nodes)]
        print(f"Adding nodes {nodes}")
        for n in nodes:
            self.G.add_node(n, layer=f"Layer {self.layer_counter}", role=role, layer_index=self.layer_counter)
        # self.layers.append(nodes)
        self.layer_counter += 1
        return nodes

    # Flatten out nested Sequential modules
    def iterate_layers(self, m):
        for name, layer in m.named_children():
            if isinstance(layer, nn.Sequential):
                yield from self.iterate_layers(layer)
            else:
                yield layer

    def build_graph(self, weights = "basic"):
        for layer in self.iterate_layers(self.model):
            print(f"On layer {layer}")
            if isinstance(layer, nn.Conv2d):
                if not self.added_input_layer:
                    in_channels = layer.in_channels
                    self.prev_nodes = self.add_layer_nodes(in_channels, role='input')
                    self.added_input_layer = True

                out_channels = layer.out_channels
                curr_nodes = self.add_layer_nodes(out_channels, role='hidden')

                # if weights == "pearson_correlation" and self.prev_conv_layer != None:
                #     collected = []
                #     print(f"shape 1 {self.activation_dict[self.prev_conv_layer].shape}")
                #     # print(f"shape 2 {self.activation_dict[layer].shape}")
                #     collected.append(self.activation_dict[self.prev_conv_layer])
                #     # collected.append(self.activation_dict[layer])

                #     A = torch.cat(collected, dim=0)  # shape: [N, C, H, W]
                #     N, C, H, W = A.shape

                #     # Reshape to [N * H * W, C]
                #     A_flat = A.permute(0, 2, 3, 1).reshape(-1, C)

                #     print(A_flat.shape)
                #     print(prev_nodes)
                #     print(curr_nodes)
                #     self.add_edges(prev_nodes, curr_nodes)

                # else:
                # print(f"From {prev_nodes}. To {curr_nodes}")
                self.add_edges(self.prev_nodes, curr_nodes)
                self.prev_nodes = curr_nodes
                self.prev_conv_layer = layer

            elif isinstance(layer, nn.Linear):
                if not self.added_input_layer:
                    # Use in_features as a proxy for input layer
                    in_features = layer.in_features
                    self.prev_nodes = self.add_layer_nodes(in_features, role='input')
                    self.added_input_layer = True

                out_features = layer.out_features
                weight_matrix = layer.weight.detach().numpy()
                if weights == "basic" or self.prev_lin_layer == None:
                    weight_matrix = layer.weight.detach().numpy()

                elif weights == "pearson_correlation":
                    prev_activation = self.activation_dict[self.prev_lin_layer].numpy()
                    curr_activation = self.activation_dict[layer].numpy()
                    # Assume A and B are numpy arrays of shape [N, D1] and [N, D2]
                    prev = prev_activation.shape[1]
                    curr = curr_activation.shape[1]

                    M = np.zeros((prev, curr))

                    for i in range(prev):
                        for j in range(curr):
                            M[i, j] = np.corrcoef(prev_activation[:, i], curr_activation[:, j])[0, 1]
                   
                    weight_matrix = M.T
  
                is_final = layer == list(self.iterate_layers(model))[-1]
                role = 'output' if is_final else 'hidden'
                curr_nodes = self.add_layer_nodes(out_features, role=role)

                self.add_edges(self.prev_nodes, curr_nodes, weights=weight_matrix)
                self.prev_nodes = curr_nodes
                self.prev_lin_layer = layer

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                # Skip these in graph layout, but keep neuron flow intact
                continue
            
            #NOTE: This is why graph isn't connected rn ??
            elif isinstance(layer, nn.MaxPool2d):
                # Represent pool as identity layer — one node per feature map
                if self.prev_nodes:
                    curr_nodes = self.add_layer_nodes(len(self.prev_nodes), role='hidden')
                    self.add_edges(self.prev_nodes, curr_nodes)
                    self.prev_nodes = curr_nodes

            #NOTE: Maybe this is also a problem
            elif isinstance(layer, nn.Flatten):
                # Represent flatten as passthrough
                continue

    def custom_layered_layout(self, v_spacing=1.5, h_spacing=3.0):
        '''
        Custom spacing function because visualizing large neural network models becomes unwieldy without being able to determine node positioning
        '''

        # Collect nodes by layer_index
        layers = {}
        for node, attrs in self.G.nodes(data=True):
            idx = attrs['layer_index']
            layers.setdefault(idx, []).append(node)

        pos = {}
        for layer_idx, nodes in layers.items():
            # Stack nodes vertically for each layer
            for i, node in enumerate(nodes):
                x = layer_idx * h_spacing
                y = -i * v_spacing  # negative for top-down
                pos[node] = (x, y)
        return pos

    #TODO: Make visualization cleaner for deeper neural networks, also maybe there's way to group neurons together when creating initial graph
    def draw_graph(self, pathname = "Graph", v_spacing=1.5, h_spacing=3.0, node_size_frac = 1):
        # Assign color based on role
        role_colors = {'input': 'lightgreen', 'hidden': 'lightblue', 'output': 'salmon'}
        node_colors = [role_colors[self.G.nodes[n]['role']] for n in self.G.nodes()]

        # Get layout based on layer index
        pos = self.custom_layered_layout(v_spacing=v_spacing, h_spacing=h_spacing)


        node_size_frac = node_size_frac

        # Compute edge widths from weights
        edge_weights = [abs(self.G[u][v]['weight']) for u, v in self.G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0

        edge_widths = [(1 + 4 * (w / max_weight))*node_size_frac for w in edge_weights]

        nx.draw(
            self.G, 
            pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=1000*node_size_frac, 
            width=edge_widths,
            font_size=6,  
            arrows=True,connectionstyle="arc3,rad=0.1"
        )

        # Optional: Label edges with weight values
        # edge_labels = { (u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges() }
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

        plt.savefig(pathname)

    
    def hook(self, module, input, output):
        self.activation_dict[module] = output.detach().cpu()
        # Come back and decide which one you really want
        # self.activation_mat.append(self.activation_dict[layer_name])

    #Tell the model to run the 'get actication hook' function, which just records the activation from the layer
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self.hook)

    #TODO: Add different correlation metrics, currently just using Pearson Correlation
    def compute_co_activation_matrix(self, dataset: Dataset):
        dl = DataLoader(dataset=dataset, batch_size= 64)

        self.register_hooks()
        # self.model.no_grad()
        for batch in dl:
            inputs, _ = batch
            self.model(inputs)


        


    def build_co_activation_graph(self, dataset: Dataset):

        coactivation_matrix = self.compute_co_activation_matrix(dataset)

        n_neurons = coactivation_matrix.shape[0]

        # Add nodes
        self.G.add_nodes_from(range(n_neurons))

        # probably wrong
        threshold = 0.5
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                weight = coactivation_matrix[i, j]
                if abs(weight) > threshold:
                    self.G.add_edge(i, j, weight=weight)



if __name__ == "__main__":
    mnist_loader = MnistDataloader('train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
                                    't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    train_dataset, test_dataset = mnist_loader.load_dataset()

    plt.figure(figsize=(18, 10))        
    model = LeNet5(reduction_factor=16)

    
    LeNetGraph = Neural_Net_Graph(model)
    LeNetGraph.compute_co_activation_matrix(test_dataset)
    print(LeNetGraph.activation_dict.keys())
    LeNetGraph.build_graph(weights="pearson_correlation")
    

    LeNetGraph.draw_graph(pathname = "co-activation.png", node_size_frac= .5)
    # draw_interactive_graph(LeNetGraph.G)

    # plt.clear()
    

