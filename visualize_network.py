import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from network import LeNet5

class Neural_Net_Graph():
    def __init__(self, model):
        self.G = nx.DiGraph()
        self.layer_counter = 0
        self.layers = []
        self.model = model
        self.added_input_layer = False
        self.prev_nodes = None


    def add_edges(self, prev_nodes, curr_nodes, weights=None):
        for j, dst in enumerate(curr_nodes):
            for i, src in enumerate(prev_nodes):
                if weights is not None:
                    self.G.add_edge(src, dst, weight=weights[j][i])
                else:
                    self.G.add_edge(src, dst, weight=1.0)

    def add_layer_nodes(self, num_nodes, role):
        nodes = [f"L{self.layer_counter}_N{i}" for i in range(num_nodes)]
        for n in nodes:
            self.G.add_node(n, layer=f"Layer {self.layer_counter}", role=role, layer_index=self.layer_counter)
        self.layers.append(nodes)
        self.layer_counter += 1
        return nodes

    # Flatten out nested Sequential modules
    def iterate_layers(self, m):
        for name, layer in m.named_children():
            if isinstance(layer, nn.Sequential):
                yield from self.iterate_layers(layer)
            else:
                yield layer


    def build_graph(self):
        for layer in self.iterate_layers(self.model):
            if isinstance(layer, nn.Conv2d):
                if not self.added_input_layer:
                    in_channels = layer.in_channels
                    prev_nodes = self.add_layer_nodes(in_channels, role='input')
                    self.added_input_layer = True

                out_channels = layer.out_channels
                curr_nodes = self.add_layer_nodes(out_channels, role='hidden')

                self.add_edges(prev_nodes, curr_nodes)
                self.prev_nodes = curr_nodes

            elif isinstance(layer, nn.Linear):
                if not self.added_input_layer:
                    # Use in_features as a proxy for input layer
                    in_features = layer.in_features
                    self.prev_nodes = self.add_layer_nodes(in_features, role='input')
                    self.added_input_layer = True

                out_features = layer.out_features
                weight_matrix = layer.weight.detach().numpy()
                is_final = layer == list(self.iterate_layers(model))[-1]
                role = 'output' if is_final else 'hidden'
                curr_nodes = self.add_layer_nodes(out_features, role=role)

                self.add_edges(self.prev_nodes, curr_nodes, weights=weight_matrix)
                self.prev_nodes = curr_nodes

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                # Skip these in graph layout, but keep neuron flow intact
                continue

            elif isinstance(layer, nn.MaxPool2d):
                # Represent pool as identity layer — one node per feature map
                if self.prev_nodes:
                    curr_nodes = self.add_layer_nodes(len(self.prev_nodes), role='hidden')
                    self.add_edges(self.prev_nodes, curr_nodes)
                    self.prev_nodes = curr_nodes

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

if __name__ == "__main__":
    plt.figure(figsize=(18, 10))         # Bigger canvas

    model = LeNet5()

    # model = nn.Sequential(
    #     nn.Linear(2, 4),
    #     nn.ReLU(),
    #     nn.Linear(4, 3),
    #     nn.ReLU(),
    #     nn.Linear(3, 1)
    # )
    
    LeNetGraph = Neural_Net_Graph(model)
    LeNetGraph.build_graph()
    LeNetGraph.draw_graph(pathname = "LeNet5(3).png")

    # plt.clear()
    

