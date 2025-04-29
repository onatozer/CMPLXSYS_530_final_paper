# About
The following contains the code I used in writing this [paper](Visualizing_Neural_Networks.pdf)

## Installation

To run the code, make sure you've ran

```bash
pip install -r requirements.txt
```

## Usage
To reproduce all the plots I made, simply run. This will use the network model weights that I previously saved. 

```bash
python3 generate_graphs.py
```

If you want to train your own network run:

```bash
python3 train_network.py
```

And then adjust the model paths in generate_graphs accordingly if you want to visualize that network instead
