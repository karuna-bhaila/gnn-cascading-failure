# GNNs for cascading failure analysis
This repository contains code and datasets for the paper titled "Cascading Failure Prediction in Power Grid Using Node and Edge Attributed Graph Neural Networks".

## Data
The simulated datasets include 5000 scenarios each for the IEEE 39-bus and 118-bus test systems. The datasets can be downloaded from: https://drive.google.com/file/d/1B3DoAPBqHVq6tr4_KhnFF-OsLFHnCLaI/view?usp=drive_link. The data was converted from MATLAB object format to PyG dataset format and includes bus features, branch features, node adjacency, node-edge incidence, node labels (binary), and edge labels (binary). Please refer to the paper for an explanation of node and edge features.

## Run
The main code to train/test is in `main.py`.
Use the following arguments to specify hyperparameters and settings.

| Argument      | Description             |
|---------------|-------------------------|
| dataset		    | Name of dataset (case39, case118)|
| model			    | GNN architecture (gcn, sage, gat, gate, gin, nea_gnn) |
| num_layers    | Number of GNN convolution layers |
| hidden        | Embedding size in GNN |
| lr            | Learning rate |
| dropout       | Dropout rate |
| epochs        | No. of training epochs |
| beta          | Weight for node label loss component (set to 1) |
| class_weight  | Weight for majority class (1) (set to 0.5 for case118) |
