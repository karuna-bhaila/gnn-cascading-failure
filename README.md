# GNNs for cascading failure analysis
This repository contains code and datasets for the paper titled "Cascading Failure Prediction in Power Grid Using Node and Edge Attributed Graph Neural Networks".

## Data
The repo includes the pre-processed datasets in pytorch-geometric `Data` format only for the 39-bus test system. Since the files are too large, the pre-processed 118-bus scenarios and the raw scenario data for 39-bus and 118-bus in MATLAB object formats can be obtained from http://tinyurl.com/casacdedata.

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
