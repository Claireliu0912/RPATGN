# RPATGN

The implementation for "Regulating Related Party Transactions with Role Perceptual Augmented Temporal Graph Network".

## Dataset

We conducted experiments on four real-world financial datasets:

- **RPT:** The dataset was collected from regularly disclosed RPT data in the Chinese financial market. It includes profiles of listed companies and related companies, RPT operation information, and financial exchanges involved in these transactions. In the graph,nodes represent listed companies or related parties, and an edge can be considered as an indication of a related party transaction between two nodes. It is a private dataset, and we are currently working closely with our partners on data anonymization and open sourcing, aiming to contribute this valuable industry-level dataset to the academic community and promote research on RPT. 
- **Elliptic:** The dataset maps Bitcoin transactions to real entities belonging to licit categories versus illicit ones. A node in the graph represents a transaction, an edge can be viewed as a flow of Bitcoins between one transaction and the other. 
- **Bitcoin OTC:** It is who-trusts-whom network of people who trade using Bitcoin on a platform called Bitcoin OTC. Nodes represent Bitcoin users, edges represent ratings between users, and it can be used to predict whether a user will rate another user in the next time step. 
- **Bitcoin Alpha:** The dataset was created in the same way as the Bitcoin OTC dataset, except that the users and ratings come from a different trading platform. 

## Baselines

We compare the performance of our proposed method with the state-of-the-art baselines:

- **GAE: ** Encodes node features using a GCN and reconstructs the graph's adjacency matrix using an inner product decoder.
- **VGAE:** An extension of GAE that learns the latent representation of nodes through variational inference, better handling the uncertainty in graph data.  
- **DySAT:** The model employs attention mechanisms along spatial and temporal dimensions, using GAT and transformers respectively.
- **EvolveGCN:** EvolveGCN adapts the GCN model over the time dimension by evolving GCN parameters through RNNs to capture the dynamics of graph sequences.
- **GRUGCN:** The model combines CNNs and RNNs for graph-structured data to simultaneously recognize spatial structures and dynamic patterns.
- **VGRNN:** It adds higher-order latent variables to the graph recurrent neural network to model the complex dynamics of the graph. 
- **HTGN:** A node representation learning framework based on hyperbolic geometry, mapping temporal graphs to hyperbolic space to learn temporal regularities, topological dependencies, and implicit hierarchical organization.
- **DGCN:** A dynamic graph representation learning model that combines GCN and LSTM to capture both global structure and temporal properties, using a novel Dice similarity measure for node aggregation.
- **HGWaveNet:** A hyperbolic graph neural network that leverage the fitness between hyperbolic spaces and data distributions for temporal link prediction.
- **RTGCN:** The model utilize global structural role information for temporal graph representation learning, combining structural role-based hypergraphs, GNN, and GRU modules.

## Usage

To test implementations of RPATGN, run:

```python
python main.py --dataset otc --nhid 32 --nout 32 --nb_window 5
python main.py --dataset alpha --nhid 32 --nout 32 --nb_window 5
```

### Parameter settings

We conducted the experiment five times and took the average results to avoid random errors. For datasets without node features, we use one-hot node-degree as the input feature. We set the embeddings dimension as 128 for RPT, Elliptic datasets and 64 for Bitcoin-OTC, Bitcoin-Alpha datasets. We uniformly set the historical snapshot recording window to 5. When determining node roles, we follow the "Pareto principle". We consider the top 20% of nodes by link count as the central roles, while the remaining nodes are considered peripheral. During training, we used early stopping with a patience of 50 and a maximum of 200 iterations to prevent overfitting and optimize performance. Our method is implemented using PyTorch 1.12.1 with CUDA 11.3 and Python 3.7 as the backend. The model is trained on a server with two 32GB NVIDIA Tesla V100 GPUs.

## Supplement Files Structure

- `data/`: dataset files. We provide processed data that can be used directly, including Bitcoin OTC and Bitcoin Alpha. 
- `scripts/`: implementations of models;

