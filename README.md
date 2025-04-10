# Enhancing Graph Representations via Generalized Edge-to-Vertex Transforms and Complementary Information Bottleneck

## Introduction

Unsupervised graph representation learning has garnered considerable attention for its remarkable performance in modelling complex structures and high-dimensional features. However, current graph representation learning methods face practical limitations for the following reasons: Insufficient expressiveness of the single discriminant representation as well as redundancy in the information transfer process result in learned feature vectors that may deviate from high-quality representations. To address these limitations, we explore a novel generalized graph edge-to-vertex lossless transforms. Building upon this transforms process's line graphs, we propose the CONGIB network as a complementary graph information bottleneck solution. CONGIB encourages the compression of original and line graph information to facilitate the model's learning of more compact feature vectors. Moreover, it leverages line graph features to supplement the abstract spatial transfer states of the original graph, thus enabling richer information mapping. Extensive experiments on 13 public datasets demonstrate that our approach surpasses the latest state-of-the-art methods across various downstream tasks.

### Experiments
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora
```
