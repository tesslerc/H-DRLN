# Utilities
__This is in development stage__

- activations_hdf5_to_tsne.lua - use this in order to extract 2d tSNE map of the activations. (also left some code to convert data structure from t7 to h5 if needed).
- clean_data.py - runs over data and keeps only trajectories that result in a successful result. We don't want to learn a skill of the agent stuck in the corner.
- cluster.py - will run multiple Gaussian Mixture Models in order to find the optimal # of clusters to fit the data.
- learn_weights.py - tensorflow, will learn the weights + save them. Weights are an extension of the "expert network".
