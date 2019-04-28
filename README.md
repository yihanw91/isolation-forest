# Isolation Forest from Scratch

This is my implementation of Isolation Forest from scratch in Python. Isolation Forest is an unsupervised machine learning algorithm for anomaly detection created by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. More details can be found in their paper Isolation-based Anomaly Detection: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.673.5779&rep=rep1&type=pdf

At a high level, Isolation Forest is based on the idea that anomalies (unusual points) can be isolated more quickly than normal points via random partitioning. In the context of a recursive tree, this means anomalies are more likely to be found in a leaf of lower depth that is closer to the root of the tree.
