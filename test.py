#%%
import numpy as np
import normalization as norm
import networkx as nx
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris,load_wine
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold,ShuffleSplit,StratifiedKFold
import pandas as pd
#%%
g=nx.Graph()
g.add_node("a")
g.add_node("b")
g.add_node("c")
g.add_node("d")
g.add_node("e")
g.add_node("f")
g.add_edge("a","b")
g.add_edge("a","c")
g.add_edge("c","d")
g.add_edge("b","e")
g.add_edge("c","f")
g.add_edge("f","d")

nx.draw(g,with_labels=True)

# %%
nneighbors=nx.single_source_shortest_path_length(g, "a", cutoff=1)
list(nneighbors)

# %%
