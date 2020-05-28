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
def getDataCSV(url,className):
    dataset = load_iris()
    data = pd.read_csv(url, keep_default_na=False,  na_values=np.nan)
    if(len(data.values[0])==1):
        data = pd.read_csv(url,";", keep_default_na=False,  na_values=np.nan)
    dataset['target']=data[className].values
    dataset['data']=data.drop(className,axis=1).values
    return dataset
#%%
dataset = load_iris()
X_train, X_predict, Y_train, Y_predict= train_test_split(dataset['data'],dataset["target"],test_size=0.7)
knn=5
normalization=2
link=0
(X_train,X_predict)=norm.preprocess(X_train,X_predict,normalization)

#X_net, X_opt, Y_net, Y_opt= train_test_split(X_train,Y_train,test_size=0.2)

#%%

def networkBuildKnn(X_net,Y_net,knn,labels=False):
    g=nx.Graph()
    lnNet=len(X_net)
    g.graph["lnNet"]=lnNet
    g.graph["classNames"]=list(set(Y_net))
    for index,instance in enumerate(X_net):
        g.add_node(str(index), value=instance ,typeNode='net',label=Y_net[index])
    values=X_net
    
    if(isinstance(values[0],(int,float,str))):
        values=[e[0] for e in values]
        
    nbrs= NearestNeighbors(knn+1,metric='euclidean')
    nbrs.fit(values)

    distances,indices = nbrs.kneighbors(values)
    indices=indices[:, 1:]
    distances=distances[:, 1:]
    eRadius=np.quantile(distances,0.5)
    nbrs.set_params(radius=eRadius)
    
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if( g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"] or not labels):
                g.add_edge(str(indice),str(indiceNode),weight=distances[indiceNode][tmpi])
    
    distances,indices = nbrs.radius_neighbors([instance])
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if(not str(indice)==str(indiceNode)):
                if( g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"] or not labels):
                    g.add_edge(str(indice),str(indiceNode),weight=distances[indiceNode][tmpi])
    g.graph["index"]=lnNet
    return g,nbrs
def drawGraph(g):
    plt.figure("Graph",figsize=(12,12))
    pos=nx.spring_layout(g)
    node_color=[]
    for node,typeNode in g.nodes(data='label'):
        typeNode=str(typeNode)
        if(typeNode=='0'):
            node_color.append('b')
        if(typeNode=='1'):
            node_color.append('r')
        if(typeNode=='2'):
            node_color.append('g')
        if(typeNode=='?'):
            node_color.append('black')
    for index, (node,typeNode) in enumerate(g.nodes(data='typeNode')):
        if(typeNode=='test'):
            node_color[index]='black'
    nx.draw(g,pos,node_color=node_color, node_size=200)
    plt.show()
#%%
g,nbrs=networkBuildKnn(X_train, Y_train, knn,labels=True)
drawGraph(g)
print(g.graph)
print(len(list(g.nodes())))
print(len(list(g.edges())))
#%%
def nodeInsertion(g,nbrs,instance,nodeIndex,label):
    g.add_node(str(nodeIndex), value=instance ,typeNode='test',label=label)
    if(isinstance(instance,(int,float,str))):
        instance=[instance]
    distances,indices = nbrs.kneighbors([instance])
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            g.add_edge(str(indice),str(nodeIndex),weight=distances[indiceNode][tmpi])
    distances,indices = nbrs.radius_neighbors([instance])
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if(not str(indice)==str(indiceNode)):
                g.add_edge(str(indice),str(nodeIndex),weight=distances[indiceNode][tmpi])
def linkProb(g,nodei:str,nodej:str):
    ni_nj=list(nx.common_neighbors(g,str(nodei),str(nodej)))
    prob=[len(list(g.neighbors(i))) for i in ni_nj]
    total=0
    for i in prob:
        total+=1/math.log(i)
    return total
    # return len(ni_nj)
def linkProbNode(g,nodei,labels=True,remove=False):
    classNames=g.graph["classNames"]
#     print("Node: ",nodei)
    ni=list(g.neighbors(str(nodei)))
    nn=[]
#     print("NEIGhBORS: ",ni)
    for neighbor in ni:
#         print("Neighbor: ",neighbor)
        auxNN=list(g.neighbors(str(neighbor)))
#         print("Neighbors: ",auxNN)
        nn.extend(auxNN)
#         print("Operation->",nn)
#     print("NNeighbors: ",nn)
    nn=list(set(nn))
#     print("NNeighbors: ",nn)
    nn.remove(str(nodei))
    for i in ni:
        if(str(i) in nn):
            nn.remove(str(i))
    classConnections=[0]*len(classNames)

    for e in nn:
        force=linkProb(g, e, nodei)
        aux=g.nodes()[str(e)]["label"]
        classConnections[aux]+=force
        
    classConnection=classNames[ np.argmax(classConnections)]
    if(remove):
        g.remove_node(str(nodei))
        return classConnection
    g.nodes()[str(nodei)]['label']=classConnection
    if(labels):
        for neighbor in ni:
            if(not g.nodes()[str(neighbor)]['label']==classConnection):
                g.remove_edge(str(neighbor),str(nodei))
    return classConnection

def manyInserts(g,nbrs,X_predict,Y_predict=[]):
    for nodeIndex, instance in enumerate(X_predict):
        label='?'
        if(not len(Y_predict)==0):
            label=Y_predict[nodeIndex]
        nodeInsertion(g, nbrs, instance, nodeIndex+g.graph['index'], label)


def predict(g, nbrs, X_predict,Y_predict=[]):
    predictions=[]
    for nodeIndex, instance in enumerate(X_predict):
        label='?'
        if(not len(Y_predict)==0):
            label=Y_predict[nodeIndex]
        nodeInsertion(g, nbrs, instance, nodeIndex+g.graph['index'], label)
        tmp=linkProbNode(g, nodeIndex+g.graph['index'],False)
        predictions.append(tmp) 
    return np.array(predictions)
results=predict(g, nbrs, X_predict,Y_predict)
print(results)
print(Y_predict)
print(np.mean(results==Y_predict))
drawGraph(g)
print(g.graph)
print(len(list(g.nodes())))
print(len(list(g.edges())))

                
                
                
                
                
                
                
