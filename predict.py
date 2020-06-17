import numpy as np
import math
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import drawGraph as draw
def _nNeighbors(g,index,label,deep,result):
    if(deep==0):
        if(not index in result):
            result.append(index)
        return
    index = str(index)
    colors=g.graph["colors"]
    classNames=g.graph["classNames"]
    neighbors = list(nx.neighbors(g, index))
    for n in neighbors:
        result.append(n)
    result =list(set(result))
    for node in neighbors:
        if(label!="?"):
            color=colors[classNames.index(label)]
            g.edges[str(index),str(node)]["color"]=color
        _nNeighbors(g,node,label,deep-1,result)

def nNeighbors(g,index,deep):
    result = []
    label=g.nodes[str(index)]["label"]
    _nNeighbors(g,index,label,deep,result)
    if(str(index) in result):
        result.remove(str(index))
    result=list(set(result))
    return result
def firstNeighbors(g,index):
    result = nx.neighbors(g,index)
    return list(set(result))
def subGraph(g,nodes):
    g1=g.copy()
    for node in g.nodes():
        if(not node in nodes):
            g1.remove_node(node)
    return g1
def classSubGraph(subG,funcEvaluation):
    result=[]
    classNames = subG.graph["classNames"]
    graphs=[]
    for className in classNames:
        classG=subG.copy()
        for indexNode in subG.nodes():
            node=classG.nodes()[indexNode]
            if(node["typeNode"]=="net"):
                if(node["label"]!=className):
                    classG.remove_node(indexNode)
        tmpBeetweenness=funcEvaluation(classG)
        # print("CLASS ",className, "Evaluation : ", tmpBeetweenness)
        result.append(tmpBeetweenness)
        classG.graph["class"]=className
        graphs.append(classG)
    return result, graphs
def predictionBetweetness(g,index,deep=1):
    index=str(index)
    classNames = g.graph["classNames"]
    neighbors=nNeighbors(g,index,deep)
    # neighbors=firstNeighbors(g,index)
    neighbors.append(str(index))
    subG=subGraph(g,neighbors)
    evaluationResults, classSubGraphs=classSubGraph(subG,nx.betweenness_centrality)
    
    result=[]
    for element in evaluationResults:
        result.append(element[index])
    if sum(result)==0 :
        result=[]
        for element in evaluationResults:
            result.append(len(element))
    return result
def connected(g):
    if(nx.is_empty(g) or nx.is_connected(g)):
        return g
    else:
        largest_cc = max(nx.connected_components(g), key=len)
        subG = g.subgraph(largest_cc)
        return subG 
def prediction(g,index,deep=1):
    index = str(index)
    classNames = g.graph["classNames"]
    currentRWB=[]
    insertionRWB=[]
    result=[]
    nlinks=[]
    for indexClassName, _ in enumerate(classNames):
        classNodes = g.graph["classNodes"][indexClassName]
        classNodes.append(index)
        subG = g.subgraph(classNodes)
        neighbors = list(nx.single_source_shortest_path_length(subG, index, cutoff=deep))

        # neighbors.remove(index)
        # subG = g.subgraph(neighbors)


        # # rwbListB={}
        # # if(not nx.is_empty(subG) and nx.is_connected(subG) and len(neighbors)>2):
        # #     rwbListB=nx.current_flow_betweenness_centrality(subG)

        # neighbors.append(index)
        
        subG = g.subgraph(neighbors)
        rwbListA={}
        nlinks.append(len(neighbors)-1)
        if(len(neighbors)>3):
            rwbListA=nx.current_flow_betweenness_centrality(subG)
        if(len(rwbListA)<=3):
            result.append(1)
        else:
            currentRWB=rwbListA[index]
            tmp=0
            for key in rwbListA:
                tmp=tmp+abs(rwbListA[key]-currentRWB)
            tmp/=len(rwbListA)
            if(tmp==0):
                tmp=1.
            result.append(tmp)

    # result= abs(np.subtract(currentRWB,insertionRWB))
    resultT=result
    tnlinks=nlinks
    result = 1/np.array(result)
    result = np.array(result)/sum(result)
    nlinks =np.array(nlinks)
    nlinks = (nlinks/sum(nlinks))
    # nlinks = 1 - nlinks
    resultFinal = (result*0.6+nlinks*0.4)
    resultFinal = resultFinal/sum(resultFinal)
    
    indexMin=np.argmax(resultFinal)
    tmpLabel=g.nodes[index]["label"]
    classifyLabel=classNames[indexMin]
    if(not tmpLabel=='?' and tmpLabel!=classifyLabel):
        neighbors = list(nx.single_source_shortest_path_length(g, index, cutoff=deep))
        neighbors.append(index)
        subG = g.subgraph(neighbors)
        draw.drawGraph(subG,"Pre insert class predicted:"+str(classNames[indexMin])+" "+str(resultFinal)+" REAL: "+str(g.nodes[index]["label"]))
    return resultFinal