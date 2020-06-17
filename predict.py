import numpy as np
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
    lens=[]
    for className in classNames:
        classNodes=[ e for e in  g.nodes if g.nodes[e]["label"]==className and e!= index]
        classNodes.append(index)
        classG = g.subgraph(classNodes)

        # neighbors=nNeighbors(classG,index,deep)
        neighbors=firstNeighbors(classG,index)
        # rootNN=np.array(neighbors)
        subG = g.subgraph(neighbors)
        # draw.drawGraph(subG,"Pre insert class:"+str(className))
        subG=connected(subG)
        if(len(subG.nodes)>2 and len(subG.edges)>1):
        # if(len(neighbors)!=0 and len(neighbors)!=1):
            rwbList=nx.current_flow_betweenness_centrality(subG)
            tmp=0
            for element in rwbList:
                if np.isnan(rwbList[element]):
                    continue
                if rwbList[element] :
                    tmp+= rwbList[element]
            currentRWB.append(tmp/len(neighbors))
        else:
            currentRWB.append(0)
        # tmpN=list(nx.neighbors(classG,index))
        classTMPG = g.subgraph(classNodes)
        neighbors.append(index)
        lens.append(len(neighbors))
        subG = g.subgraph(neighbors)
        if(len(neighbors)>2):
            rwbList=nx.current_flow_betweenness_centrality(subG)
            tmp=0
            for element in rwbList:
                if np.isnan(rwbList[element]):
                    tmp+=1
                else:
                    tmp+= rwbList[element]
            insertionRWB.append(tmp/len(neighbors))
        else:
            insertionRWB.append(1)
    result= abs(np.subtract(currentRWB,insertionRWB))
    lensNorm= np.array(lens)/np.sum(lens)
    resultFinal = result / lensNorm

    # indexMin=np.argmin(resultFinal)
    # tmpLabel=g.nodes[index]["label"]
    # correctLabel=classNames[indexMin]
    # if(tmpLabel!=correctLabel):
    #     neighbors=nNeighbors(g,index,deep)
    #     neighbors.append(index)
    #     subG = g.subgraph(neighbors)
    #     draw.drawGraph(subG,"Pre insert class predicted:"+str(classNames[indexMin])+" REAL: "+str(g.nodes[index]["label"]))
    return resultFinal