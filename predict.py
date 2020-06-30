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
def connected(g,index):
    if(nx.is_empty(g) or nx.is_connected(g)):
        return g
    else:
        # largest_cc = max(nx.connected_components(g), key=len)
        for component in nx.connected_components(g):
            if index in component:
                largest_cc=component
        subG = g.subgraph(largest_cc)
        return subG 
def prediction(g,index,deep=1, b=8):
    index = str(index)
    classNames = g.graph["classNames"]
    result=[]
    nlinks=[]
    for indexClassName, _ in enumerate(classNames):
        classNodes =[e for e in g.graph["classNodes"][indexClassName]]
        classNodes.append(index)
        subG = g.subgraph(classNodes)
        # neighbors = list(nx.single_source_shortest_path_length(subG, index, cutoff=deep))

        # neighbors.remove(index)
        # subG = g.subgraph(neighbors)


        # # rwbListB={}
        # # if(not nx.is_empty(subG) and nx.is_connected(subG) and len(neighbors)>2):
        # #     rwbListB=nx.current_flow_betweenness_centrality(subG)

        # neighbors.append(index)
        
        # subG = g.subgraph(neighbors)
        lenNN=len(list(subG.neighbors(index)))
        nlinks.append(lenNN)
        subG=connected(subG,index)
        rwbListA={}
        if(len(classNodes)>3 and lenNN>3):
            # rwbListA=nx.betweenness_centrality(subG,k=int(len(g.nodes())*0.2))
            if(not nx.is_connected(subG)):
                draw.drawGraph(subG)
            # rwbListA=nx.current_flow_closeness_centrality(subG)
            rwbListA=nx.betweenness_centrality(subG,k=b)
        if(len(classNodes)<=3 or lenNN<=3):
            if(lenNN==0):
                result.append(1)
            else:
                result.append(None)
        else:
            currentRWB=rwbListA[index]
            # g.nodes()[key]['betweenness']=currentRWB
            tmp=[]
            for key in rwbListA:
                tmp.append(abs(rwbListA[key]-currentRWB))
            tmp.sort()
            tmp=tmp[:b]
            result.append(sum(tmp)/len(tmp))

    resultT=[e for e in result]
    tnlinks=nlinks
    for indexResult, e in enumerate(result):
        if(e==None):
            result[indexResult]=1.0

    nlinks =np.array(nlinks)
    nlinks = (nlinks/sum(nlinks))
    result = 1-((np.array(result)))
    result = np.array(result)/sum(result+1.0e-16)
    # nlinks = 1 - nlinks
    for indexResult,e in enumerate(resultT):
        if(e==None):
            result[indexResult]=nlinks[indexResult]

    # resultFinal = (result+nlinks)/2
    # resultFinal = resultFinal/sum(resultFinal)
    resultFinal = result

    # indexMin=np.argmax(resultFinal)
    # tmpLabel=g.nodes[index]["label"]
    # classifyLabel=classNames[indexMin]
    # if(not tmpLabel=='?' and tmpLabel!=classifyLabel):
    #     neighbors = list(nx.single_source_shortest_path_length(g, index, cutoff=deep))
    #     neighbors.append(index)
    #     subG = g.subgraph(neighbors)
    #     # draw.drawGraph(subG,"Pre insert class predicted:"+str(classNames[indexMin])+" "+str(np.round(resultFinal,4))+" REAL: "+str(g.nodes[index]["label"]))
    #     draw.drawGraph(subG,"Wrong Classification")
    return resultFinal