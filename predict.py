import networkx as nx

def _nNeighbors(g,index,label,deep,result):
    if(deep==0):
        result.append(index)
        return
    index = str(index)
    neighbors = list(nx.neighbors(g, index))
    colors=g.graph["colors"]
    classNames=g.graph["classNames"]
    for node in neighbors:
        if(label!="?"):
            color=colors[classNames.index(label)]
            g.edges[str(index),str(node)]["color"]=color
        _nNeighbors(g,node,label,deep-1,result)

def nNeighbors(g,index,deep):
    result = []
    label=g.nodes()[str(index)]["label"]
    _nNeighbors(g,index,label,deep,result)
    result.append(str(index))
    result=list(set(result))
    return result
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
def prediction(g,index,deep=1):
    index=str(index)
    classNames = g.graph["classNames"]
    neighbors=nNeighbors(g,index,deep)
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