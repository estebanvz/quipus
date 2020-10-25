import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx.algorithms.community as nx_comm
import random
import tools
import math

# class chaski:
#     def __init__(self,g):
#         self.g = g;
class Chaski:
    path = []
    deep = 0
    subgraph = None

    def __init__(self, start):
        self.path = []
        self.path.append(start)

    def startRun(self, g, deep):
        current = self.path[0]
        for i in range(deep):
            tmp = getNext(g, current)
            self.path.append(tmp)
            current = tmp
        self.subgraph = g.subgraph(self.path)
        self.getMeasures()

    def getMeasures(self):
        nn = self.subgraph.number_of_nodes()
        if nn > 1:
            self.assortativity = nx.degree_assortativity_coefficient(self.subgraph)
        else:
            self.assortativity = 0
        # self.triangles = nx.triangles(self.subgraph)
        if math.isnan(self.assortativity):
            self.assortativity = 0
        self.transitivity = nx.transitivity(self.subgraph)
        # self.richClub= nx.rich_club_coefficient(self.subgraph)
        return (self.assortativity, self.transitivity)


class ChaskiList:
    chaskis = []
    className = None

    def __init__(self, className):
        self.className = className
        self.chaskis = []

    def addChaski(self, chaski):
        self.chaskis.append(chaski)

    def draw(self):
        tools.drawGraphs(
            [e.subgraph for e in self.chaskis], sizeGraph=(4, 4), labels=True
        )

    def getMeasures(self):
        result = []
        for e in self.chaskis:
            result.append(e.getMeasures())
        return result


def getNext(g, index):
    n = nx.neighbors(g, str(index))
    l = list(n)
    if len(l) == 0:
        return str(index)
    else:
        return random.choice(l)


def graphParticle(g):
    deep = 5
    n_chaskis = 10
    classChaskis = []
    for index, nodes in enumerate(g.graph["classNodes"]):
        chaskis = ChaskiList(g.graph["classNames"][index])
        for _ in range(n_chaskis):
            r = random.choice(nodes)
            c = Chaski(r)
            c.startRun(g.subgraph(nodes), deep)
            chaskis.addChaski(c)
        classChaskis.append(chaskis)
    return classChaskis


def getInsertedChaskiPath(g):
    deep = 5
    n_chaskis = 10
    classChaskis = []
    for index, nodes in enumerate(g.graph["classNodes"]):
        chaskis = ChaskiList(g.graph["classNames"][index])
        for _ in range(n_chaskis):
            r = g.graph["index"] - 1
            c = Chaski(str(r))
            c.startRun(g.subgraph(nodes), deep)
            chaskis.addChaski(c)
        classChaskis.append(chaskis)
    return classChaskis
