import matplotlib.pyplot as plt
import networkx as nx


def drawGraph(g,title="Graph"):
    plt.figure("Graph", figsize=(12, 12))
    plt.title(title)
    pos = nx.spring_layout(g)
    color_group = g.graph["colors"]
    classNames = g.graph["classNames"]
    node_color = []
    edge_color = []
    # print("CLASES:", classes)
    for node, label in g.nodes(data="label"):
        if(g.nodes[node]["typeNode"] == "net"):
            node_color.append(color_group[classNames.index(label)])
        if(g.nodes[node]["typeNode"] == "opt"):
            node_color.append("#000000")
    for node_a, node_b, color in g.edges.data("color", default="#9db4c0"):
        edge_color.append(color)

    # for node,typeNode in g.nodes(data='label'):
    #     typeNode=str(typeNode)
    #     if(typeNode=='0'):
    #         node_color.append('b')
    #     if(typeNode=='1'):
    #         node_color.append('r')
    #     if(typeNode=='2'):
    #         node_color.append('g')
    #     if(typeNode=='?'):
    #         node_color.append('black')
    # for index, (node,typeNode) in enumerate(g.nodes(data='typeNode')):
    #     if(typeNode=='test'):
    #         node_color[index]='black'
    # nx.draw(g,node_size=200)
    nx.draw(g, pos, node_color=node_color,
            edge_color=edge_color, width=1.3,node_size=150,with_labels=True)
    plt.show()
