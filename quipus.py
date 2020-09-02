import networkBuilding as nBuilding
import numpy as np
import networkx as nx
import predict as predict
import tools
import pyswarms as ps


class HLNB_BC:
    knn = None
    X_train = []
    Y_train = []

    def predict(self, X_test, Y_test=[]):
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test,1)
        result = []
        # (self.X_train, X_test) = norm.preprocess(self.X_train, X_test)
        g, nbrs = nBuilding.networkBuildKnn(
            self.X_train, self.Y_train, self.knn, self.ePercentile, labels=True
        )
        # nBuilding.getProperty(g)
        # draw.drawGraph(g,title="Graph Iris Dataset k="+str(self.knn)+" e="+str(self.ePercentile)+ " b=10 α=0.0" )
        # draw.drawGraph(g,title="" )
        results = []
        for index, instance in enumerate(X_test):
            # CHECK INDEX LNNET WAS REMOVED + index
            indexNode = g.graph["lnNet"]
            if len(Y_test) == 0:
                nBuilding.insertNode(g, nbrs, instance)
            else:
                nBuilding.insertNode(g, nbrs, instance, Y_test[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.prediction(g, self.bnn, self.alpha)
            results.append(tmpResults)
            maxIndex = np.argmax(tmpResults)
            newLabel = g.graph["classNames"][maxIndex]
            result.append(newLabel)
            # g.remove_node(str(indexNode))
            g.nodes[str(indexNode)]["label"] = newLabel
            nn = list(nx.neighbors(g, str(indexNode)))
            for node in nn:
                if g.nodes[str(node)]["label"] != newLabel:
                    g.remove_edge(str(node), str(indexNode))
            # draw.drawGraph(g,"Final Node")
            for edge in g.edges:
                g.edges[edge]["color"] = "#9db4c0"
            g.graph["index"] += 1
        # draw.drawGraph(g,title="")

        if len(Y_test) != 0:
            # print("RESULT:", np.array(result))
            # print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(g.graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append([element, Y_test[index], results[index]])
            acc /= len(X_test)

            # print("ERRORS: ", err)
            print("Accuracy ", round(acc, 2), "%")
        return result

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def __init__(self, knn=3, ePercentile=0.5, bnn=3, alpha=1.0):
        self.knn = knn
        self.bnn = bnn
        self.alpha = alpha
        self.ePercentile = ePercentile

    def get_params(self, deep=False):
        return {
            "knn": self.knn,
            "ePercentile": self.ePercentile,
            "bnn": self.bnn,
            "alpha": self.alpha,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X_test, Y_test):
        self.predict(X_test, Y_test)


class Quipus(HLNB_BC):
    graphs = []

    def predict(self, X_test, Y_test=[]):
        # self.graphs = nBuilding.quipusBuildKnn(
        #     self.X_train, self.Y_train, self.knn, self.ePercentile, labels=True
        # )
        result = []
        results = []
        self.classes = self.graphs[0].graph["classNames"]
        for index, instance in enumerate(X_test):
            # indexNode = self.graphs[0].graph["lnNet"] + index
            if len(Y_test) == 0:
                nBuilding.quipusInsert(self.graphs, instance)
            else:
                nBuilding.quipusInsert(self.graphs, instance, Y_test[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.quipusPrediction(self.graphs, self.bnn, self.alpha)
            test = tmpResults * self.w
            test = np.transpose(test)
            test = [np.sum(e) for e in test]
            test2 = test / np.sum(test)
            results.append(test2)
            result.append(list(set(self.Y_train))[np.argmax(test2)])

        if len(Y_test) != 0:
            print("RESULT:", np.array(result))
            print("Y_TEST:", np.array(Y_test))
            acc = 0
            err = []
            err.append(self.graphs[0].graph["classNames"])
            for index, element in enumerate(result):
                if element == Y_test[index]:
                    acc += 1
                else:
                    err.append([element, Y_test[index], results[index][self.classes.index(element)], results[index][self.classes.index(Y_test[index])]])
            acc /= len(X_test)

            print("ERRORS: ", err)
            print("Accuracy ", round(acc, 4), "%")
        # print(test2)
        # print(tmpResults)
        return result
        # tmpResults = predict.prediction(g, indexNode, self.bnn, self.deepNeighbors)
        # results.append(tmpResults)
        # maxIndex = np.argmax(tmpResults)
        # newLabel = g.graph["classNames"][maxIndex]
        # result.append(newLabel)
        # # g.remove_node(str(indexNode))
        # g.nodes[str(indexNode)]["label"] = newLabel
        # nn = list(nx.neighbors(g, str(indexNode)))
        # for node in nn:
        #     if g.nodes[str(node)]["label"] != newLabel:
        #         g.remove_edge(str(node), str(indexNode))
        # # draw.drawGraph(g,"Final Node")
        # for edge in g.edges:
        #     g.edges[edge]["color"] = "#9db4c0"
        # g.graph["index"] += 1
        # draw.drawGraph(g,title="")

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.dim = (len(self.X_train[0]), len(list(set(self.Y_train))))
        self.w = np.ones(self.dim[0] * self.dim[1])
        G = nBuilding.quipusBuildKnn(
            X_train, Y_train, self.knn, self.ePercentile, labels=True
        )
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        probabilities = []

        for index, instance in enumerate(X_train):
            # indexNode = self.graphs[0].graph["lnNet"] + index
            if len(Y_train) == 0:
                nBuilding.quipusInsert(G, instance)
            else:
                nBuilding.quipusInsert(G, instance, Y_train[index])
            # draw.drawGraph(g,"New Dark Node Inserted")
            tmpResults = predict.quipusPrediction(G, self.bnn, self.alpha)
            probabilities.append(tmpResults)
            # test = tmpResults * self.w
            # test = np.transpose(test)
            # test = [np.sum(e) for e in test]
            # test2 = test / np.sum(test)
            # print(test2)
            # print(tmpResults)
        # print(probabilities)
        max_bound = np.ones(self.dim[0] * self.dim[1])*self.dim[1]
        # min_bound = np.zeros(self.dim[0] * self.dim[1])
        min_bound = - max_bound
        bounds = (min_bound, max_bound)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=50,
            dimensions=self.dim[0] * self.dim[1],
            options=options,
            bounds=bounds,
        )
        cost, pos = optimizer.optimize(
            optimizacion,
            iters=20,
            probabilidades=probabilities,
            Y=Y_train,
            dim=self.dim,
        )
        self.w = np.reshape(pos, (self.dim))
        self.graphs = G
        # max_bound = [0,0,0,0,0]
        # min_bound = [0,0,0,0,0]
        # bounds = [min_bound,max_bound]
        # max_bound = np.ones(len(self.X_train[0]) * len(list(set(self.Y_train)))).tolist()
        # min_bound = np.zeros(len(self.X_train[0]) * len(list(set(self.Y_train))).tolist()
        # bounds = [min_bound,max_bound]
        # optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
        # return


def optimizacion(particles, probabilidades, Y, dim):
    accs = []
    classes = list(set(Y))
    for w_particle in particles:
        w = np.reshape(w_particle, dim)
        acc = 0
        Y_predicted = []
        for index, instanceProb in enumerate(probabilidades):
            test = instanceProb * w
            test = np.transpose(test)
            test = [np.sum(e) for e in test]
            Y_predicted.append(classes[np.argmax(test)])
        acc = sum(1 for x, y in zip(Y, Y_predicted) if x == y) / len(Y)
        accs.append(1.0 - acc)
    return accs
