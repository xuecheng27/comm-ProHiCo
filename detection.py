def leiden(graph):
    return graph.community_leiden(objective_function='modularity')

def fast_greedy(graph):
    return graph.community_fastgreedy().as_clustering()

def label_propagation(graph):
    return graph.community_label_propagation()

def info_map(graph):
    return graph.community_infomap()

def walk_trap(graph, **kwargs):
    return graph.community_walktrap(**kwargs).as_clustering()