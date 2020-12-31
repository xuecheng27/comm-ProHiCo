from igraph.clustering import Clustering

def sub_clustering(vlist, cl):
    membership = cl.membership
    vmembership = [0] * len(vlist)
    for idx, v in enumerate(vlist):
        vmembership[idx] = membership[v]

    return Clustering(vmembership)

def count_jaccard_recall_and_precision(cl1, cl2):
    # Requirement: r1 > 0 and r2 > 0 equivalently, cl1 and cl2 are not singleton partitions
    r1 = __count_combinations(cl1)
    r2 = __count_combinations(cl2)
    r12 = __count_common_parts(cl1, cl2)

    recall = r12 / r1
    precision = r12 / r2
    jaccard = r12 / (r1 + r2 - r12)

    return jaccard, recall, precision

def __count_common_parts(cl1, cl2):
    # return the number of intra-edges in the Clustering Intersection(cl1, cl2)
    # Requirement: cl1 and cl2 have the same number of elements
    result = 0
    
    for cluster1 in cl1:
        sub_cl1 = sub_clustering(cluster1, cl2)
        sizes = sub_cl1.sizes()
        for sz in sizes:
            if sz > 0:
                result += sz * (sz - 1) / 2

    return result

def __count_combinations(cl):
    # return the number of intra-edges in the Clustering cl
    result = 0
    sizes = cl.sizes()

    for sz in sizes:
        if sz > 0:
            result += sz * (sz - 1) / 2
    
    return result

