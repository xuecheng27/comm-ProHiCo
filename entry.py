from deception.sbm import ProHiCoSBM
from deception.dcsbm import ProHiCoDCSBM
from igraph import Graph
from detection import *

INPUT_SETTINGS = {
    'path': 'samples/blogs.gml',
    'alpha': 1000,
    'beta': 0,
    'detection_func': leiden,
    'func_args': {},
    'interval': 50
}

if __name__ == "__main__":
    g = Graph.Read_GML(INPUT_SETTINGS['path'])
    true_partitions = INPUT_SETTINGS['detection_func'](g, **INPUT_SETTINGS['func_args'])

    example1 = ProHiCoSBM(graph=Graph.Read_GML(INPUT_SETTINGS['path']), true_partitions=true_partitions, target_partitions_index=list(range(len(true_partitions))), **INPUT_SETTINGS)
    example1.run()

    example2 = ProHiCoDCSBM(graph=Graph.Read_GML(INPUT_SETTINGS['path']), true_partitions=true_partitions, target_partitions_index=list(range(len(true_partitions))), **INPUT_SETTINGS)
    example2.run()