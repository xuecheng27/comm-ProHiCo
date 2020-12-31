import logging.config
from typing import List
from settings import LOGGING_SETTINGS
import random
import metric

logging.config.dictConfig(LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class ProHiCoSBM(object):
    ''' The SBM algorithm in the ProHiCo framework to hide specified communities.

    Attributes:
        __graph: a Graph in which we hide specified communities
        __true_partitions: a Clustering object indicating the ground-truth communities
        __target_partitions_index: a list of integers specifying the targeted communities
        __alpha: an integer specifying the maximum number of allowed edge additions
        __beta: an integer specifying the maximum number of allowed edge removals
        __detection_func: a function indicating the community detection algorithm
        __func_args: a dictionary specifying the arguments of the __detection_func
        __interval: an integer specifying the interval of records
        __path: a string specifying the path of the graph
        __operations: a list of 0/1s indicating the order of edge additions and removals
    '''
    def __init__(self, graph, true_partitions, target_partitions_index, alpha, beta, detection_func, func_args, interval, path, operations=None, **kwargs):
        self.__graph = graph
        self.__true_partitions = true_partitions
        self.__target_partitions_index = set(target_partitions_index)
        self.__alpha = alpha
        self.__beta = beta
        self.__detection_func = detection_func
        self.__func_args = func_args
        self.__interval = interval
        self.__path = path
        if operations:
            self.__operations = operations
        else:
            self.__operations = [1] * self.__alpha + [0] * self.__beta

        self.__partitions_num = len(self.__true_partitions)
        self.__weight_matrix = [0] * (self.__partitions_num * self.__partitions_num)
        self.__weight_vector = [0] * self.__partitions_num
        self.__target_node_list: List = list()

        self.__count = 1

    def __preprocess(self):
        if len(self.__target_partitions_index) == self.__partitions_num:
            self.__target_node_list = list(range(self.__graph.vcount()))
        else:
            for i in range(self.__partitions_num):
                if i in self.__target_partitions_index:
                    self.__target_node_list.extend(self.__true_partitions[i])

        for i in range(self.__partitions_num):
            if i in self.__target_partitions_index:
                part_subgraph = self.__graph.subgraph(self.__true_partitions[i])
                part_ecount = part_subgraph.ecount()
                self.__weight_vector[i] = part_ecount

        for i in range(self.__partitions_num):
            for j in range(self.__partitions_num):
                if i == j:
                    continue
                elif (i not in self.__target_partitions_index) and (j not in self.__target_partitions_index):
                    continue
                else:
                    part_i_subgraph = self.__graph.subgraph(self.__true_partitions[i])
                    part_i_vcount = part_i_subgraph.vcount()
                    part_i_ecount = part_i_subgraph.ecount()

                    part_j_subgraph = self.__graph.subgraph(self.__true_partitions[j])
                    part_j_vcount = part_j_subgraph.vcount()
                    part_j_ecount = part_j_subgraph.ecount()

                    part_ij = list()
                    part_ij.extend(self.__true_partitions[i])
                    part_ij.extend(self.__true_partitions[j])
                    part_ij_subgraph = self.__graph.subgraph(part_ij)
                    part_ij_ecount = part_ij_subgraph.ecount()

                    num_nonedges = (part_i_vcount * part_j_vcount) - (part_ij_ecount - part_i_ecount - part_j_ecount)
                    self.__weight_matrix[i * self.__partitions_num + j] = num_nonedges


    def __start(self):
        logger.info('=' * 60)
        logger.info(f'Path: {self.__path}')
        logger.info(f'Vcount: {self.__graph.vcount()}')
        logger.info(f'Ecount: {self.__graph.ecount()}')
        logger.info(f'Parts: {self.__partitions_num}')
        logger.info(f'Targets: {self.__target_partitions_index}')
        logger.info(f'Alpha: {self.__alpha}')
        logger.info(f'Beta: {self.__beta}')
        logger.info(f'Detection: {self.__detection_func.__name__}')
        logger.info(f'Deception: prohico-sbm')
        logger.info(f'Interval: {self.__interval}')
        logger.info('=' * 60)

    def __quit(self):
        logger.info('=' * 60)
        logger.info("\n\n")

    def __should_count(self, count):
        return divmod(count, self.__interval)[1]

    def __sample(self, list_s, list_t, mode):
        if mode == 'add':
            s = random.choice(list_s)
            t = random.choice(list_t)
            while s == t or self.__graph.are_connected(s, t):
                s = random.choice(list_s)
                t = random.choice(list_t)
        else:
            s = random.choice(list_s)
            t = random.choice(list_t)
            while s == t or not self.__graph.are_connected(s, t):
                s = random.choice(list_s)
                t = random.choice(list_t)
        
        return (s, t)

    def __analyze(self):
        target_partitions = metric.sub_clustering(self.__target_node_list, self.__true_partitions)

        current_partitions = self.__detection_func(self.__graph, **self.__func_args)
        current_target_partitions = metric.sub_clustering(self.__target_node_list, current_partitions)

        nmi = current_target_partitions.compare_to(target_partitions, method='nmi')
        jaccard, recall, precision = metric.count_jaccard_recall_and_precision(target_partitions, current_target_partitions)

        logger.info(f"{self.__count:<5d} nmi: ({nmi:8.7f}), jaccard: ({jaccard:8.7f}), recall: ({recall:8.7f}), precision: ({precision:8.7f})")

    def __deception(self):
        for op in self.__operations:
            if op == 1: 
                ij = random.choices(range(len(self.__weight_matrix)), weights=self.__weight_matrix, k=1)[0]
                part_i = self.__true_partitions[divmod(ij, self.__partitions_num)[0]]
                part_j = self.__true_partitions[divmod(ij, self.__partitions_num)[1]]
                edge_to_add = self.__sample(part_i, part_j, mode='add')

                self.__graph.add_edge(*edge_to_add)
                self.__weight_matrix[ij] -= 1

                if not self.__should_count(self.__count):
                    self.__analyze()
                self.__count += 1
            else:
                i = random.choices(range(len(self.__weight_vector)), weights=self.__weight_vector, k=1)[0]
                part = self.__true_partitions[i]
                edge_to_del = self.__sample(part, part, mode='del')

                self.__graph.delete_edges([edge_to_del])
                self.__weight_vector[i] -= 1

                if not self.__should_count(self.__count):
                    self.__analyze()
                self.__count += 1
            

    def run(self):
        self.__preprocess()
        self.__start()
        self.__deception()
        self.__quit()