from typing import List
import logging.config
from settings import LOGGING_SETTINGS
import random
import sys
import metric

logging.config.dictConfig(LOGGING_SETTINGS)
logger = logging.getLogger('normal')

class ProHiCoDCSBM(object):
    ''' The DCSBM algorithm in the ProHiCo framework to hide specified communities
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
        self.__weight_matrix: List = [0] * (self.__partitions_num * self.__partitions_num)
        self.__weight_vector: List = [0] * self.__partitions_num
        self.__target_node_list: List = list()
        self.__sorted_partitions: List[List[int]] = list()

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

        for idx, part in enumerate(self.__true_partitions):
            self.__sorted_partitions.append(sorted(part, key=lambda x: self.__graph.degree(x)))
        
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
        logger.info(f'Deception: prohico-dcsbm')
        logger.info(f'Interval: {self.__interval}')
        logger.info('=' * 60)

    def __quit(self):
        logger.info('=' * 60)
        logger.info("\n\n")

    def __should_count(self, count):
        return divmod(count, self.__interval)[1]

    def __analyze(self):
        target_partitions = metric.sub_clustering(self.__target_node_list, self.__true_partitions)

        current_partitions = self.__detection_func(self.__graph, **self.__func_args)
        current_target_partitions = metric.sub_clustering(self.__target_node_list, current_partitions)

        nmi = current_target_partitions.compare_to(target_partitions, method='nmi')
        jaccard, recall, precision = metric.count_jaccard_recall_and_precision(target_partitions, current_target_partitions)

        logger.info(f"{self.__count:<5d} nmi: ({nmi:8.7f}), jaccard: ({jaccard:8.7f}), recall: ({recall:8.7f}), precision: ({precision:8.7f})")

    def __find_optimal_edge2add(self, part_a, part_b):
        ahead, atail = 0, len(part_a) - 1
        bhead, btail = 0, len(part_b) - 1
        T = sys.maxsize

        while ahead <= atail and bhead <= btail:
            for k in range(bhead, btail + 1):
                if self.__graph.degree(part_a[ahead]) * self.__graph.degree(part_b[k]) >= T:
                    btail = k - 1
                    break
                if not self.__graph.are_connected(part_a[ahead], part_b[k]):
                    u, v = part_a[ahead], part_b[k]
                    T = self.__graph.degree(u) * self.__graph.degree(v)
                    btail = k - 1
                    break
            for k in range(ahead, atail + 1):
                if self.__graph.degree(part_b[bhead]) * self.__graph.degree(part_a[k]) >= T:
                    atail = k - 1
                    break
                if not self.__graph.are_connected(part_b[bhead], part_a[k]):
                    u, v = part_b[bhead], part_a[k]
                    T = self.__graph.degree(u) * self.__graph.degree(v)
                    atail = k - 1
                    break
            ahead += 1
            bhead += 1

        return (u, v)

    def __find_optimal_edge2del(self, part):
        start, end = len(part) - 1, 0
        T = -1
        while start > end:
            for k in range(start - 1, end - 1, -1):
                if self.__graph.degree(part[k]) * self.__graph.degree(part[start]) <= T:
                    end = k + 1
                    break
                if self.__graph.are_connected(part[k], part[start]):
                    u, v = part[start], part[k]
                    T = self.__graph.degree(u) * self.__graph.degree(v)
                    end = k + 1
                    break
            start -= 1
        
        return (u, v)

    def __deception(self):
        for op in self.__operations:
            if op == 1: 
                ij = random.choices(range(len(self.__weight_matrix)), weights=self.__weight_matrix, k=1)[0]
                i, j = divmod(ij, self.__partitions_num)[0], divmod(ij, self.__partitions_num)[1]
                part_i, part_j = self.__sorted_partitions[i], self.__sorted_partitions[j]
                edge_to_add = self.__find_optimal_edge2add(part_i, part_j)

                self.__graph.add_edge(*edge_to_add)
                self.__weight_matrix[ij] -= 1
                self.__sorted_partitions[i].sort(key=lambda x: self.__graph.degree(x))
                self.__sorted_partitions[j].sort(key=lambda x: self.__graph.degree(x))

                if not self.__should_count(self.__count):
                    self.__analyze()
                self.__count += 1
            else:
                i = random.choices(range(len(self.__weight_vector)), weights=self.__weight_vector, k=1)[0]
                part = self.__sorted_partitions[i]
                edge_to_del = self.__find_optimal_edge2del(part)

                self.__graph.delete_edges([edge_to_del])
                self.__weight_vector[i] -= 1
                self.__sorted_partitions[i].sort(key=lambda x: self.__graph.degree(x))

                if not self.__should_count(self.__count):
                    self.__analyze()
                self.__count += 1
            
    
    def run(self):
        self.__preprocess()
        self.__start()
        self.__deception()
        self.__quit()
