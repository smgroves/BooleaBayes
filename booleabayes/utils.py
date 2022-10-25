import os
import os.path as op
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import pickle
from scipy import stats
from graph_tool import all as gt
from graph_tool.topology import label_components

def idx2binary(idx, n):
    """Convert index (int, base 10) to a binary str

    :param idx: index value of vertex (base 10)
    :type idx: int
    :param n: base for binary conversion
    :type n: int
    :return: binary version of index
    :rtype: str
    """
    # For debugging:
    # print('idx2binary args:',idx, n)
    binary = "{0:b}".format(idx)
    return "0" * (n - len(binary)) + binary

def state2idx(state):
    """Convert binary str to index (int, base 10)

    :param state: binary version of index
    :type state: str
    :return: index value of vertex (base 10)
    :rtype: int
    """
    return int(state, 2)

# Returns 0 if state is []
def state_bool2idx(state):
    """Convert list of boolean values (T/F) to index (int, base 10)

    :param state: Boolean version of state
    :type state: list of bool values
    :return: index value of vertex (base 10)
    :rtype: int
    """
    n = len(state) - 1
    d = dict({True: 1, False: 0})
    idx = 0
    for s in state:
        idx += d[s] * 2 ** n
        n -= 1
    return idx

# Hamming distance between 2 states
def hamming(x, y):
    s = 0
    for i, j in zip(x, y):
        if i != j: s += 1
    return s

# Hamming distance between 2 states, where binary states are given by decimal code
def hamming_idx(x, y, n):
    return hamming(idx2binary(x, n), idx2binary(y, n))

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

# Split a dataset into testing and training dataset
# save dir = directory to save the split data 
# fname = str filename to append to default save file name
def split_train_test(data, data_t1, clusters, save_dir, fname=None, random_state = 1234):
    """Split a dataset into testing and training dataset

    :param data: Dataset or first timepoint of temporal dataset to be split into training/testing datasets
    :type data: Pandas dataframe
    :param data_t1: Second timepoint of temporal dataset, optional
    :type data_t1: {Pandas dataframe, None}
    :param clusters: Cluster assignments for each sample; see ut.get_clusters() to generate
    :type clusters: Pandas DataFrame
    :param save_dir: File path for saving training and testing sets
    :type save_dir: str
    :param fname: Suffix to add to file names for saving, defaults to None
    :type fname: str, optional
    :return: List of dataframes split into training and testing: `data` (training set, t0), test (testing set, t1), data_t1 (training set, t1), test_t (testing set, t1), clusters_train (cluster IDs of training set), clusters_test (cluster IDs of testing set)
    :rtype: Pandas dataframes
    """
    df = list(data.index)

    # print("Splitting into train and test datasets...")
    kf = ms.StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    train_index, test_index = next(kf.split(df, clusters.loc[df, 'class']))

    T = {'test_cellID': [df[i] for i in test_index], 'test_index': test_index,
         'train_index': train_index,
         'train_cellID': [df[i] for i in train_index]}
    
    with open(f'{save_dir}/test_train_indices_{fname}.p', 'wb') as f:
        pickle.dump(T, f)
        
    test = data.loc[T['test_cellID']]
    data = data.loc[T['train_cellID']]
    test.to_csv(f'{save_dir}/test_t0_{fname}.csv')
    data.to_csv(f'{save_dir}/train_t0_{fname}.csv')

    if data_t1 is not None:
        test_t1 = data_t1.loc[T['test_cellID']]
        data_t1 = data_t1.loc[T['train_cellID']]
        test_t1.to_csv(f'{save_dir}/test_t1_{fname}.csv')
        data_t1.to_csv(f'{save_dir}/train_t1_{fname}.csv')
    else:
        test_t1 = None
        data_t1 = None

    clusters_train = clusters.loc[T['train_cellID']]
    clusters_test = clusters.loc[T['test_cellID']]
    clusters_train.to_csv(f'{save_dir}/clusters_train_{fname}.csv')
    clusters_test.to_csv(f'{save_dir}/clusters_test_{fname}.csv')

    return data, test, data_t1, test_t1, clusters_train, clusters_test

# Given a graph calculate the graph condensation (all nodes are reduced to strongly
# connected components). Returns the condensation graph, a dictionary mapping
# SCC->[nodes in G], as well as the output of graph_tool's label_components.
# I often use this on the output of graph_sim.prune_stg_edges, or a deterministic stg
def condense(G, directed=True, attractors=True):
    # label_components comes from graph_tool directly
    components = label_components(G, directed=directed, attractors=attractors)
    c_G = gt.Graph()
    c_G.add_vertex(n=len(components[1]))

    vertex_dict = dict()
    for v in c_G.vertices(): vertex_dict[int(v)] = []
    component = components[0]

    for v in G.vertices():
        c = component[v]
        vertex_dict[c].append(v)
        for w in v.out_neighbors():
            cw = component[w]
            if cw == c: continue
            if c_G.edge(c, cw) is None:
                edge = c_G.add_edge(c, cw)
    return c_G, vertex_dict, components
 
def average_state(idx_list, n):
    av = np.zeros(n)
    for idx in idx_list:
        av = av + np.asarray([float(i) for i in idx2binary(idx, n)]) / (1. * len(idx_list))
    return av

# look at how likely it is to leave a state
def inspect_state(i, stg, vidx, rules, regulators_dict, nodes, n):
    v = stg.vertex(i)
    for a in zip(nodes,idx2binary(vidx[v],n), get_flip_probs(vidx[v], rules, regulators_dict, nodes)): 
        print(a)
    print(max(get_flip_probs(vidx[v], rules, regulators_dict, nodes)))
    print(sum(get_flip_probs(vidx[v], rules, regulators_dict, nodes)))

def update_node(rules, regulators_dict, node, node_i, nodes, node_indices, state_bool, return_state=False):
    """_summary_

    :param rules: _description_
    :type rules: _type_
    :param regulators_dict: _description_
    :type regulators_dict: _type_
    :param node: _description_
    :type node: _type_
    :param node_i: _description_
    :type node_i: _type_
    :param nodes: _description_
    :type nodes: _type_
    :param node_indices: _description_
    :type node_indices: _type_
    :param state_bool: _description_
    :type state_bool: _type_
    :param return_state: _description_, defaults to False
    :type return_state: bool, optional
    :return: _description_
    :rtype: _type_
    """
    rule = rules[node]
    regulators = regulators_dict[node]
    regulator_indices = [node_indices[i] for i in regulators]
    regulator_state = [state_bool[i] for i in regulator_indices]
    rule_leaf = state_bool2idx(regulator_state)
    flip = rule[rule_leaf]
    if state_bool[node_i]: flip = 1-flip
    
    neighbor_state = [i for i in state_bool]
    neighbor_state[node_i] = not neighbor_state[node_i]
    neighbor_idx = state_bool2idx(neighbor_state)
    
    if return_state: return neighbor_idx, neighbor_state, flip
    return neighbor_idx, flip

# Given an stg with edge weights, each pair of neighboring states has a weighted edge
# from A->B and another from B->A. This prunes all edges with weight < threshold.
# WARNING: If threshold > 0.5, it is possible for both A->B and B->A to get pruned.
# If you are using a reprogrammed stg, make sure to use nu, not n
def prune_stg_edges(stg, edge_weights, n, threshold = 0.5):
    d_stg = gt.Graph()
    for edge in stg.edges():
        if edge_weights[edge]*n >= threshold:
            d_stg.add_edge(edge.source(), edge.target())
    return d_stg
    # print("Finding strongly connected components")

    

### ------------ GETTERS ------------ ###

def get_nodes(vertex_dict, graph):
    v_names = graph.vertex_properties['name'] 
    nodes = sorted(vertex_dict.keys())
    
    return v_names, nodes
    
def get_clusters(data, data_test=None, is_data_split=False, cellID_table=None, cluster_header_list=None):
    # Dataset passed (data) is not split into a training and test set
    if is_data_split == False:
        if cellID_table is not None and cluster_header_list is not None:
            clusters = pd.read_csv(cellID_table, index_col=0, header=0, delimiter=',')
            clusters.columns = cluster_header_list
        else:
            clusters = pd.DataFrame([0] * len(data.index), index=data.index, columns=['class'])
    # Datasets passed are the training (data) and test (data_test) set 
    elif is_data_split == True and data_test is not None:
        if cellID_table is not None and cluster_header_list is not None:
            clusters = pd.read_csv(cellID_table, index_col=0, header=0, delimiter=',')
            clusters.columns = cluster_header_list
            clusters_train = clusters.loc[data.index]
            clusters_test = clusters.loc[data_test.index]
        else:
            clusters = pd.DataFrame([0]*len(data.index), index = data.index, columns=['class'])
     
    return clusters
    
def get_avg_state_index(nodes, average_states, outfile, save_dir=None):
    n = len(nodes)
    
    for j in nodes:
        outfile.write(f",{j}")
    outfile.write("\n")
    
    for k in average_states.keys():
        file_idx = open(f'{save_dir}/average_states_idx_{k}.txt', 'w+')
        file_idx.write('average_state\n')
        att = idx2binary(average_states[k], n)
        outfile.write(f"{k}")
        for i in att:
            outfile.write(f",{i}")
            file_idx.write(f"{i}\n")
        outfile.write("\n")
        file_idx.close()
    outfile.close()

def get_reprogramming_rules(rules, regulators_dict, on_nodes, off_nodes):
    rules = rules.copy()
    regulators_dict = regulators_dict.copy()
    for node in on_nodes:
        rules[node] = np.asarray([1.])
        regulators_dict[node] = []
    for node in off_nodes:
        rules[node] = np.asarray([0.])
        regulators_dict[node] = []
    return rules, regulators_dict

def get_flip_probs(idx, rules, regulators_dict, nodes, node_indices = None):
    n = len(nodes)
    if node_indices is None: node_indices = dict(zip(nodes,range(len(nodes))))
    state_bool = [{'0':False,'1':True}[i] for i in idx2binary(idx,n)]
    flips = []
    for i,node in enumerate(nodes):
        rule = rules[node]
        regulators = regulators_dict[node]
        regulator_indices = [node_indices[j] for j in regulators]
        regulator_state = [state_bool[j] for j in regulator_indices]
        rule_leaf = state_bool2idx(regulator_state)

        flip = rule[rule_leaf]
        if state_bool[i]: flip = 1-flip
        flips.append(flip)
        
    return flips
    
def get_avg_min_distance(binarized_data, n, min_dist=20):
    dist_dict = dict()
    
    for k in sorted(binarized_data.keys()):
        print(k)
        distances = []
        for s in binarized_data[k]:
            if len(binarized_data[k]) == 1:
                distances = [4]
            else:
                # min_dist = 20
                for t in binarized_data[k]:
                    if s == t: pass
                    else:
                        dist = hamming_idx(s,t,n)
                        if dist < min_dist: min_dist = dist
                distances.append(min_dist)
        try:
            dist_dict[k] = int(np.ceil(np.mean(distances)))
            print("Average minimum distance between cells in cluster: ", dist_dict[k])
        except ValueError: print("Not enough data in group to find distances.")
    return dist_dict        
     
# Returns dict mapping attractor components to states (idx) within that component
# atts = list of True/False indicating whether component_i is an attractor or not
# c_vertex_dict = dict mapping vertex component to list of states in the component
# vert_idx is a vertex_property mapping vertex -> idx. Used if the internal index
# of vertices in the stg are not equivalent to their state index.
def get_attractors(atts, c_vertex_dict, vert_idx = None):
    attractors = dict()
    for i, is_attractor in enumerate(atts):
        if is_attractor:
            if vert_idx is None: attractors[i] = [int(state) for state in c_vertex_dict[i]]
            else: attractors[i] = [vert_idx[state] for state in c_vertex_dict[i]]
    return attractors

# Given probabilistic rules! Simulates the probabilistic graph/rules and
# returns a state transition graph and edge_weights
def get_partial_stg(start_states, rules, nodes, regulators_dict, radius, on_nodes = [], off_nodes = [], pthreshold=0.):
    """Simulates the probabilistic graph/rules and returns a state transition graph and edge_weights

    :param start_states: list of start state indices (int, base 10)
    :type start_states: list of int
    :param rules: dictionary of rules in the form {gene_name: [list of leaf probabilities]}
    :type rules: dict
    :param nodes: list of nodes in network
    :type nodes: list of str
    :param regulators_dict: _description_
    :type regulators_dict: _type_
    :param radius: _description_
    :type radius: _type_
    :param on_nodes: _description_, defaults to []
    :type on_nodes: list, optional
    :param off_nodes: _description_, defaults to []
    :type off_nodes: list, optional
    :param pthreshold: _description_, defaults to 0.
    :type pthreshold: _type_, optional
    :return: state transition graph and edge_weights
    :rtype: _type_
    """
    stg = gt.Graph()
    edge_weights = stg.new_edge_property('float')
    stg.edge_properties['weight'] = edge_weights

    vert_idx = stg.new_vertex_property('long') # This is the vertex's real idx, which corresponds to it's real state
    stg.vertex_properties['idx'] = vert_idx

    node_indices = dict(zip(nodes,range(len(nodes))))
    n = len(nodes)
    
    unperturbed_nodes = [i for i in nodes if not i in on_nodes+off_nodes]
    nu = len(unperturbed_nodes)
    norm_fact = 1.*nu
    
    added_indices = set()
    pending_vertices = set()
    out_of_bounds_indices = set()
    # if it reaches the radius without finding an attractor, it is out of bounds
    out_of_bounds_vertex = stg.add_vertex()
    vert_idx[out_of_bounds_vertex]=-1
    
    # Add the start states to the stg
    idx_vert_dict = dict()
    for idx in start_states:
        v = stg.add_vertex()
        vert_idx[v]=idx
        idx_vert_dict[idx]=v
        added_indices.add(idx)
        
    # Add edges from the start states to 
    for idx in start_states:
        v = idx_vert_dict[idx]
        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]
        neighbor_bool = [i for i in state_bool]
        for node in on_nodes:
            ni = node_indices[node]
            neighbor_bool[ni]=True
        for node in off_nodes:
            ni = node_indices[node]
            neighbor_bool[ni]=False
        neighbor_idx = state_bool2idx(neighbor_bool)
        
        if (neighbor_idx != idx):
            if neighbor_idx in added_indices: w = idx_vert_dict[neighbor_idx]
            else:
                w = stg.add_vertex()
                vert_idx[w]=neighbor_idx
                added_indices.add(neighbor_idx)
                idx_vert_dict[neighbor_idx] = w
            edge = stg.add_edge(v, w)
            edge_weights[edge]=1.
            pending_vertices.add(w)
        else: pending_vertices.add(v)
    
    start_states = set([idx2binary(i,n) for i in start_states]) # This is remembered and used to calculate the hamming distance of every visited state from the start states
    start_bools = [[{'0':False,'1':True}[i] for i in state] for state in start_states]
        
    # Go through the full list of visited vertices
    while len(pending_vertices) > 0:

        # Get the state for the next vertex
        v = pending_vertices.pop()
        idx = vert_idx[v]
        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]

        # Go through all the nodes and update it
        for node_i, node in enumerate(nodes):
            if not node in unperturbed_nodes: continue
            neighbor_idx, neighbor_state, flip_prob = update_node(rules, regulators_dict, node, node_i, nodes, node_indices, state_bool, return_state=True)

            if (flip_prob > pthreshold): # Add an edge to this neighbor
            
                # Have we seen this neighbor before in out_of_bounds?
                if neighbor_idx in out_of_bounds_indices:
                    if out_of_bounds_vertex in v.out_neighbors():
                        edge = stg.edge(v,out_of_bounds_vertex)
                        edge_weights[edge] += flip_prob / norm_fact
                    else:
                        edge = stg.add_edge(v, out_of_bounds_vertex)
                        edge_weights[edge] = flip_prob / norm_fact
                    continue
                        
                # Otherwise check to see if it IS out of bounds   
                min_dist = radius + 1
                for start_bool in start_bools:
                    dist = hamming(start_bool, neighbor_state)
                    if dist < min_dist:
                        min_dist = dist
                        break
                if min_dist > radius: # If it is out of bounds, add an edge to out_of_bounds_vertex
                    out_of_bounds_indices.add(neighbor_idx) # Add it to out_of_bounds_indices
                    if out_of_bounds_vertex in v.out_neighbors():
                        edge = stg.edge(v,out_of_bounds_vertex)
                        edge_weights[edge] += flip_prob / norm_fact
                    else:
                        edge = stg.add_edge(v, out_of_bounds_vertex)
                        edge_weights[edge] = flip_prob / norm_fact
                    continue
                        
                else: # Otherwise, it is in bounds
                    if neighbor_idx in added_indices: w = idx_vert_dict[neighbor_idx]
                    else: # Create the neighbor, and add it to the pending_vertices set
                        w = stg.add_vertex()
                        vert_idx[w]=neighbor_idx
                        added_indices.add(neighbor_idx)
                        idx_vert_dict[neighbor_idx] = w
                        pending_vertices.add(w)
                
                    # Either way, add an edge from the current state to this one
                    edge = stg.add_edge(v, w)
                    edge_weights[edge]=flip_prob / norm_fact
                 
    return stg, edge_weights

