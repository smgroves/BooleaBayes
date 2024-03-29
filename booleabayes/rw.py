from . import utils as ut
from .plot import plot_histograms

import os
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from graph_tool import all as gt
from graph_tool.topology import label_components
from collections import Counter
import random
import multiprocessing as mp
from functools import partial

def random_walks(
    attractor_dict,
    rules,
    regulators_dict,
    nodes,
    save_dir,
    radius=2,
    perturbations=False,
    iters=1000,
    max_steps=500,
    stability=False,
    reach_or_leave="leave",
    random_start=0,
    on_nodes=[],
    off_nodes=[],
    basin=0,
    overwrite_walks=True,
    overwrite_perturbations=True,
    verbose = 2
):
    """
    Wrapper function to perform random walks.

    Parameters
    ----------
    attractor_dict : dictionary
        Dictionary of attractors used to get to steady states
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    save_dir : string or path-like object
        Path/directory to save output
    radius : int or list
        Single or multiple radii values for random walks
    perturbations : bool
        Whether to perform random walks with perturbations or not
    iters : int
        Number of iterations of random walks
    max_steps : int
        Max number of steps to take in random walks
    stability : bool
        Whether to perform stability testing with multiple radii or not
    reach_or_leave : string
        Define what type of random walk to perform
    random_start: int
        Whether to perform walks with a random list of start states. If >0, run that many random starts
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    basin : int
        Define a basin for random_walk_until_reach_basin
    overwrite_walks: bool, default True
        If False and walks/{start_idx} already exists, do not overwrite the results. Instead move on to the next start_idx. Note that if this is set to False and walks/{start_idx} already exists, the code skips all random walks for this start_idx (including any perturbations and stability testing)
    overwrite_perturbations: bool, default = True
        If False and perturbations/{start_idx} already exists, do not overwrite the results. Instead move on to the next start_idx.

    Returns
    -------
    None
    """
    # Determine type of radius and error check
    if type(radius) == list:
        for item in radius:
            if not isinstance(item, int) or item <= 0:
                raise Exception("Elements of radius list must be integers.")
    elif type(radius) == int:
        if radius <= 0:
            raise Exception("Radius must be an integer greater than 0.")
    else:
        raise Exception(
            "Radius must be an integer greater than 0 or a list of integers."
        )

    # Create output folder for walks
    try:
        os.mkdir(op.join(save_dir, "walks"))
    except FileExistsError:
        pass

    # Perform random walk (reach or leave) with stability
    if stability:
        # Make sure radius is a list, if not use default option
        if type(radius) != list:
            if verbose > 0: print("Performing stability with default list: [1,2,3,4,5,6,7,8]")
            radius = [1, 2, 3, 4, 5, 6, 7, 8]
    # If not doing stability, make the integer radius a list so it's iterable
    else:
        radius = [radius]

    # Run starting from steady_states
    for k in attractor_dict.keys():
        if verbose > 0: print(k)
        steady_states = attractor_dict[k]
        # Run random walk for every radius in list or the single integer
        for radius_ in radius:
            if verbose == 2: print("Radius: ", radius_)
            for start_idx in steady_states:
                switch_counts_0 = dict()
                for node in nodes:
                    switch_counts_0[node] = 0
                n_steps_to_leave_0 = []
                try:
                    os.mkdir(op.join(save_dir, "walks/%d" % start_idx))
                except (
                    FileExistsError
                ):  # if the walks were already done and overwrite_walks = False, skip this start_idx
                    if overwrite_walks:
                        pass
                    else:
                        continue

                outfile = open(
                    op.join(
                        save_dir, f"walks/%d/results_radius_{radius_}.csv" % start_idx
                    ),
                    "w+",
                )
                out_len = open(
                    op.join(save_dir, f"walks/%d/len_walks_{radius_}.csv" % start_idx),
                    "w+",
                )

                # Perform walks without perturbations first
                # Print progress of random walk every 10% of the way through iters
                prog = 0
                if verbose > 0: print("...Random walks for", start_idx)

                for iter_ in range(iters):
                    # print("Iteration:", iter_)
                    # print("Progress:")
                    if iter_ % 100 == 0:
                        if verbose == 2: print(str(iter_ / iters * 100) + "%")
                    #     prog = iter_ / 10
                    #     print("Progress: ", prog)

                    # 'counts': histogram of walk
                    # 'switches': count which TFs flipped
                    # 'distance': starting state to current state; walk until take max steps or leave basin
                    if reach_or_leave == "leave":
                        (
                            walk,
                            counts,
                            switches,
                            distances,
                        ) = random_walk_until_leave_basin(
                            start_idx,
                            rules,
                            regulators_dict,
                            nodes,
                            radius_,
                            max_steps=max_steps,
                            on_nodes=on_nodes,
                            off_nodes=off_nodes,
                        )
                    elif reach_or_leave == "reach":
                        (
                            walk,
                            counts,
                            switches,
                            distances,
                        ) = random_walk_until_reach_basin(
                            start_idx,
                            rules,
                            regulators_dict,
                            nodes,
                            radius=radius_,
                            max_steps=max_steps,
                            on_nodes=on_nodes,
                            off_nodes=off_nodes,
                            basin=basin,
                        )
                    else:
                        raise Exception(
                            "Value for `reach_or_leave` must be string 'reach' or 'leave'."
                        )

                    n_steps_to_leave_0.append(len(distances))
                    for node in switches:
                        if node is not None:
                            switch_counts_0[node] += 1
                    outfile.write(f"{walk}\n")
                    out_len.write(f"{len(walk)}\n")
                outfile.close()
                out_len.close()

                # Perform walks with perturbations
                if perturbations:
                    try:
                        os.mkdir(op.join(save_dir, "perturbations"))
                    except FileExistsError:
                        if overwrite_perturbations:
                            pass
                        else:
                            continue
                    # Run all possible single perturbations
                    if len(on_nodes) == 0 and len(off_nodes) == 0:
                        if verbose > 0: print("...Perturbations for ", start_idx)
                        try:
                            os.mkdir(op.join(save_dir, "perturbations/%d" % start_idx))
                        except FileExistsError:

                            pass

                        outfile = open(
                            op.join(
                                save_dir, f"perturbations/%d/results.csv" % start_idx
                            ),
                            "w+",
                        )

                        for expt_node in nodes:
                            # Arrays of # steps when activating or knocking out
                            n_steps_activate = []
                            n_steps_knockout = []
                            prog = 0

                            expt = "%s_activate" % expt_node
                            for iter_ in range(iters):
                                if iter_ % 100 == 0:
                                    if verbose == 2: print(str(iter_ / iters * 100) + "%")
                                # To perturb more than one node, add to on_nodes or off_nodes
                                if reach_or_leave == "leave":
                                    (
                                        walk_on,
                                        counts_on,
                                        switches_on,
                                        distances_on,
                                    ) = random_walk_until_leave_basin(
                                        start_idx,
                                        rules,
                                        regulators_dict,
                                        nodes,
                                        radius_,
                                        max_steps=max_steps,
                                        on_nodes=[
                                            expt_node,
                                        ],
                                        off_nodes=[],
                                    )
                                # elif reach_or_leave == "reach":

                                n_steps_activate.append(len(distances_on))

                            # mean of non-perturbed vs perturbed: loc_0 and loc_1
                            # histogram plots: inverse gaussian?
                            loc_0, loc_1, stabilized = plot_histograms(
                                n_steps_to_leave_0,
                                n_steps_activate,
                                expt,
                                bins=60,
                                fname=op.join(
                                    save_dir,
                                    "perturbations/%d/%s.pdf" % (start_idx, expt),
                                ),
                            )

                            outfile.write(
                                op.join(
                                    save_dir,
                                    "perturbations/%d,%s,%s,activate,%f\n"
                                    % (start_idx, k, expt_node, stabilized),
                                )
                            )
                            expt = "%s_knockdown" % expt_node
                            for iter_ in range(iters):
                                if iter_ % 100 == 0:
                                    if verbose == 2: print(str(iter_ / iters * 100) + "%")
                                (
                                    walk_off,
                                    counts_off,
                                    switches_off,
                                    distances_off,
                                ) = random_walk_until_leave_basin(
                                    start_idx,
                                    rules,
                                    regulators_dict,
                                    nodes,
                                    radius_,
                                    max_steps=max_steps,
                                    on_nodes=[],
                                    off_nodes=[
                                        expt_node,
                                    ],
                                )

                                n_steps_knockout.append(len(distances_off))

                            loc_0, loc_1, stabilized = plot_histograms(
                                n_steps_to_leave_0,
                                n_steps_knockout,
                                expt,
                                bins=60,
                                fname=op.join(
                                    save_dir,
                                    "perturbations/%d/%s.pdf" % (start_idx, expt),
                                ),
                            )
                            outfile.write(
                                op.join(
                                    save_dir,
                                    "perturbations/%d,%s,%s,knockdown,%f\n"
                                    % (start_idx, k, expt_node, stabilized),
                                )
                            )
                        outfile.close()

    if random_start > 0:
        try:
            os.mkdir(f"{save_dir}/walks/random")
        except FileExistsError:
            pass
        random_list = []
        for i in range(random_start):
            rand_state = random.choices([0, 1], k=len(nodes))
            rand_idx = ut.state_bool2idx(rand_state)
            random_list.append(rand_idx)
        for radius_ in radius:
            for start_idx in random_list:
                switch_counts_0 = dict()
                for node in nodes:
                    switch_counts_0[node] = 0
                n_steps_to_leave_0 = []
                try:
                    os.mkdir(op.join(save_dir, "walks/random/%d" % start_idx))
                except FileExistsError:
                    pass

                outfile = open(
                    op.join(
                        save_dir,
                        f"walks/random/%d/results_radius_{radius_}.csv" % start_idx,
                    ),
                    "w+",
                )
                out_len = open(
                    op.join(
                        save_dir, f"walks/random/%d/len_walks_{radius_}.csv" % start_idx
                    ),
                    "w+",
                )

                # Perform walks without perturbations first
                # Print progress of random walk every 10% of the way through iters
                prog = 0
                for iter_ in range(iters):
                    if iter_ % 100 == 0:
                        if verbose == 2: print(str(iter_ / iters * 100) + "%")

                    # 'counts': histogram of walk
                    # 'switches': count which TFs flipped
                    # 'distance': starting state to current state; walk until take max steps or leave basin
                    if reach_or_leave == "leave":
                        (
                            walk,
                            counts,
                            switches,
                            distances,
                        ) = random_walk_until_leave_basin(
                            start_idx,
                            rules,
                            regulators_dict,
                            nodes,
                            radius_,
                            max_steps=max_steps,
                            on_nodes=on_nodes,
                            off_nodes=off_nodes,
                        )
                    elif reach_or_leave == "reach":
                        (
                            walk,
                            counts,
                            switches,
                            distances,
                        ) = random_walk_until_reach_basin(
                            start_idx,
                            rules,
                            regulators_dict,
                            nodes,
                            radius=radius_,
                            max_steps=max_steps,
                            on_nodes=on_nodes,
                            off_nodes=off_nodes,
                            basin=basin,
                        )
                    else:
                        raise Exception(
                            "Value for `reach_or_leave` must be string 'reach' or 'leave'."
                        )

                    n_steps_to_leave_0.append(len(distances))
                    for node in switches:
                        if node is not None:
                            switch_counts_0[node] += 1
                    outfile.write(f"{walk}\n")
                    out_len.write(f"{len(walk)}\n")
                outfile.close()
                out_len.close()


def random_walks_parallel(
    attractor_dict,
    rules,
    regulators_dict,
    nodes,
    save_dir,
    radius=2,
    perturbations=False,
    iters=1000,
    max_steps=500,
    stability=False,
    reach_or_leave="leave",
    random_start=0,
    on_nodes=[],
    off_nodes=[],
    basin=0,
    overwrite_walks=True,
    overwrite_perturbations=True,
    cpu_usage=0.5,
    cpus = None,
    verbose = 1
):
    """
    Wrapper function to perform random walks.

    Parameters
    ----------
    attractor_dict : dictionary
        Dictionary of attractors used to get to steady states
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    save_dir : string or path-like object
        Path/directory to save output
    radius : int or list
        Single or multiple radii values for random walks
    perturbations : bool
        Whether to perform random walks with perturbations or not
    iters : int
        Number of iterations of random walks
    max_steps : int
        Max number of steps to take in random walks
    stability : bool
        Whether to perform stability testing with multiple radii or not
    reach_or_leave : string
        Define what type of random walk to perform
    random_start: int
        Whether to perform walks with a random list of start states. If >0, run that many random starts
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    basin : int
        Define a basin for random_walk_until_reach_basin
    overwrite_walks: bool, default True
        If False and walks/{start_idx} already exists, do not overwrite the results. Instead move on to the next start_idx. Note that if this is set to False and walks/{start_idx} already exists, the code skips all random walks for this start_idx (including any perturbations and stability testing)
    overwrite_perturbations: bool, default = True
        If False and perturbations/{start_idx} already exists, do not overwrite the results. Instead move on to the next start_idx.

    Returns
    -------
    None
    """
    if cpus is not None:
        cpus_use = cpus
    else:
        num_cores = mp.cpu_count()
        cpus_use = int(cpu_usage * num_cores)
    print(f"Using {cpus_use} cores to run random walks in parallel. ")
    print("Warning: Only use this function within __main__.")

    # List of attractor dicts with one value for one key
    list_attr_dict = []
    for k in attractor_dict.keys():
        for v in attractor_dict[k]:
            list_attr_dict.append({k: [v]})

    with mp.Pool(cpus_use) as P:

        results = P.map(
            partial(random_walks,  
                rules = rules,
                regulators_dict = regulators_dict,
                nodes = nodes,
                save_dir = save_dir,
                radius = radius,
                perturbations = perturbations,
                iters = iters,
                max_steps = max_steps,
                stability = stability,
                reach_or_leave = reach_or_leave,
                random_start = random_start,
                on_nodes = on_nodes,
                off_nodes = off_nodes,
                basin = basin,
                overwrite_walks = overwrite_walks,
                overwrite_perturbations = overwrite_perturbations, 
                verbose = verbose),
            list_attr_dict
        )
    

def simple_random_walk(stg, edge_weights, start_idx, steps):
    """
    Perform random walk on a state transition graph with known edge weights.

    Paramters
    ---------
    stg : graph tools Graph() object
        State transition graph
    start_idx : int
        Index of the vertext to start the walk
    steps : int
        Walk length

    Returns
    -------
    verts : list
        Path of vertices taken during random walk
    """
    verts = []
    next_vert = stg.vertex(start_idx)
    verts.append(next_vert)
    for i_ in range(steps):
        r = np.random.rand()
        running_p = 0
        for w in next_vert.out_neighbors():
            running_p += edge_weights[next_vert, w]
            if running_p > r:
                next_vert = w
                break
        verts.append(next_vert)
    return verts


def random_walk_until_leave_basin(
    start_state,
    rules,
    regulators_dict,
    nodes,
    radius=2,
    max_steps=10000,
    on_nodes=[],
    off_nodes=[],
):
    """
    Parameters
    ----------
    start_state : int
        Index of attractor to start walk from
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    radius : int
        Radius to stay within during walk
    max_steps : int
        Max number of steps to take in random walks
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    Returns
    -------
    walk : list
        Path of vertices taken during random walk
    Counter(walk) :
        Histogram of walk
    flipped_nodes : list
        Transcription factors that flipped during walk
    distances : list
        Starting state to next step in walk
    """
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes, range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []

    start_bool = [{"0": False, "1": True}[i] for i in ut.idx2binary(start_state, n)]
    for i, node in enumerate(nodes):
        if node in on_nodes:
            start_bool[i] = True
        elif node in off_nodes:
            start_bool[i] = False

    next_step = start_bool
    next_idx = ut.state_bool2idx(start_bool)
    distance = 0
    distances = []
    step_i = 0
    while distance <= radius and step_i < max_steps:
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes:
                continue
            neighbor_idx, flip = ut.update_node(
                rules, regulators_dict, node, node_i, nodes, node_indices, next_step
            )
            r = r - flip**2 / (1.0 * nu)
            if r <= 0:
                next_step = [
                    {"0": False, "1": True}[i] for i in ut.idx2binary(neighbor_idx, n)
                ]
                next_idx = neighbor_idx
                flipped_nodes.append(node)
                distance = ut.hamming(next_step, start_bool)
                break
        if r > 0:
            flipped_nodes.append(None)
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


def random_walk_until_reach_basin(
    start_state,
    rules,
    regulators_dict,
    nodes,
    radius=2,
    max_steps=10000,
    on_nodes=[],
    off_nodes=[],
    basin=1,
):
    """
    Parameters
    ----------
    start_state :
        .
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    radius : int
        Radius to stay within during walk
    max_steps : int
        Max number of steps to take in random walks
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    basin: int or list
        List of attractors to reach (or a single attractor) by state index (integer)
    Returns
    -------
    walk : list
        Path of vertices taken during random walk
    Counter(walk) :
    flipped_nodes : list

    distances : list
        All distances to basin
    """
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes, range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []

    start_bool = [{"0": False, "1": True}[i] for i in ut.idx2binary(start_state, n)]
    for i, node in enumerate(nodes):
        if node in on_nodes:
            start_bool[i] = True
        elif node in off_nodes:
            start_bool[i] = False

    next_step = start_bool
    next_idx = ut.state_bool2idx(start_bool)
    distance = 0
    if isinstance(basin, list):
        # Random high number to be replaced by actual distances
        min_dist = 200
        for i in basin:
            distance = ut.hamming_idx(start_state, i, len(nodes))
            if distance < min_dist:
                min_dist = distance
        distance = min_dist
    # Find the distance to a certain basin and stop when within radius
    elif isinstance(basin, int):
        distance = ut.hamming_idx(start_state, basin, len(nodes))
    else:
        print(
            "Only integer state or list of integer states accepted for basin argument."
        )

    distances = []
    step_i = 0
    while distance >= radius and step_i < max_steps:
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes:
                continue
            neighbor_idx, flip = ut.update_node(
                rules, regulators_dict, node, node_i, nodes, node_indices, next_step
            )
            r = r - flip**2 / (1.0 * nu)
            if r <= 0:
                next_step = [
                    {"0": False, "1": True}[i] for i in ut.idx2binary(neighbor_idx, n)
                ]
                next_idx = neighbor_idx
                flipped_nodes.append(node)

                if isinstance(basin, list):
                    # If basin is a list, loop through all attractors and find the distance to the closest one
                    min_dist = (
                        200  # Random high number to be replaced by actual distances
                    )
                    for i in basin:
                        distance = ut.hamming_idx(next_idx, i, len(nodes))
                        if distance < min_dist:
                            min_dist = distance
                    distance = min_dist
                elif isinstance(basin, int):
                    # If basin is an integer, find the distance to that attractor
                    distance = ut.hamming(next_step, basin)
                else:
                    print(
                        "Only integer state or list of integer states accepted for basin argument."
                    )
                break
        if r > 0:
            flipped_nodes.append(None)
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


def _long_random_walk(
    start_state,
    rules,
    regulators_dict,
    nodes,
    max_steps=10000,
    on_nodes=[],
    off_nodes=[],
):
    """
    Function to perform random walks on a BooleaBayes network out to max_steps
    The only difference with this function and bb.utils.random_walks_until_leave_basin() is that this function doesn't
    require a radius parameter, and will just keep walking until max_steps.

    Parameters
    ----------
    start_state : int
        Index of attractor to start walk from
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    max_steps : int
        Max number of steps to take in random walks
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    Returns
    -------
    walk : list
        Path of vertices taken during random walk
    Counter(walk) :
        Histogram of walk
    flipped_nodes : list
        Transcription factors that flipped during walk
    distances : list
        Starting state to next step in walk
    """
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes, range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []

    start_bool = [{"0": False, "1": True}[i] for i in ut.idx2binary(start_state, n)]
    for i, node in enumerate(nodes):
        if node in on_nodes:
            start_bool[i] = True
        elif node in off_nodes:
            start_bool[i] = False

    next_step = start_bool
    next_idx = ut.state_bool2idx(start_bool)
    distance = 0
    distances = []
    step_i = 0
    while step_i < max_steps:
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes:
                continue
            neighbor_idx, flip = ut.update_node(
                rules, regulators_dict, node, node_i, nodes, node_indices, next_step
            )
            r = r - flip**2 / (1.0 * nu)
            if r <= 0:
                next_step = [
                    {"0": False, "1": True}[i] for i in ut.idx2binary(neighbor_idx, n)
                ]
                next_idx = neighbor_idx
                flipped_nodes.append(node)
                distance = ut.hamming(next_step, start_bool)
                break
        if r > 0:
            flipped_nodes.append(None)
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


def long_random_walks(
    starting_attractors,
    attractor_dict,
    rules,
    regulators_dict,
    nodes,
    save_dir,
    on_nodes=[],
    off_nodes=[],
    max_steps=2000,
    iters=100,
    overwrite_walks=False,
):
    """Alternative way to do random walks (until reach basin). Instead of looking for a specific basin, keep walking some length of steps.
    Can be used for visualizing effects of perturbations using bb.plot.plot_random_walks.

    :param starting_attractors: name of the attractors to start the walk from (key in attractor_dict)
    :type starting_attractors: list
    :param attractor_dict: Dictionary of attractors
    :type attractor_dict: dict()
    :param rules: Rules from BooleaBayes rule fitting
    :type rules: dict()
    :param regulators_dict: Dictionary of regulators from rule fitting
    :type regulators_dict: dict()
    :param nodes: list of nodes in the network
    :param save_dir: Directory to save output
    :type save_dir: str
    :param on_nodes: activating perturbations to run simulations for, defaults to []
    :type on_nodes: list, optional
    :param off_nodes: knockdown perturbations to run simulations for, defaults to []
    :type off_nodes: list, optional
    :param max_steps: Length of random walks, defaults to 2000
    :type max_steps: int, optional
    :param iters: Number of iterations to run, defaults to 100
    :type iters: int, optional
    :param overwrite_walks: If false, don't rewrite walks if the folder already exists, defaults to False
    :type overwrite_walks: bool, optional
    """
    try:
        os.mkdir(f"{save_dir}/walks/long_walks/")
    except FileExistsError:
        pass
    for s in starting_attractors:
        print(s)
        for start_idx in attractor_dict[s]:
            print("Starting state: ", start_idx)

            switch_counts_0 = dict()
            for node in nodes:
                switch_counts_0[node] = 0
            n_steps_to_leave_0 = []
            try:
                os.mkdir(f"{save_dir}/walks/long_walks/{max_steps}_step_walks/")
            except FileExistsError:
                pass

            try:
                os.mkdir(
                    f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}"
                )
            except FileExistsError:
                if overwrite_walks:
                    pass
                else:
                    continue

            outfile = open(
                f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results.csv",
                "w+",
            )

            # 1000 iterations; print progress of random walk every 10% of the way
            # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
            # no perturbations
            for iter_ in range(iters):
                if iter_ % 10 == 0:
                    print(str(iter_ / iters * 100) + "%")
                walk, counts, switches, distances = _long_random_walk(
                    start_idx, rules, regulators_dict, nodes, max_steps=max_steps
                )
                n_steps_to_leave_0.append(len(distances))
                for node in switches:
                    if node is not None:
                        switch_counts_0[node] += 1
                outfile.write(f"{walk}\n")
            outfile.close()

        print("Running TF Perturbations...")

        for perturb in off_nodes:
            for start_idx in attractor_dict[s]:
                print("Starting state: ", start_idx)
                print("Perturbation: ", perturb)

                switch_counts_0 = dict()
                for node in nodes:
                    switch_counts_0[node] = 0
                n_steps_to_leave_0 = []

                outfile = open(
                    f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results_{perturb}_kd.csv",
                    "w+",
                )

                # 1000 iterations; print progress of random walk every 10% of the way
                # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
                # no perturbations
                for iter_ in range(iters):
                    if iter_ % 10 == 0:
                        print(str(iter_ / iters * 100) + "%")
                    walk, counts, switches, distances = _long_random_walk(
                        start_idx,
                        rules,
                        regulators_dict,
                        nodes,
                        max_steps=max_steps,
                        off_nodes=[perturb],
                    )
                    n_steps_to_leave_0.append(len(distances))
                    for node in switches:
                        if node is not None:
                            switch_counts_0[node] += 1
                    outfile.write(f"{walk}\n")
                outfile.close()

        for perturb in on_nodes:
            for start_idx in attractor_dict[s]:
                print("Starting state: ", start_idx)
                print("Perturbation: ", perturb)

                switch_counts_0 = dict()
                for node in nodes:
                    switch_counts_0[node] = 0
                n_steps_to_leave_0 = []

                outfile = open(
                    f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results_{perturb}_act.csv",
                    "w+",
                )

                # 1000 iterations; print progress of random walk every 10% of the way
                # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
                # no perturbations
                for iter_ in range(iters):
                    if iter_ % 10 == 0:
                        print(str(iter_ / iters * 100) + "%")
                    walk, counts, switches, distances = _long_random_walk(
                        start_idx,
                        rules,
                        regulators_dict,
                        nodes,
                        max_steps=max_steps,
                        on_nodes=[perturb],
                    )
                    n_steps_to_leave_0.append(len(distances))
                    for node in switches:
                        if node is not None:
                            switch_counts_0[node] += 1
                    outfile.write(f"{walk}\n")
                outfile.close()
