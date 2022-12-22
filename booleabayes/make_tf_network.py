from . import enrichr
import networkx as nx
import time
import os


def prune(G_orig, prune_sources=True, prune_sinks=True):
    """Prune graph to remove nodes with no incoming or outgoing edges

    :param G_orig: NetworkX graph to prune
    :type G_orig: networkx.DiGraph
    :param prune_sources: Remove nodes with no incoming edges, defaults to True
    :type prune_sources: bool, optional
    :param prune_sinks: Remove nodes with no outgoing edges, defaults to True
    :type prune_sinks: bool, optional
    :return: Pruned network
    :rtype: networkx.DiGraph
    """
    G = G_orig.copy()
    n = len(G.nodes())
    nold = n + 1

    while n != nold:
        nold = n
        for tf in list(G.nodes()):
            if prune_sources == True:
                if G.in_degree(tf) == 0:
                    G.remove_node(tf)
            if prune_sinks == True:
                if G.out_degree(tf) == 0:
                    G.remove_node(tf)
            else:
                if G.in_degree(tf) == 0 and G.out_degree(tf) == 0:
                    G.remove_node(tf)
        n = len(G.nodes())
    return G


def prune_info(G_orig, prune_self_loops=True):
    """Prune graph to only include edges with evidence in multiple databases

    :param G_orig: NetworkX graph to prune
    :type G_orig: networkx.DiGraph
    :param prune_self_loops: Whether to prune self-loops in the network (edges where parent node == child node), defaults to True
    :type prune_self_loops: bool, optional
    :return: Pruned network
    :rtype: networkx.DiGraph
    """
    G = G_orig.copy()
    for tf in list(G.nodes()):
        edges = G.adj[tf]
        for target in list(edges.keys()):
            if tf == target and prune_self_loops:
                G.remove_edge(tf, target)
                continue
            if "db" not in edges[target]:
                G.remove_edge(tf, target)
            elif len(edges[target]["db"]) < 2:
                G.remove_edge(tf, target)
    return prune(G)


def prune_to_chea(G_orig, prune_self_loops=True):
    """Prune graph to only include edges with evidence in ChEA databases

    :param G_orig: NetworkX graph to prune
    :type G_orig: networkx.DiGraph
    :param prune_self_loops: Whether to prune self-loops in the network (edges where parent node == child node), defaults to True
    :type prune_self_loops: bool, optional
    :return: Pruned network
    :rtype: networkx.DiGraph
    """
    G = G_orig.copy()
    for tf in list(G.nodes()):
        edges = G.adj[tf]
        for target in list(edges.keys()):
            if tf == target and prune_self_loops:
                G.remove_edge(tf, target)
                continue
            if "db" in edges[target]:
                if not True in ["ChEA" in i for i in edges[target]["db"]]:
                    G.remove_edge(tf, target)
    #                if len(edges[target]['db']) < 2: G.remove_edge(tf, target)
    return prune(G)


def make_network(
    tfs,
    outdir="",
    do_prune=True,
    prune_sinks=True,
    prune_sources=True,
    do_prune_info=True,
    prune_self_loops=True,
    do_prune_to_chea=True,
    save_unfiltered=False,
    network_name="network",
):
    """Make network from list of tfs and save as csv files with various levels of pruning.

    :param tfs: List of transcription factor gene names that will be searched in enrichR databases.
    :type tfs: List[str]
    :param outdir: Output directory where network csvs will be saved, defaults to ""
    :type outdir: str or None
    :param do_prune: Prune network to remove nodes with no sinks and/or sources and generate a new network file called <network_name>_pruned.csv, defaults to True
    :type do_prune: bool, optional
    :param prune_sinks: Whether to prune sink nodes from the network (no outgoing nodes), defaults to True
    :type prune_sinks: bool, optional
    :param prune_sources: Whether to prune source nodes the network (no incoming nodes), defaults to True
    :type prune_sources: bool, optional
    :param do_prune_info: Whether to prune the network to edges with evidence in more than one database from enrichR and generate a new network file called <network_name>_high_evidence.csv, defaults to True
    :type do_prune_info: bool, optional
    :param prune_self_loops: Whether to prune self-loops in the network (edges where parent node == child node), defaults to True
    :type prune_self_loops: bool, optional
    :param do_prune_to_chea: Whether to prune the network to only edges with evidence in ChEA databases and generate a new network file called <network_name>_chea.csv, defaults to True
    :type do_prune_to_chea: bool, optional
    :param save_unfiltered: Whether to save the unfiltered network before pruning, defaults to False. Note that if all pruning options are False and this is set to False, no network will be saved.
    :type save_unfiltered: bool, optional
    :param network_name: Prefix name of network csv files, defaults to `network`
    :type network_name: str
    :return: Network with highest level of pruning
    :rtype: NetworkX Graph
    """

    G = nx.DiGraph()
    # prelim_G = nx.DiGraph()
    # with open("/Users/sarahmaddox/Dropbox (Vanderbilt)/Quaranta_Lab/SCLC/Network/mothers_network.csv") as infile:
    #     for line in infile:
    #         line = line.strip().split(',')
    #         prelim_G.add_edge(line[0], line[1])

    for tf in tfs:
        G.add_node(tf)

    for tf in tfs:
        enrichr.build_tf_network(G, tf, tfs)
        time.sleep(1)

    # for edge in prelim_G.edges():
    #     if edge[0] in tfs and edge[1] in tfs:
    #         G.add_edge(edge[0], edge[1])

    if save_unfiltered:
        outfile = open(
            os.path.join(outdir, f"{network_name}_unfiltered.csv"),
            "w",
        )
        for edge in G.edges():
            outfile.write("%s,%s\n" % (edge[0], edge[1]))
        outfile.close()

    if do_prune:
        Gp = prune(G, prune_sinks=prune_sinks, prune_sources=prune_sources)

        outfile = open(
            os.path.join(outdir, f"{network_name}_pruned.csv"),
            "w",
        )
        for edge in Gp.edges():
            outfile.write("%s,%s\n" % (edge[0], edge[1]))
        outfile.close()
    else:
        Gp = G

    if do_prune_info:
        Gpp = prune_info(Gp, prune_self_loops=prune_self_loops)

        outfile = open(
            os.path.join(outdir, f"{network_name}_high_evidence.csv"),
            "w",
        )
        for edge in Gpp.edges():
            outfile.write("%s,%s\n" % (edge[0], edge[1]))
        outfile.close()
    else:
        Gpp = Gp

    if do_prune_to_chea:
        Gppp = prune_to_chea(Gpp, prune_self_loops=prune_self_loops)

        outfile = open(
            os.path.join(outdir, f"{network_name}_chea.csv"),
            "w",
        )
        for edge in Gppp.edges():
            outfile.write("%s,%s\n" % (edge[0], edge[1]))
        outfile.close()
    else:
        Gppp = Gpp

    return Gppp
