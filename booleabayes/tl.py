from . import utils as ut
from .plot import *

import pandas as pd
import numpy as np
import resource
import os
import os.path as op
import pickle
import seaborn as sns
from scipy import stats
from graph_tool import all as gt
from graph_tool import GraphView
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from graph_tool.topology import label_components
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import json
from sklearn.metrics import *


### ------------ RULE FITTING ----------- ###


def reorder_binary_decision_tree(old_regulator_order, regulators):
    n = len(regulators)
    new_order = []

    # Map the old order into the new order
    old_index_to_new = [regulators.index(i) for i in old_regulator_order]

    # Loop through the leaves for the new rule
    for leaf in range(2**n):
        # Get the binary for this leaf, ordered by the new order
        binary = ut.idx2binary(leaf, n)
        # Figure out what this binary would have been in the old order
        oldbinary = "".join([binary[i] for i in old_index_to_new])
        # What leaf was that in the old order?
        oldleaf = ut.state2idx(oldbinary)
        # Map that old leaf to the current, reordered leaf
        new_order.append(oldleaf)

    return new_order


# If A=f(B,C,D), this checks whether B being ON or OFF has an impact > threshold for any combination of C={ON/OFF} D={ON/OFF}
def detect_irrelevant_regulator(regulators, rule, threshold=0.1):
    n = len(regulators)
    max_difs = []
    tot_difs = []
    signed_tot_difs = []

    irrelevant = []
    for r, regulator in enumerate(regulators):
        print(f"...checking if {regulator} is irrelevant")
        max_dif = 0
        tot_dif = 0
        signed_tot_dif = 0
        leaves = ut.get_leaves_of_regulator(2**n, r)
        for i, j in zip(*leaves):
            dif = np.abs(rule[j] - rule[i])
            max_dif = max(dif, max_dif)
            tot_dif = tot_dif + dif
            signed_tot_dif = signed_tot_dif + rule[j] - rule[i]
        max_difs.append(max_dif)
        tot_difs.append(tot_dif)
        signed_tot_difs.append(signed_tot_dif)
        if max_dif < threshold:
            irrelevant.append(regulator)
    return (
        dict(zip(regulators, max_difs)),
        dict(zip(regulators, tot_difs)),
        dict(zip(regulators, signed_tot_difs)),
    )  ### added signed_tot_dif to output
    # return irrelevant


# the user can determine if they want to just save, just show, or save and show the plot
def get_rules_scvelo(
    data,
    data_t1,
    vertex_dict,
    plot=False,
    show_plot=False,
    save_plot=True,
    threshold=0.1,
    save_dir="rules",
    hlines=None,
):
    v_names = dict()
    # Invert the vertex_dict
    for vertex_name in list(vertex_dict):
        v_names[vertex_dict[vertex_name]] = vertex_name
    nodes = list(vertex_dict)
    rules = dict()
    regulators_dict = dict()
    strengths = pd.DataFrame(index=nodes, columns=nodes)
    signed_strengths = pd.DataFrame(index=nodes, columns=nodes)

    for gene in nodes:
        print(gene)
        # for each node of the network
        irrelevant = []
        n_irrelevant_new = 0
        regulators = [
            v_names[v]
            for v in vertex_dict[gene].in_neighbors()
            if not v_names[v] in irrelevant
        ]

        # Define a set of regulators as the in_neighbors of the node
        # This breaks when all regulators have been deemed irrelevant, or none have
        while True:
            n_irrelevant_old = n_irrelevant_new
            regulators_dict[gene] = regulators
            n = len(regulators)

            # We have to make sure we haven't stripped all the regulators as irrelevant
            if n > 0:
                # This becomes the eventual probabilistic rule. It has 2 rows
                # that describe prob(ON) and prob(OFF). At the end these rows
                # are normalized to sum to 1, such that the rule becomes
                # prob(ON) / (prob(ON) + prob(OFF)
                prob_01 = np.zeros((2, 2**n))

                # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
                heat = np.ones((data.shape[0], 2**n))

                for leaf in range(2**n):
                    if leaf % 50 == 0:
                        print(leaf)
                    binary = ut.idx2binary(leaf, len(regulators))
                    binary = [{"0": False, "1": True}[i] for i in binary]
                    # Binary becomes a list of lists of T and Fs to represent each column
                    for i, idx in enumerate(data.index):
                        # for each row in data column...
                        # grab that row (df) and the expression value for the current node (left side of rule plot) (val)
                        df = data.loc[idx]
                        val = np.float(data_t1.loc[idx, gene])
                        for col, on in enumerate(binary):
                            # for each regulator in each column in decision tree...
                            regulator = regulators[col]
                            # if that regulator is on in the decision tree, multiply the weight in the heatmap for that
                            # row of data and column of tree with a weight that = probability that that node is on in the data
                            # df(regulator) = expression value of regulator in data for that row
                            # multiply for each regulator (parent TF) in leaf
                            if on:
                                heat[i, leaf] *= np.float(df[regulator])
                            else:
                                heat[i, leaf] *= 1 - np.float(df[regulator])
                        # the probability for that leaf becomes the value of expression (val) times that square in the heatmap
                        # this loops over the rows in the heatmap and keeps multiplying in the weight * expression value
                        prob_01[0, leaf] += (
                            val * heat[i, leaf]
                        )  # Probabilitiy of being ON
                        prob_01[1, leaf] += (1 - val) * heat[i, leaf]

                # We weigh each column by adding in a sample with prob=50% and
                # a weight given by 1-max(weight). So leaves where no samples
                # had high weight will end up with a high weight of 0.5. For
                # instance, if the best sample has a weight 0.1 (crappy), the
                # rule will have a sample added with weight 0.9, and 50% prob.
                max_heat = 1 - np.max(heat, axis=0)
                for i in range(prob_01.shape[1]):
                    prob_01[0, i] += max_heat[i] * 0.5
                    prob_01[1, i] += max_heat[i] * 0.5

                # The rule is normalized so that prob(ON)+prob(OFF)=1
                rules[gene] = prob_01[0, :] / np.sum(prob_01, axis=0)
                (
                    max_regulator_relevance,
                    tot_regulator_relevance,
                    signed_tot_regulator_relevance,
                ) = detect_irrelevant_regulator(
                    regulators, rules[gene], threshold=threshold
                )

                old_regulator_order = [i for i in regulators]
                regulators = sorted(
                    regulators, key=lambda x: max_regulator_relevance[x], reverse=True
                )
                if max_regulator_relevance[regulators[-1]] < threshold:
                    irrelevant.append(regulators[-1])
                    old_regulator_order.remove(regulators[-1])
                    regulators.remove(regulators[-1])
                regulators = sorted(
                    regulators, key=lambda x: tot_regulator_relevance[x], reverse=True
                )
                regulators_dict[gene] = regulators
                # regulators = old_regulator_order
                # irrelevant += detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold)
                n_irrelevant_new = len(irrelevant)

            if len(regulators) == 0 and gene not in irrelevant:
                regulators = [
                    gene,
                ]
                regulators_dict[gene] = [
                    gene,
                ]
            elif n_irrelevant_old == n_irrelevant_new or len(regulators) == 0:
                break

        if len(regulators) > 0:
            importance_order = reorder_binary_decision_tree(
                old_regulator_order, regulators
            )
            heat = heat[:, importance_order]

            rules[gene] = rules[gene][importance_order]
            # rules[gene] = smooth_rule(rules[gene], regulators, tot_regulator_relevance, np.max(heat,axis=0))

            #strengths and signed_strengths should have child nodes as rows with columns as parent nodes
            strengths.loc[gene] = tot_regulator_relevance
            signed_strengths.loc[gene] = signed_tot_regulator_relevance

            if plot:
                plot_rule(
                    gene,
                    rules[gene],
                    regulators,
                    heat,
                    data,
                    save_dir=save_dir,
                    save=save_plot,
                    show_plot=show_plot,
                    hlines=hlines,
                )

    return rules, regulators_dict, strengths, signed_strengths


# data=dataframe with rows=samples, cols=genes
# nodes = list of nodes in network
# vertex_dict = dictionary mapping gene name to a vertex in a graph_tool Graph()
# v_names - A dictionary mapping vertex in graph to name
# plot = boolean - make the resulting plot
# threshold = float from 0.0 to 1.0, used as threshold for removing irrelevant regulators. 0 removes nothing. 1 removes all.
def get_rules(
    data, 
    vertex_dict, 
    plot=False, 
    threshold=0.1, 
    save_dir="rules", 
    save_plot = True,
    show_plot = False,
    hlines=None
):
    v_names = dict()
    for vertex_name in list(vertex_dict):
        v_names[vertex_dict[vertex_name]] = vertex_name  # invert the vertex_dict
    nodes = list(vertex_dict)
    rules = dict()
    regulators_dict = dict()
    strengths = pd.DataFrame(index=nodes, columns=nodes)
    signed_strengths = pd.DataFrame(index=nodes, columns=nodes)
    total_nodes = len(nodes)
    for xx, gene in enumerate(nodes):
        print("Fitting ", xx, "/", total_nodes, "rules")
        print(gene)
        # for each node of the network
        irrelevant = []
        n_irrelevant_new = 0
        regulators = [
            v_names[v]
            for v in vertex_dict[gene].in_neighbors()
            if not v_names[v] in irrelevant
        ]
        # define a set of regulators as the in_neighbors of the node
        while (
            True
        ):  # This breaks when all regulators have been deemed irrelevant, or none have
            n_irrelevant_old = n_irrelevant_new
            regulators_dict[gene] = regulators
            n = len(regulators)

            # we have to make sure we haven't stripped all the regulators as irrelevant
            if n > 0:
                # This becomes the eventual probabilistic rule. It has 2 rows
                # that describe prob(ON) and prob(OFF). At the end these rows
                # are normalized to sum to 1, such that the rule becomes
                # prob(ON) / (prob(ON) + prob(OFF)

                prob_01 = np.zeros((2, 2**n))

                # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
                heat = np.ones((data.shape[0], 2**n))

                for leaf in range(2**n):
                    if leaf % 100 == 0:
                        print(leaf)
                    binary = ut.idx2binary(leaf, len(regulators))
                    binary = [{"0": False, "1": True}[i] for i in binary]
                    # binary becomes a list of lists of T and Fs to represent each column
                    for i, idx in enumerate(data.index):
                        # for each row in data column...
                        # grab that row (df) and the expression value for the current node (left side of rule plot) (val)
                        df = data.loc[idx]
                        val = np.float(data.loc[idx, gene])
                        for col, on in enumerate(binary):
                            # for each regulator in each column in decision tree...
                            regulator = regulators[col]
                            # if that regulator is on in the decision tree, multiply the weight in the heatmap for that
                            # row of data and column of tree with a weight that = probability that that node is on in the data
                            # df(regulator) = expression value of regulator in data for that row
                            # multiply for each regulator (parent TF) in leaf
                            if on:
                                heat[i, leaf] *= np.float(df[regulator])
                            else:
                                heat[i, leaf] *= 1 - np.float(df[regulator])
                        # the probability for that leaf becomes the value of expression (val) times that square in the heatmap
                        # this loops over the rows in the heatmap and keeps multiplying in the weight * expression value
                        prob_01[0, leaf] += (
                            val * heat[i, leaf]
                        )  # Probabilitiy of being ON
                        prob_01[1, leaf] += (1 - val) * heat[i, leaf]

                # We weigh each column by adding in a sample with prob=50% and
                # a weight given by 1-max(weight). So leaves where no samples
                # had high weight will end up with a high weight of 0.5. For
                # instance, if the best sample has a weight 0.1 (crappy), the
                # rule will have a sample added with weight 0.9, and 50% prob.
                max_heat = 1 - np.max(heat, axis=0)
                for i in range(prob_01.shape[1]):

                    prob_01[0, i] += max_heat[i] * 0.5
                    prob_01[1, i] += max_heat[i] * 0.5

                # The rule is normalized so that prob(ON)+prob(OFF)=1
                rules[gene] = prob_01[0, :] / np.sum(prob_01, axis=0)
                (
                    max_regulator_relevance,
                    tot_regulator_relevance,
                    signed_tot_regulator_relevance,
                ) = detect_irrelevant_regulator(
                    regulators, rules[gene], threshold=threshold
                )

                old_regulator_order = [i for i in regulators]
                regulators = sorted(
                    regulators, key=lambda x: max_regulator_relevance[x], reverse=True
                )
                if max_regulator_relevance[regulators[-1]] < threshold:
                    irrelevant.append(regulators[-1])
                    old_regulator_order.remove(regulators[-1])
                    regulators.remove(regulators[-1])
                regulators = sorted(
                    regulators, key=lambda x: tot_regulator_relevance[x], reverse=True
                )
                regulators_dict[gene] = regulators
                # regulators = old_regulator_order
                # irrelevant += detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold)

                n_irrelevant_new = len(irrelevant)

            if len(regulators) == 0 and gene not in irrelevant:
                regulators = [
                    gene,
                ]
                regulators_dict[gene] = [
                    gene,
                ]
            elif n_irrelevant_old == n_irrelevant_new or len(regulators) == 0:
                break

        if len(regulators) > 0:
            importance_order = reorder_binary_decision_tree(
                old_regulator_order, regulators
            )
            heat = heat[:, importance_order]

            rules[gene] = rules[gene][importance_order]
            # rules[gene] = smooth_rule(rules[gene], regulators, tot_regulator_relevance, np.max(heat,axis=0))
            strengths.loc[gene] = tot_regulator_relevance
            signed_strengths.loc[gene] = signed_tot_regulator_relevance

            if plot:
                plot_rule(
                    gene,
                    rules[gene],
                    regulators,
                    heat,
                    data,
                    save_dir=save_dir,
                    save=save_plot,
                    show_plot=show_plot,
                    hlines=hlines,
                )


    return rules, regulators_dict, strengths, signed_strengths


def save_rules(rules, regulators_dict, fname="rules.txt", delimiter="|"):
    lines = []
    for k in regulators_dict.keys():
        rule = ",".join(["%f" % i for i in rules[k]])
        regulators = ",".join(regulators_dict[k])
        lines.append("%s|%s|%s" % (k, regulators, rule))

    outfile = open(fname, "w")
    outfile.write("\n".join(lines))
    outfile.close()


### ------------ FIT VALIDATION ------------ ###


def parent_heatmap(data, regulators_dict, gene):
    regulators = [i for i in regulators_dict[gene]]
    n = len(regulators)

    # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
    heat = np.ones((data.shape[0], 2**n))
    for leaf in range(2**n):
        binary = ut.idx2binary(leaf, len(regulators))
        # Binary becomes a list of lists of T and Fs to represent each column
        binary = [{"0": False, "1": True}[i] for i in binary]

        for i, idx in enumerate(data.index):
            # for each row in data column...
            # grab that row (df) and the expression value for the current node (left side of rule plot) (val)
            df = data.loc[idx]
            val = np.float(data.loc[idx, gene])
            for col, on in enumerate(binary):
                # for each regulator in each column in decision tree...
                regulator = regulators[col]
                # if that regulator is on in the decision tree, multiply the weight in the heatmap for that
                # row of data and column of tree with a weight that = probability that that node is on in the data
                # df(regulator) = expression value of regulator in data for that row
                # multiply for each regulator (parent TF) in leaf
                if on:
                    heat[i, leaf] *= np.float(df[regulator])
                else:
                    heat[i, leaf] *= 1 - np.float(df[regulator])

    regulator_order = [i for i in regulators]
    return heat, regulator_order


def roc(
    validation,
    node,
    n_thresholds=10,
    plot=False,
    show_plot=False,
    save=False,
    save_dir=None,
):
    tprs = []
    fprs = []
    for i in np.linspace(0, 1, n_thresholds, endpoint=False):
        p, r = calc_roc(validation, i)
        tprs.append(p)
        fprs.append(r)
    # area = auc(fprs, tprs)
    #### AUC function wasn't working... replace with np.trapz
    area = np.abs(np.trapz(x=fprs, y=tprs))
    if plot == True:
        plot_roc(
            fprs,
            tprs,
            area,
            node,
            save=save,
            save_dir=save_dir,
            show_plot=show_plot,
        )

    return tprs, fprs, area

#TODO: replace this function with sklearn.metrics.ROC_curve
def calc_roc(validation, threshold):
    # P: True positive over predicted condition positive (of the ones predicted positive, how many are actually
    # positive?)
    # R: True positive over all condition positive (of the actually positive, how many are predicted to be positive?)
    predicted = validation.loc[validation["predicted"] > threshold]
    actual = validation.loc[validation["actual"] > 0.5]
    predicted_neg = validation.loc[validation["predicted"] <= threshold]
    actual_neg = validation.loc[validation["actual"] <= 0.5]
    true_positive = len(set(actual.index).intersection(set(predicted.index)))
    false_positive = len(set(actual_neg.index).intersection(set(predicted.index)))
    true_negative = len(set(actual_neg.index).intersection(set(predicted_neg.index)))
    if len(actual.index.values) == 0 or len(actual_neg.index.values) == 0:
        return -1, -1
    else:
        # print((true_positive+true_negative)/(len(validation)))
        tpr = true_positive / len(actual.index)
        fpr = false_positive / len(actual_neg.index)
        return tpr, fpr


# this function is broken for some reason UGH
def auc(fpr, tpr):
    # fpr is x axis, tpr is y axis
    print("Calculating area under discrete ROC curve")
    area = 0
    i_old, j_old = 0, 0
    for c, i in enumerate(fpr):
        j = tpr[c]
        if c == 0:
            i_old = i
            j_old = j
        else:
            area += np.abs(i - i_old) * j_old + 0.5 * np.abs(i - i_old) * np.abs(
                j - j_old
            )
            i_old = i
            j_old = j
    return area


def save_auc_by_gene(area_all, nodes, save_dir):
    outfile = open(f"{save_dir}/aucs.csv", "w+")
    for n, a in enumerate(area_all):
        outfile.write(f"{nodes[n]},{a} \n")
    outfile.close()


# Function to run fit validation
# first runs plot_acuracy(_scvelo)() to get validation dataframe
# then it calculates roc giving the user the option to plot and save the dataframes and/or plots
# val_type = validation type; use the plot scvelo accuracy function or just the plot_accuracy function
# fname = optional name to append to default file save name
# returns: validation, tprs_all, fprs_all, and area_all
def fit_validation(
    data_test,
    nodes,
    regulators_dict,
    rules,
    data_test_t1 = None,
    save=False,
    save_dir=None,
    fname="",
    clusters=None,
    plot=True,
    plot_clusters=False,
    show_plots=False,
    save_df=False,
    n_thresholds=50,
    customPalette=sns.color_palette("Set2"),
):
    # create output file
    if fname != "":
        outfile = open(f"{save_dir}/tprs_fprs_{fname}.csv", "w+")
    else:
        outfile = open(f"{save_dir}/tprs_fprs.csv", "w+")

    ind = [x for x in np.linspace(0, 1, 50)]
    tpr_all = pd.DataFrame(index=ind)
    fpr_all = pd.DataFrame(index=ind)
    area_all = []

    outfile.write(f",,")
    for j in ind:
        outfile.write(str(j) + ",")
    outfile.write("\n")

    for node in nodes:
        # print(node)
        validation = plot_accuracy(
            data = data_test,
            node = node,
            regulators_dict = regulators_dict,
            rules = rules,
            data_t1= data_test_t1,
            plot_clusters=plot_clusters,
            clusters=clusters,
            save=save,
            save_dir=save_dir,
            show_plot=show_plots,
            save_df=save_df,
            customPalette=customPalette,
        )

        tprs, fprs, area = roc(
            validation,
            node,
            n_thresholds=n_thresholds,
            plot=plot,
            show_plot=show_plots,
            save=save,
            save_dir=save_dir,
        )
        tpr_all[node] = tprs
        fpr_all[node] = fprs
        outfile.write(f"{node},tprs,{tprs}\n")
        outfile.write(f"{node},fprs,{fprs}\n")
        area_all.append(area)

    outfile.close()
    return validation, tpr_all, fpr_all, area_all


# validation_dir = where the validation files were saved; to be read for each node
# get roc output from validation files if they exist already? could be helpful for testing so
# fit validation doesn't have to be run again
def roc_from_file(
    validation_dir,
    nodes,
    n_thresholds=50,
    plot=False,
    show_plots=False,
    save=False,
    save_dir=None,
    fname="",
):
    ind = [i for i in np.linspace(0, 1, 50)]
    tpr_all = pd.DataFrame(index=ind)
    fpr_all = pd.DataFrame(index=ind)
    area_all = []

    for node in nodes:
        validation = pd.read_csv(
            f"{validation_dir}/{node}_validation.csv", index_col=0, header=0
        )
        tprs, fprs, area = roc(
            validation,
            node,
            n_thresholds=n_thresholds,
            fname=fname,
            plot=plot,
            show_plot=show_plots,
            save=save,
            save_dir=save_dir,
        )
        tpr_all[node] = tprs
        fpr_all[node] = fprs
        area_all.append(area)
    return tpr_all, fpr_all, area_all


def get_sklearn_metrics(VAL_DIR, plot_cm = True, show = False, save = True, save_stats = True):
    files = glob.glob(f"{VAL_DIR}/accuracy_plots/*.csv")
    summary_stats = pd.DataFrame(columns = ['gene','accuracy','balanced_accuracy_score','f1','roc_auc_score', "precision",
                                                "recall", "explained_variance", 'max_error', 'r2','log-loss'])
    if len(files) == 0:
        print("You must first run tl.fit_validation() to generate the appropriate files.")
        return summary_stats
    else:

        for f in files:
            val_df = pd.read_csv(f, header = 0, index_col=0)
            val_df['actual_binary'] = [{True:1, False:0}[x] for x in val_df['actual']> 0.5]
            val_df['predicted_binary'] = [{True:1, False:0}[x] for x in val_df['predicted']> 0.5]
            gene = f.split("/")[-1].split("_")[0]
            #classification stats
            acc = accuracy_score(val_df['actual_binary'], val_df['predicted_binary'])
            bal_acc = balanced_accuracy_score(val_df['actual_binary'], val_df['predicted_binary'])
            f1 = f1_score(val_df['actual_binary'], val_df['predicted_binary'])
            prec = precision_score(val_df['actual_binary'], val_df['predicted_binary'])
            rec = recall_score(val_df['actual_binary'], val_df['predicted_binary'])

            #regression stats
            roc_auc = roc_auc_score(val_df['actual_binary'], val_df['predicted'])
            expl_var = explained_variance_score(val_df['actual'], val_df['predicted'])
            max_err = max_error(val_df['actual'], val_df['predicted'])
            r2 = r2_score(val_df['actual'], val_df['predicted'])
            ll = log_loss(val_df['actual_binary'], val_df['predicted'])

            summary_stats = summary_stats.append(pd.Series([gene, acc, bal_acc,f1,roc_auc,prec,rec,expl_var,max_err,r2,ll],
                                                        index=summary_stats.columns),ignore_index=True)
            if plot_cm:
                plt.figure()
                cm = confusion_matrix(val_df['actual_binary'], val_df['predicted_binary'])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap = "Blues")
                plt.title(gene)
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{VAL_DIR}/accuracy_plots/{gene}_confusion_matrix.pdf")
                    plt.close()
        
        summary_stats = summary_stats.sort_values('gene').reset_index().drop("index",axis = 1)

        if save_stats:
            summary_stats.to_csv(f"{VAL_DIR}/summary_stats.csv")
        return summary_stats

### ------------ ATTRACTORS ------------ ###

# tf_basin --> if -1, use average distance between clusters. otherwise use the same size basin for all phenotypes
def find_attractors(
    binarized_data,
    rules,
    nodes,
    regulators_dict,
    tf_basin,
    save_dir=None,
    threshold=0.5,
    on_nodes=[],
    off_nodes=[],
):
    att = dict()
    n = len(nodes)
    dist_dict = None

    if tf_basin < 0:
        dist_dict = ut.get_avg_min_distance(binarized_data, n)

    for k in binarized_data.keys():
        print(k)
        att[k] = []
        outfile = open(
            f"{save_dir}/attractors_{k}.txt", "w+"
        )  # if there are no  clusters, comment this out
        outfile.write("start-state,dist-to-start,attractor\n")
        start_states = list(binarized_data[k])
        cnt = 0

        for i in start_states:
            start_states = [i]
            # print(start_states)
            # print("Getting partial STG...")

            # getting entire stg is too costly, so just get stg out to 5 TF neighborhood
            if type(tf_basin) == int and tf_basin >= 0:
                if len(on_nodes) == 0 and len(off_nodes) == 0:
                    stg, edge_weights = ut.get_partial_stg(
                        start_states, rules, nodes, regulators_dict, tf_basin
                    )
                else:
                    stg, edge_weights = ut.get_partial_stg(
                        start_states,
                        rules,
                        nodes,
                        regulators_dict,
                        tf_basin,
                        on_nodes=on_nodes,
                        off_nodes=off_nodes,
                    )

            # elif type(tf_basin) == dict:
            elif dist_dict is not None:
                if len(on_nodes) == 0 and len(off_nodes) == 0:
                    stg, edge_weights = ut.get_partial_stg(
                        start_states, rules, nodes, regulators_dict, dist_dict[k]
                    )
                else:
                    stg, edge_weights = ut.get_partial_stg(
                        start_states,
                        rules,
                        nodes,
                        regulators_dict,
                        dist_dict[k],
                        on_nodes=on_nodes,
                        off_nodes=off_nodes,
                    )
            else:
                print(
                    "tf_basin needs to be an integer or < 0 to indicate dictionary of integers for each subtype."
                )

            # Directed stg pruned with threshold .5
            # n = number of nodes that can change (each TF gets chosen with equal probability)
            #     EXCEPT nodes that are held ON or OFF (no chance of changing)
            # each edge actually has a probability of being selected * chance of changing
            # print("Pruning STG edges...")
            d_stg = ut.prune_stg_edges(
                stg,
                edge_weights,
                n - len(on_nodes) - len(off_nodes),
                threshold=threshold,
            )

            # Each strongly connected component becomes a single node
            # components[2] tells if its an attractor
            # components[0] of v tells what components does v belong to
            # print('Condensing STG...')
            c_stg, c_vertex_dict, components = ut.condense(d_stg)

            vidx = stg.vertex_properties["idx"]
            # maps graph_tools made up index in partial stg to an index of state that means something to us
            # print("Checking for attractors...")
            for v in stg.vertices():
                # loop through every state in stg and if it's an attractor
                if components[2][components[0][v]]:
                    if v != 0:
                        outfile.write(
                            f"{start_states[0]},{ut.hamming_idx(vidx[v],start_states[0],n)}, {vidx[v]}\n"
                        )
                        # print(i, int(v), vidx[v])
                        att[k].append(vidx[v])
            cnt += 1
            if cnt % 100 == 0:
                print(
                    "...", np.round(cnt / (len(binarized_data[k])) * 100, 2), "% done"
                )

        outfile.close()

    for k in att.keys():
        att[k] = list(set(att[k]))
    return att


def write_attractor_dict(attractor_dict, nodes, outfile):
    for j in nodes:
        outfile.write(f",{j}")
    outfile.write("\n")
    for k in attractor_dict.keys():
        att = [ut.idx2binary(x, len(nodes)) for x in attractor_dict[k]]
        for i, a in zip(att, attractor_dict[k]):
            outfile.write(f"{k}")
            for c in i:
                outfile.write(f",{c}")
            outfile.write("\n")
    outfile.close()


### NOT WORKING WITH TEST DATA ****
# Arguments:
# phenotypes = list of phenotypes to filter by?**
# average_states = dictionary of average states **
# attractor_dict = attractor dictionary **
# avg_state_idx_dir = path to average_states_index_<phenotype>.txt files
# attractor_dir = pathe to attractors_<phenotype>.txt
# save_dir = directory to save output file
# nodes = nodes of tf network
def filter_attractors(
    phenotypes,  # average_states, attractor_dict,
    avg_state_idx_dir,
    attractor_dir,
    save_dir,
    nodes,
):
    # TEST
    average_states = {"Tumor1": [], "Tumor2": []}
    attractor_dict = {"Tumor1": [], "Tumor2": []}
    for phen in phenotypes:
        d = pd.read_csv(
            f"{avg_state_idx_dir}/average_states_idx_{phen}.txt", sep=",", header=0
        )
        average_states[f"{phen}"] = list(np.unique(d["average_state"]))
        d = pd.read_csv(f"{attractor_dir}/attractors_{phen}.txt", sep=",", header=0)
        attractor_dict[f"{phen}"] = list(np.unique(d["attractor"]))

        # Below code compares each attractor to average state for each subtype instead of
        # closest single binarized data point
        a = attractor_dict.copy()
        # attractor_dict = a.copy()
        for p in attractor_dict.keys():
            print(p)
            for q in attractor_dict.keys():
                print("q", q)
                if p == q:
                    continue
                n_same = list(
                    set(attractor_dict[p]).intersection(set(attractor_dict[q]))
                )
                print("n_same:", n_same, len(n_same))
                if len(n_same) != 0:
                    for x in n_same:
                        print("average_states:", average_states)
                        print("hamm p args:", x, average_states[p], len(nodes))
                        print("hamm q args:", x, average_states[q], len(nodes))
                        # Error passing np.unique list [0,1] in average states to idx2binary in hamming_idx function
                        p_dist = ut.hamming_idx(x, average_states[p], len(nodes))
                        q_dist = ut.hamming_idx(x, average_states[q], len(nodes))
                        # Code never gets here
                        print("p_dist:", p_dist, "q_dist", q_dist)
                        if p_dist < q_dist:
                            a[q].remove(x)
                        elif q_dist < p_dist:
                            a[p].remove(x)
                        else:
                            a[q].remove(x)
                            a[p].remove(x)
                            try:
                                a[f"{q}_{p}"].append(x)
                            except KeyError:
                                a[f"{q}_{p}"] = [x]
    print(a)
    attractor_dict = a
    print(attractor_dict)
    file = open(f"{save_dir}/attractors_filtered.txt", "w+")

    for j in nodes:
        file.write(f",{j}")
    file.write("\n")
    for k in attractor_dict.keys():
        print("attractor_dict key:", k)
        att = [ut.idx2binary(x, len(nodes)) for x in attractor_dict[k]]
        for i, a in zip(att, attractor_dict[k]):
            file.write(f"{k}")
            for c in i:
                file.write(f",{c}")
            file.write("\n")

    file.close()


### ------------ AVG STATES ------------ ###


def find_avg_states(binarized_data, nodes, save_dir):
    n = len(nodes)
    average_states = dict()

    for k in binarized_data.keys():
        ave = ut.average_state(binarized_data[k], n)
        state = ave.copy()
        state[state < 0.5] = 0
        state[state >= 0.5] = 1
        state = [int(i) for i in state]
        idx = ut.state2idx("".join(["%d" % i for i in state]))
        average_states[k] = idx

    file = open(f"{save_dir}/average_states.txt", "w+")
    ut.get_avg_state_index(nodes, average_states, file, save_dir=save_dir)
    return average_states


### ------------ PERTURBATIONS SUMMARY ------------ ###

def perturbations_summary(attractor_dict,perturbations_dir, show = False, save = True, plot_by_attractor = False, save_dir = "clustered_perturb_plots", save_full = True,
    significance = 'both', fname = "", ncols = 5, mean_threshold = -0.3):
    if plot_by_attractor:
        plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True, clustered = False)

    print("Plotting perturbation summary plots...")
    plot_destabilization_scores(attractor_dict, perturbations_dir, show = show, save = save, save_dir=save_dir)

    print("Testing significance of TF perturbations...")
    perturb_dict, full = ut.get_perturbation_dict(attractor_dict, perturbations_dir, significance = significance, save_full=False, mean_threshold=mean_threshold)
    perturb_gene_dict = ut.reverse_perturb_dictionary(perturb_dict)
    if save_full:
        ut.write_dict_of_dicts(perturb_gene_dict, 
            file = f"{perturbations_dir}/{save_dir}/perturbation_TF_dictionary{fname}.txt")


    full_sig = ut.get_ci_sig(full, group_cols=['cluster','gene','perturb'])

    if save_full:
        full_sig.to_csv(f"{perturbations_dir}/{save_dir}/perturbation_stats.csv")
    
    plot_perturb_gene_dictionary(perturb_gene_dict, full,perturbations_dir,show = False, save = True, ncols = ncols, fname = fname)

    return perturb_gene_dict, full, full_sig