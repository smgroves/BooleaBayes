from . import utils as ut

import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from scipy.stats import skewnorm, invgauss
import seaborn as sns
import scipy.stats as ss
import glob
import networkx as nx
from graph_tool import all as gt


### ------------ ACCURACY PLOTS ------------ ###

def plot_sklearn_summ_stats(summary_stats, VAL_DIR, fname = ""):
    df = pd.melt(summary_stats, id_vars='gene')
    df = df.astype({'variable':'string'})
    q1 = df.groupby(df.variable).quantile(0.25)['value']
    q3 = df.groupby(df.variable).quantile(0.75)['value']
    outlier_top_lim = q3 + 1.5 * (q3 - q1)
    outlier_bottom_lim = q1 - 1.5 * (q3 - q1)
    plt.figure()
    sns.boxplot(x="variable", y="value", data=pd.melt(summary_stats.drop('gene', axis = 1)))
    col_dict = {i:j for i,j in zip(summary_stats.drop('gene', axis = 1).columns,range(len(summary_stats.columns)-1))}
    for row in df.itertuples():
        variable = row.variable
        val = row.value
        if (val > outlier_top_lim[variable]) or (val < outlier_bottom_lim[variable]):
            print(val, row.gene)
            plt.annotate(s = row.gene, xy = (col_dict[variable]+0.1,val), fontsize = 8)
    plt.xticks(rotation = 45, ha = 'right')
    plt.xlabel("Model Metric")
    plt.ylabel("Score")
    plt.title("Metrics for BooleaBayes Rule Fitting across All TFs")
    plt.tight_layout()
    plt.savefig(f"{VAL_DIR}/summary_stats_boxplot{fname}.pdf")
    
def plot_sklearn_metrics(VAL_DIR, show = False, save = True):
    try:
        summary_stats = pd.read_csv(f"{VAL_DIR}/summary_stats.csv", header = 0, index_col=0)
    except FileNotFoundError:
        print("You must run tl.get_sklearn_metrics first.")
    if save:
        try:
            os.mkdir(f"{VAL_DIR}/summary_plots")
        except FileExistsError:
            pass
    metric_dict = {'accuracy':{"Best":'1', "Worst":'0'},
                   'balanced_accuracy_score':{"Best":'1', "Worst":'0'},
                   'f1':{"Best":'1', "Worst":'0'},
                   'roc_auc_score':{"Best":'1', "Worst":'0'},
                   "precision":{"Best":'1', "Worst":'0'},
                   "recall":{"Best":'1', "Worst":'0'},
                   "explained_variance":{"Best":'1', "Worst":'0'},
                   'max_error':{"Best":'0', "Worst":"High"},
                   'r2':{"Best":'1', "Worst":'0'},
                   'log-loss':{"Best":"Low", "Worst":"High"}}
    for c in sorted(list(set(summary_stats.columns).difference({'gene'}))):
        print(c)
        plt.figure(figsize = (20,8))
        my_order = summary_stats.sort_values(c)['gene'].values
        sns.barplot(data = summary_stats, x = 'gene', y = c, order = my_order)
        plt.xticks(rotation = 90, fontsize = 8)
        plt.ylabel(f"{c.capitalize()} (Best: {metric_dict[c]['Best']}, Worst: {metric_dict[c]['Worst']})")
        plt.title(c.capitalize())
        if show:
            plt.show()
        if save:
            plt.savefig(f"{VAL_DIR}/summary_plots/{c}.pdf")
            plt.close()

def plot_roc(
    fprs, tprs, area, node, save=False, save_dir=None,  show_plot=True
):
    fig = plt.figure()
    ax = plt.subplot()
    plt.plot(fprs, tprs, "-", marker="o")
    plt.title(node + " ROC Curve" + "\n AUC: " + str(round(area, 3)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    if save == True:
        if not os.path.exists(f"{save_dir}/accuracy_plots"):
            os.makedirs(f"{save_dir}/accuracy_plots")
        plt.savefig(f"{save_dir}/accuracy_plots/{node}_roc.pdf")

    elif show_plot == True:
        plt.show()
    plt.close()


def plot_aucs(VAL_DIR, save=False, show_plot=True):
    aucs = pd.read_csv(f'{VAL_DIR}/aucs.csv', header=None, index_col=0)
    plt.figure()
    plt.hist(aucs[1])
    if save == True:
        plt.savefig(f"{VAL_DIR}/aucs_plot.pdf")
    if show_plot == True:
        plt.show()
    plt.close()


def plot_validation_avgs(
    fpr_all, tpr_all, num_nodes, area_all, save=False, save_dir=None, show_plot=False
):
    plt.figure()
    ax = plt.subplot()
    plt.plot(fpr_all.sum(axis=1) / num_nodes, tpr_all.sum(axis=1) / num_nodes, "-o")
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(f"ROC Curve Data \n {np.sum(area_all) / num_nodes}")
    if save == True:
        plt.savefig(f"{save_dir}/ROC_AUC_average.pdf")
    if show_plot == True:
        plt.show()
    plt.close()


# def plot_accuracy_scvelo(
#     data,
#     data_t1,
#     node,
#     regulators_dict,
#     rules,
#     phenotypes=None,
#     clusters=None,
#     plot_clusters=False,
#     save=True,
#     save_dir=None,
#     fname="",
#     show_plot=False,
#     save_df=True,
#     customPalette=sns.color_palette("Set2"),
# ):
#     try:
#         os.mkdir(op.join(f"{save_dir}", f"accuracy_plots"))
#     except FileExistsError:
#         pass
#     try:
#         heat, order = parent_heatmap(data, regulators_dict, node)
#         # print("Order",order)
#         # print(f"Regulators_dict[{node}]", regulators_dict[node])
#         # importance_order = reorder_binary_decision_tree(order, regulators_dict[node])
#         rule = rules[node]
#         # dot product of weights of test sample and rule will give the predicted value for that sample for that TF
#         predicted = np.dot(heat, rule)
#         p = pd.DataFrame(predicted, columns=["predicted"], index=data.index)
#         # print(len(list(set(p.index).intersection(set(data_t1.index)))))
#         p["actual"] = data_t1[node]

#         if save_df == True:
#             p.to_csv(f"{save_dir}/accuracy_plots/{node}_validation.csv")

#         if plot_clusters == True:
#             plt.figure()
#             predicted = pd.DataFrame(predicted, index=data.index, columns=["predicted"])
#             sns.set_palette(sns.color_palette("Set2"))

#             for n, c in enumerate(sorted(list(set(clusters["class"])))):
#                 clines = data.loc[clusters.loc[clusters["class"] == c].index].index
#                 sns.scatterplot(
#                     x=data.loc[clines][node],
#                     y=predicted.loc[clines]["predicted"],
#                     label=c,
#                 )
#             plt.xlabel("Actual Normalized Expression")
#             plt.ylabel("Predicted Expression from Rule")
#             legend_elements = []

#             for i, j in enumerate(sorted(list(set(clusters["class"])))):
#                 legend_elements.append(Patch(facecolor=customPalette[i], label=j))

#             plt.legend(handles=legend_elements, loc="best")
#             plt.title(str(node))
#             if save == True:
#                 plt.savefig(f"{save_dir}/accuracy_plots/{node}_validation_plot.pdf")
#             elif show_plot == True:
#                 plt.show()
#             plt.close()
#         else:
#             plt.figure()
#             sns.regplot(x=data[node], y=predicted)
#             plt.xlabel("Actual Normalized Expression")
#             plt.ylabel("Predicted Expression from Rule")
#             plt.title(str(node))
#             plt.xlim(0, 1)
#             plt.ylim(0, 1)
#             if ut.r2(data[node], predicted) == 0:
#                 plt.title(str(node))
#             else:
#                 plt.title(
#                     str(node) + "\n" + str(round(ut.r2(data[node], predicted), 2))
#                 )

#             if save == True:
#                 plt.savefig(f"{save_dir}/accuracy_plots/{node}_validation_plot.pdf")
#             if show_plot == True:
#                 print(node)
#                 plt.show()
#             plt.close()
#         return p
#     except IndexError:
#         print(f"{node} had no parent nodes and cannot be accurately predicted.")


def parent_heatmap(data, regulators_dict, gene):
    regulators = [i for i in regulators_dict[gene]]
    n = len(regulators)

    # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
    heat = np.ones((data.shape[0], 2**n))
    for leaf in range(2**n):
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

    regulator_order = [i for i in regulators]
    return heat, regulator_order


def plot_accuracy(
    data,
    node,
    regulators_dict,
    rules,
    data_t1 = None,
    plot_clusters=False,
    clusters=None,
    save=True,
    save_dir=None,
    show_plot = False,
    save_df=True,
    customPalette=sns.color_palette("Set2"),

):
    try:
        os.mkdir(op.join(f"{save_dir}", 'accuracy_plots'))
    except FileExistsError:
        pass
    try:
        heat, order = parent_heatmap(data, regulators_dict, node)
        # print("Order", order)
        # print(f"Regulators_dict[{node}]", regulators_dict[node])
        # importance_order = reorder_binary_decision_tree(order, regulators_dict[g])
        rule = rules[node]
        # dot product of weights of test sample and rule will give the predicted value for that sample for that TF
        predicted = np.dot(heat, rule)
        p = pd.DataFrame(predicted, columns=["predicted"], index=data.index)
        if data_t1 is not None: 
            p["actual"] = data_t1[node] #replicates old plot_accuracy_scvelo function
        else:
            p["actual"] = data[node]

        if save_df == True:
            p.to_csv(f"{save_dir}/accuracy_plots/{node}_validation.csv")

        if plot_clusters == True:
            plt.figure()
            predicted = pd.DataFrame(predicted, index=data.index, columns=["predicted"])
            sns.set_palette(customPalette)

            for n, c in enumerate(sorted(list(set(clusters["class"])))):
                clines = data.loc[clusters.loc[clusters["class"] == c].index].index
                sns.scatterplot(
                    x=data.loc[clines][node],
                    y=predicted.loc[clines]["predicted"],
                    label=c,
                )
            plt.xlabel("Actual Normalized Expression")
            plt.ylabel("Predicted Expression from Rule")
            # plt.title(str(g)+"\n"+str(round(r2(data[g], predicted),2)))
            legend_elements = []

            # for i, j in enumerate(phenotypes):
            for i, j in enumerate(sorted(list(set(clusters["class"])))):
                legend_elements.append(Patch(facecolor=customPalette[i], label=j))

            plt.legend(handles=legend_elements, loc="best")
            plt.title(str(node))
            if save == True:
                plt.savefig(f"{save_dir}/accuracy_plots/{node}_validation_plot.pdf")
            if show_plot == True:
                plt.show()
            plt.close()
        else:
            plt.figure()
            sns.regplot(x=data[node], y=predicted)
            plt.xlabel("Actual Normalized Expression")
            plt.ylabel("Predicted Expression from Rule")
            plt.title(str(node))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            if ut.r2(data[node], predicted) == 0:
                plt.title(str(node))
            else:
                plt.title(str(node) + "\n" + str(round(ut.r2(data[node], predicted), 2)))
            if save == True:
                plt.savefig(f"{save_dir}/accuracy_plots/{node}_validation_plot.pdf")
            if show_plot == True:
                print(node)
                plt.show()
            plt.close()
        return p
    except IndexError:
        print(f"{node} had no parent nodes and cannot be accurately predicted.")


### ------------ ATTRACTOR PLOTS ------------ ###

## Work in progress
def plot_attractors(fname, save_dir="", sep=","):
    att = pd.read_table(f"{save_dir}/{fname}", sep=sep, header=0, index_col=0)
    att = att.transpose()
    plt.figure(figsize=(4, 8))
    sns.heatmap(
        att,
        cmap="binary",
        cbar=False,
        linecolor="w",
        linewidths=1,
        square=True,
        yticklabels=True,
    )
    plt.savefig(f"{save_dir}/{fname.split('.')[0]}.pdf")


### ------------ RULE PLOTS ------------ ###

# hlines=[11,10,11,18]
def plot_rule(
    gene,
    rule,
    regulators,
    sample_weights,
    data,
    save_dir="rules",
    save=False,
    show_plot=True,
    hlines=None,
):
    n = len(regulators)
    fig = plt.figure()
    # Plot layout:
    #                    .-.-------.
    #           / \      | |       |
    #          /\ /\     | |       |
    #      .-.-------.   |-|-------|
    #      | |       |   | |       |
    #      | |       |   | |       |
    #      | |       |   | |       |
    #      '-|-.-.-.-|   |-|-------|
    #        '-'-'-'-'   '-'-------'

    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 9, 1], width_ratios=[1, 8])
    # gs.update(hspace=0, wspace=0.03)
    gs.update(hspace=0, wspace=0)

    # Make the tree (plot a bunch of lines in branching pattern, starting from the bottom)
    ax = plt.subplot(gs[0, 1])

    bottom_nodes = range(2**n)
    for layer in range(n):
        top_nodes = []
        for leaves in [i * 2 for i in range(2 ** (n - layer - 1))]:

            top_nodes.append((bottom_nodes[leaves] + bottom_nodes[leaves + 1]) / 2.0)

        for i in range(len(top_nodes)):
            ax.plot(
                [bottom_nodes[2 * i], top_nodes[i]], [layer, layer + 1], "b--", lw=0.8
            )
            ax.plot(
                [bottom_nodes[2 * i + 1], top_nodes[i]],
                [layer, layer + 1],
                "r-",
                lw=0.8,
            )

        # ax.annotate(regulators[n-1-layer], ((2*top_nodes[i] + bottom_nodes[2*i+1])/3., layer+1))
        # Progress helps position the annotation along the branch - the lower in the tree,
        #   the farther along the branch the text is placed
        progress = min(0.9, (n - layer - 1) / 6.0)
        ax.annotate(
            " %s" % regulators[n - 1 - layer],
            (
                (1 - progress) * top_nodes[i] + progress * bottom_nodes[2 * i + 1],
                layer + 1 - progress,
            ),
            fontsize=8,
        )
        bottom_nodes = top_nodes

    ax.set_xlim(-0.5, 2**n - 0.5)
    ax.set_ylim(0, n)
    ax.set_axis_off()

    # Plot the rule (horizontal bar directly under tree (now under the matrix))
    ax = plt.subplot(gs[2, 1])
    pco = ax.pcolor(rule.reshape(1, rule.shape[0]), cmap="bwr", vmin=0, vmax=1)
    pco.set_edgecolor("face")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Inferred rule for %s" % gene)

    # Plot the sample weights in greyscale (big matrix)
    ax = plt.subplot(gs[1, 1])
    pco = ax.pcolor(sample_weights, cmap="Greys", vmin=0, vmax=1)
    pco.set_edgecolor("face")
    if hlines is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yline = 0
        for hline in hlines[:-1]:
            yline += hline
            plt.plot(xlim, [yline, yline], "k--", lw=0.5)
        if n < 8:
            for xline in range(1, 2**n):
                plt.plot([xline, xline], ylim, "k--", lw=0.1)
        else:
            for xline in range(2, 2**n, 2):
                plt.plot([xline, xline], ylim, "k--", lw=0.1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot the sample expression (vertical bar on left)
    ax = plt.subplot(gs[1, 0])
    pco = ax.pcolor(
        data[
            [
                gene,
            ]
        ],
        cmap="bwr",
    )
    pco.set_edgecolor("face")
    if hlines is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yline = 0
        for hline in hlines[:-1]:
            yline += hline
            plt.plot(xlim, [yline, yline], "k--", lw=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("%s expression" % gene)

    if save == True:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "%s.pdf" % gene))

    if show_plot == True:
        plt.show()

    # Close plot
    plt.cla()
    plt.clf()
    plt.close()


### ------------ RANDOM WALK PLOTS ------------ ###


def plot_histograms(n_steps_0, n_steps_1, expt_label, bins=20, fname=None, ax=None):
    """Plot histograms for random walks compared to perturbations. This function is an internal function called by rw.random_walks() and should be used with caution.

    :param n_steps_0: list of lengths of random walks without perturbation (from rw.random_walks())
    :type n_steps_0: list of integers
    :param n_steps_1: list of lengths of random walks with perturbation (from rw.random_walks())
    :type n_steps_1: list of integers
    :param expt_label: name of perturbation (in rw.random_walks(), this is assigned to "{node}_activate" or "{node}_knockdown"
    :type expt_label: string
    :param bins: number of bins for histogram plot, defaults to 20
    :type bins: int, optional
    :param fname: name of file to save histogram plot, defaults to None
    :type fname: string or None, optional
    :param ax: if plotting on an axis that already exists, defaults to None
    :type ax: Matplotlib Axes object, optional
    :return: list of control average, perturbation average, and destabilization score
    :rtype: list of floats
    """
    f, bins = np.histogram(n_steps_0 + n_steps_1, bins=bins)

    frequency_0, steps_0 = np.histogram(n_steps_0, bins=bins)
    density_0 = frequency_0 / (1.0 * np.sum(frequency_0))
    bin_width_0 = steps_0[1] - steps_0[0]
    gap_0 = bin_width_0 * 0.2

    frequency_1, steps_1 = np.histogram(n_steps_1, bins=bins)
    density_1 = frequency_1 / (1.0 * np.sum(frequency_1))
    bin_width_1 = steps_1[1] - steps_1[0]
    gap_1 = bin_width_1 * 0.2

    if ax is None:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

    ax.bar(
        steps_0[:-1] + gap_0 / 4.0,
        density_0,
        width=bin_width_0 - gap_0,
        color="#4499CC",
        alpha=0.4,
        label="Control",
    )
    ax.bar(
        steps_1[:-1] + gap_1 / 4.0,
        density_1,
        width=bin_width_1 - gap_1,
        color="#CC9944",
        alpha=0.4,
        label=expt_label,
    )

    ylim = ax.get_ylim()

    avg_0 = np.mean(n_steps_0)
    avg_1 = np.mean(n_steps_1)
    plt.axvline(avg_0, color="#4499CC", linestyle="dashed", label="Control Mean")
    plt.axvline(avg_1, color="#CC9944", linestyle="dashed", label="Perturbation Mean")

    ax.set_ylim(ylim)
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Frequency")
    ax.legend()

    if fname is not None:
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()
    return avg_0, avg_1, ((avg_1 - avg_0) / avg_0)


## plot barplot of destabilization scores for each TF for each attractor
## one plot per perturbation type (Activating vs inhibiting)
## or plot by cluster (boxplots)

def plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True, clustered = True,
                                act_kd_together = False, save_dir = "clustered_perturb_plots"):
    for k in attractor_dict.keys():
        print(k)
        if clustered:
            try:
                os.mkdir(f"{perturbations_dir}/{save_dir}")
            except FileExistsError:
                pass
            results = pd.DataFrame(columns = ['attr','gene','perturb','score'])
            for attr in attractor_dict[k]:
                tmp = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
                tmp.columns = ["attractor_dir","cluster","gene","perturb","score"]
                for i,r in tmp.iterrows():
                    results = results.append(pd.Series([attr, r['gene'],r['perturb'],r['score']],
                                                       index = ['attr','gene','perturb','score']), ignore_index=True)
            if act_kd_together:
                plt.figure()
                my_order = sorted(np.unique(results['gene']))
                plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')

                if len(attractor_dict[k]) == 1:
                    sns.barplot(data = results, x = 'gene', y = 'score', hue = 'perturb', order = my_order,
                                palette = {"activate":sns.color_palette("tab10")[0], "knockdown":sns.color_palette("tab10")[1]})
                else:
                    sns.boxplot(data = results, x = 'gene', y = 'score',hue = 'perturb', order = my_order,
                                palette = {"activate":sns.color_palette("tab10")[0], "knockdown":sns.color_palette("tab10")[1]})
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title(f"Destabilization by TF Perturbation for {k} Attractors \n {len(attractor_dict[k])} Attractors")
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{save_dir}/{k}_scores.pdf")
                    plt.close()
            else:
                results_act = results.loc[results["perturb"] == 'activate']
                plt.figure()
                # my_order = results_act.sort_values(by = 'score')['gene'].values
                my_order = results_act.groupby(by=["gene"]).median().sort_values(by = 'score').index.values
                plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')

                if len(attractor_dict[k]) == 1:
                    sns.barplot(data = results_act, x = 'gene', y = 'score', order = my_order)
                else:
                    sns.boxplot(data = results_act, x = 'gene', y = 'score', order = my_order)
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title(f"Destabilization by TF Activation for {k} Attractors \n {len(attractor_dict[k])} Attractors")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{save_dir}/{k}_activation_scores.pdf")
                    plt.close()

                results_kd = results.loc[results["perturb"] == 'knockdown']

                plt.figure()
                # my_order = results_act.sort_values(by = 'score')['gene'].values
                my_order = results_kd.groupby(by=["gene"]).median().sort_values(by = 'score').index.values
                plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')
                if len(attractor_dict[k]) == 1:
                    sns.barplot(data = results_kd, x = 'gene', y = 'score', order = my_order)
                else:
                    sns.boxplot(data = results_kd, x = 'gene', y = 'score', order = my_order)
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title(f"Destabilization by TF Knockdown for {k} Attractors \n {len(attractor_dict[k])} Attractors")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{save_dir}/{k}_knockdown_scores.pdf")
                    plt.close()

        else:
            for attr in attractor_dict[k]:
                results = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
                results.columns = ["attractor_dir","cluster","gene","perturb","score"]
                #activation plot
                results_act = results.loc[results["perturb"] == 'activate']
                colormat=list(np.where(results_act['score']>0, 'g','r'))
                results_act['color'] = colormat

                plt.figure()
                my_order = results_act.sort_values(by = 'score')['gene'].values
                sns.barplot(data = results_act, x = 'gene', y = 'score', order = my_order,
                            palette = ['r','g'], hue = 'color')
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title("Destabilization by TF Activation")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{attr}/activation_scores.pdf")
                    plt.close()

                #knockdown plot
                results_kd = results.loc[results["perturb"] == 'knockdown']
                colormat=list(np.where(results_kd['score']>0, 'g','r'))
                results_kd['color'] = colormat

                plt.figure()
                my_order = results_kd.sort_values(by = 'score')['gene'].values
                sns.barplot(data = results_kd, x = 'gene', y = 'score', order = my_order,
                            palette = ['r','g'], hue = 'color')
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title("Destabilization by TF Knockdown")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{attr}/knockdown_scores.pdf")
                    plt.close()

def plot_perturb_gene_dictionary(p_dict, full,perturbations_dir,show = False, save = True, ncols = 5, fname = "", 
                                 palette = {"activate":"green", "knockdown":"orange"}):
    ncols = ncols
    nrows = int(np.ceil(len(p_dict.keys())/ncols))
    # fig = plt.Figure(figsize = (8,8))
    fig, axs = plt.subplots(ncols = ncols, nrows= nrows, figsize=(20, 30))

    for x, k in enumerate(sorted(p_dict.keys())):
        print(k)
        #for each gene, for associated clusters that are destabilized, make a df of scores to be used for plotting
        plot_df = pd.DataFrame(columns = ["cluster","attr","gene","perturb","score"])
        for cluster in p_dict[k]["Regulators"]:
            tmp = full.loc[(full['cluster']==cluster)&(full['gene']==k)&(full["perturb"]=="knockdown")]
            for i,r in tmp.iterrows():
                plot_df = plot_df.append(r, ignore_index=True)
        for cluster in p_dict[k]["Destabilizers"]:
            tmp = full.loc[(full['cluster']==cluster)&(full['gene']==k)&(full["perturb"]=="activate")]
            for i,r in tmp.iterrows():
                plot_df = plot_df.append(r, ignore_index=True)

        # fig.add_subplot(ncols, nrows,x+1)
        my_order = plot_df.groupby(by=["cluster"]).median().sort_values(by = 'score').index.values
        col = int(np.floor(x/nrows))
        row = int(x%nrows)
        sns.barplot(data= plot_df, x = "cluster",y = "score", hue = "perturb", order = my_order,
                    ax = axs[row,col], palette = palette, dodge = False)
        axs[row,col].set_title(f"{k} Perturbations")
        axs[row,col].set_xticklabels(labels = my_order,rotation = 45, fontsize = 8, ha = 'right')
    plt.tight_layout()
    if save:
        plt.savefig(f"{perturbations_dir}/destabilizing_tfs{fname}.pdf")
    if show:
        plt.show()

def plot_stability(attractor_dict, walks_dir, palette = sns.color_palette("tab20"), rescaled = True,
                   show = False, save = True, err_style = "bars"):

    df = pd.DataFrame(
        columns=["cluster", "attr","radius", "mean", "median", "std"]
    )

    colormap = {i:c for i,c in zip(sorted(attractor_dict.keys()), palette)}
    # folders = glob.glob(f"{walks_dir}/[0-9]*")

    for k in sorted(attractor_dict.keys()):
        print(k)
        for attr in attractor_dict[k]:
            folders = glob.glob(f"{walks_dir}/{attr}/len_walks_[0-9]*")
            for f in folders:
                radius = int(f.split("_")[-1].split(".")[0])
                try:
                    lengths = pd.read_csv(f, header = None, index_col = None)
                except pd.errors.EmptyDataError: continue
                df = df.append(pd.Series([k,attr, radius, np.mean(lengths[0]), np.median(lengths[0]), np.std(lengths[0])],
                                         index=["cluster", "attr","radius","mean", "median", "std"]),
                                             ignore_index=True)

    ## add walk lengths from random control states to df
    if os.path.exists(f"{walks_dir}/random/"):
        colormap['random'] = 'lightgrey'
        random_starts = os.listdir(f"{walks_dir}/random/")
        for state in random_starts:
            folders = glob.glob(f"{walks_dir}/random/{state}/len_walks_[0-9]*")
            for f in folders:
                radius = int(f.split("_")[-1].split(".")[0])
                try:
                    lengths = pd.read_csv(f, header = None, index_col = None)
                except pd.errors.EmptyDataError: continue
                df = df.append(pd.Series(["random",state, radius, np.mean(lengths[0]), np.median(lengths[0]), np.std(lengths[0])],
                                         index=["cluster", "attr","radius","mean", "median", "std"]),
                               ignore_index=True)
        if rescaled:
            norm_df = df.copy()[['cluster', 'attr', 'radius', 'mean']]
            df_agg = df.groupby(['cluster','radius']).agg('mean')
            norm = df_agg.xs('random', level = 'cluster')
            for i,r in norm.iterrows():
                norm_df.loc[norm_df['radius']==i,'mean'] = norm_df.loc[norm_df['radius']==i,'mean']/r["mean"]
            norm_df = norm_df.sort_values(by = "cluster")
            plt.figure()
            sns.lineplot(x = 'radius',y = 'mean',err_style=err_style,hue = 'cluster', palette=colormap,
                         data = norm_df, markers = True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title = "Attractor Subtypes")

            plt.xticks(list(np.unique(norm_df['radius'])))
            plt.xlabel("Radius of Basin")
            plt.ylabel(f"Scaled Mean number of steps to leave basin \n (Fold-change from control mean)")
            plt.title("Scaled Stability of Attractors by Subtype")
            plt.tight_layout()
            if show:
                plt.show()
            if save:
                plt.savefig(f"{walks_dir}/scaled_stability_plot.pdf")
                plt.close()

    df = df.sort_values(by = "cluster")
    plt.figure()
    sns.lineplot(x = 'radius',y = 'mean',err_style=err_style,hue = 'cluster', palette=colormap,
                      data = df, markers = True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title = "Attractor Subtypes")

    plt.xticks(list(np.unique(df['radius'])))
    plt.xlabel("Radius of Basin")
    plt.ylabel("Mean number of steps to leave basin")
    plt.title("Stability of Attractors by Subtype")
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(f"{walks_dir}/stability_plot.pdf")
        plt.close()
    return df

# att_list = list of attractor states
# phenotypes = list of phenotypes
# phenotype_color = should be st by the function; user can pass in customPallete instead
# radius = radius used for the walk
# num_paths = max number of paths to plot? i think
# PCA graphs **
def pca_plot_paths(
    att_list,
    phenotypes,
    phenotype_color,
    radius,
    start_idx,
    num_paths=100,
    pca_path_reduce=False,
    walk_to_basin=False,
):
    pca = PCA(n_components=2)
    att_new = pca.fit_transform(att_list)
    data = pd.DataFrame(att_new, columns=["0", "1"])
    comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
    print(comp.T)
    data["color"] = phenotype_color

    plt.figure(figsize=(12, 10), dpi=600)
    plt.scatter(
        x=data["0"], y=data["1"], c=data["color"], s=100, edgecolors="k", zorder=4
    )
    legend_elements = []

    for i, j in enumerate(phenotypes):
        if "null" not in set(phenotype_color):
            if j == "null":
                continue
        legend_elements.append(Patch(facecolor=customPalette[i], label=j))

    plt.legend(handles=legend_elements, loc="best")

    ## Are these attractor lists supposed to be hardcoded?
    start_type = "null"
    if start_idx in NE_attractors:
        start_type = "NE"
    elif start_idx in ML_attractors:
        start_type = "NON-NE"
    elif start_idx in MLH_attractors:
        start_type = "NEv2"
    elif start_idx in NEH_attractors:
        start_type = "NEv1"

    data_walks = pd.DataFrame(columns=["0", "1"])
    att2_list = att_list.copy()
    if walk_to_basin == False:
        with open(
            op.join(
                dir_prefix,
                f"Network/walks/walk_to_basin/MYC_network/{start_idx}"
                f"/MYC_results_radius_{radius}.csv",
            ),
            "r",
        ) as file:
            line = file.readline()
            cnt = 1
            while line:
                if cnt == 1:
                    pass
                walk = line.strip()
                walk = walk.replace("[", "").replace("]", "").split(",")
                walk_states = [ut.idx2binary(int(i), n) for i in walk]
                walk_list = []
                for i in walk_states:
                    walk_list.append([int(j) for j in i])
                    att2_list.append([int(j) for j in i])
                walk_new = pca.transform(walk_list)
                data_walk = pd.DataFrame(walk_new, columns=["0", "1"])
                data_walks = data_walks.append(data_walk)
                data_walk["color"] = [
                    (len(data_walk.index) - i) / len(data_walk.index)
                    for i in data_walk.index
                ]
                plt.scatter(
                    x=data_walk["0"],
                    y=data_walk["1"],
                    c=data_walk["color"],
                    cmap="Blues",
                    s=20,
                    edgecolors="k",
                    zorder=3,
                )
                # sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=1, dashes=True, legend=False,
                #              alpha=0.1, zorder=2)
                cnt += 1
                line = file.readline()
                if cnt == num_paths:
                    break
        plt.title(
            f"PCA of {cnt} Walks from {start_idx} ({start_type})"
            f"\n Dimensionality Reduction on Attractors"
        )

    else:
        with open(
            op.join(
                dir_prefix,
                f"Network/walks/walk_to_basin/MYC_network/"
                f"{start_idx}/MYC_results_radius_{radius}.csv",
            ),
            "r",
        ) as file:
            line = file.readline()
            cnt = 1
            while line:
                if cnt == 1:
                    pass
                walk = line.strip()
                walk = walk.replace("[", "").replace("]", "").split(",")
                walk_states = [ut.idx2binary(int(i), n) for i in walk]
                walk_list = []
                for i in walk_states:
                    walk_list.append([int(j) for j in i])
                    att2_list.append([int(j) for j in i])
                walk_new = pca.transform(walk_list)
                data_walk = pd.DataFrame(walk_new, columns=["0", "1"])
                data_walks = data_walks.append(data_walk)
                data_walk["color"] = [
                    (len(data_walk.index) - i) / len(data_walk.index)
                    for i in data_walk.index
                ]
                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                sns.lineplot(
                    x=data_walk["0"],
                    y=data_walk["1"],
                    lw=1,
                    dashes=True,
                    legend=False,
                    alpha=0.1,
                    zorder=2,
                )
                cnt += 1
                line = file.readline()
                if cnt == num_paths:
                    break
        if radius == NE_attractors:
            basin_type = "NE"
        elif radius == ML_attractors:
            basin_type = "NON-NE"
        elif radius == MLH_attractors:
            basin_type = "NEv2"
        elif radius == NEH_attractors:
            basin_type = "NEv1"
        plt.title(
            f"PCA of {cnt} Walks from {start_idx} ({start_type}) to {basin_type}"
            f"\n Dimensionality Reduction on Attractors"
        )

    sns.kdeplot(
        data_walks["0"],
        data_walks["1"],
        shade=True,
        shade_lowest=False,
        zorder=1,
        n_levels=20,
        cbar=True,
    )
    plt.show()
    if pca_path_reduce == True:
        plt.figure(figsize=(12, 10), dpi=600)
        att2_new = pca.fit_transform(att2_list)
        data2 = pd.DataFrame(att2_new, columns=["0", "1"])
        comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
        print(comp.T)
        plt.scatter(
            x=data2.iloc[0:10]["0"],
            y=data2.iloc[
                0:10,
            ]["1"],
            c=data["color"],
            s=100,
            edgecolors="k",
            zorder=4,
        )
        legend_elements = []

        for i, j in enumerate(phenotypes):
            if "null" not in set(phenotype_color):
                if j == "null":
                    continue
            legend_elements.append(Patch(facecolor=customPalette[i], label=j))

        plt.legend(handles=legend_elements, loc="best")
        plt.title(
            f"PCA of {cnt} Walks from {start_idx} ({start_type}) \n Dimensionality Reduction on All States in Paths"
        )

        sns.kdeplot(
            data2.iloc[10:]["0"],
            data2.iloc[10:]["1"],
            shade=True,
            shade_lowest=False,
            zorder=1,
            n_levels=20,
            cbar=True,
        )
        plt.show()


# Not sure if this needs to be included
def check_middle_stop(start_idx, basin, check_stops, radius=2):
    with open(
        op.join(
            dir_prefix,
            f"Network/walks/walk_to_basin/MYC_network/{start_idx}/MYC_results_radius_{basin}.csv",
        ),
        "r",
    ) as file:
        line = file.readline()
        cnt = 1
        stopped_NE = None
        stopped_NEH = None
        stopped_MLH = None
        stopped_ML = None
        for k in check_stops:
            if k in NE_attractors:
                stopped_NE = 0
            elif k in ML_attractors:
                stopped_ML = 0
            elif k in MLH_attractors:
                stopped_MLH = 0
            elif k in NEH_attractors:
                stopped_NEH = 0
        while line:
            cnt += 1
            line = file.readline()
            if cnt == 1:
                pass
            walk = line.strip()
            walk = walk.replace("[", "").replace("]", "").split(",")
            for j in check_stops:
                for i in walk:
                    if i == "":
                        continue
                    dist = ut.hamming_idx(j, int(i), n)
                    if dist < radius:
                        if j in NE_attractors:
                            stopped_NE += 1
                        elif j in ML_attractors:
                            stopped_ML += 1
                        elif j in MLH_attractors:
                            stopped_MLH += 1
                        elif j in NEH_attractors:
                            stopped_NEH += 1
                        break
    return stopped_NE, stopped_NEH, stopped_MLH, stopped_ML


### ------------ NETWORK PLOTS ------------- ###
def draw_grn(G, gene2vertex, rules, regulators_dict, fname, gene2group=None, gene2color=None, type = "", B_min = 5,
             save_edge_weights = True, edge_weights_fname = "edge_weights.csv"):
    """Plot the network and optionally save to pdf

    :param G:  Graph to plot, such as the graph outputted by load.load_network()
    :type G: graph-tool graph object
    :param gene2vertex: Vertex dictionary assigning node names to vertices in network, such as the vertex_dict outputted by load.load_network()
    :type gene2vertex: dict
    :param rules: _description_
    :type rules: rules describing the Boolean tree for each node in network, such as the rules outputted by tl.get_rules()
    :param regulators_dict: dictionary of the form {node_A:[parents_of_A], node_B:[parents_of_B],...}
    :type regulators_dict: dict
    :param fname: File path to save network plot
    :type fname: str
    :param gene2group: Dictionary of the form {node[str]:group[int]} used to group nodes by color, defaults to None
    :type gene2group: dict, optional
    :param gene2color: Dictionary of the form {node[str]:color[vector<float>]} used to group nodes by color, defaults to None
    :type gene2color: dict, optional
    :param type: network plot type, either "circle" or "", defaults to ""
    :type type: str, optional
    :param B_min: if type =='circle', B_min is argument of minimize_nested_blockmodel_dl for arranging nodes, defaults to 5
    :type B_min: int, optional
    :param save_edge_weights: Whether to save the edge weights DataFrame to a csv, defaults to True
    :type save_edge_weights: bool, optional
    :param edge_weights_fname: If save_edge_weights == True, file name for saving edge weights DataFrame, defaults to "edge_weights.csv"
    :type edge_weights_fname: str, optional
    :return: Graph with additional edge properties, edge_weight_df, edge_binary_df
    :rtype: [graph-tool graph object, Pandas DataFrame, Pandas DataFrame]
    """
    vertex2gene = G.vertex_properties['name']

    vertex_group = None
    if gene2group is not None:
        vertex_group = G.new_vertex_property("int")
        for gene in gene2group.keys():
            vertex_group[gene2vertex[gene]] = gene2group[gene]

    vertex_colors = [0.4, 0.2, 0.4, 1]
    if gene2color is not None:
        vertex_colors = G.new_vertex_property("vector<float>")
        for gene in gene2color.keys():
            vertex_colors[gene2vertex[gene]] = gene2color[gene]

    edge_weight_df = pd.DataFrame(index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys()))
    edge_binary_df = pd.DataFrame(index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys()))

    edge_markers = G.new_edge_property("string")
    edge_weights = G.new_edge_property("float")
    edge_colors = G.new_edge_property("vector<float>")
    for edge in G.edges():
        edge_colors[edge] = [0., 0., 0., 0.3]
        edge_markers[edge] = "arrow"
        edge_weights[edge] = 0.2

    for edge in G.edges():
        vs, vt = edge.source(), edge.target()
        source = vertex2gene[vs]
        target = vertex2gene[vt]
        regulators = regulators_dict[target]
        if source in regulators:
            i = regulators.index(source)
            n = 2 ** len(regulators)

            rule = rules[target]
            off_leaves, on_leaves = ut.get_leaves_of_regulator(n, i)
            if rule[off_leaves].mean() < rule[on_leaves].mean():  # The regulator is an activator
                edge_colors[edge] = [0., 0.3, 0., 0.8]
                edge_binary_df.loc[target,source] = 1
            else:
                edge_markers[edge] = "bar"
                edge_colors[edge] = [0.88, 0., 0., 0.5]
                edge_binary_df.loc[target,source] = -1

            # note: not sure why I added 0.2 to each edge weight.. skewing act larger and inh smaller?
            edge_weights[edge] = rule[on_leaves].mean() - rule[off_leaves].mean() # + 0.2
            edge_weight_df.loc[target, source] = rule[on_leaves].mean() - rule[off_leaves].mean()
    G.edge_properties["edge_weights"] = edge_weights
    if save_edge_weights:
        edge_weight_df.to_csv(edge_weights_fname)
    pos = gt.sfdp_layout(G, groups=vertex_group,mu = 1, eweight=edge_weights, max_iter=1000)
    # pos = gt.arf_layout(G, max_iter=100, dt=1e-4)
    eprops = {"color": edge_colors, "pen_width": 2, "marker_size": 15, "end_marker": edge_markers}
    vprops = {"text": vertex2gene, "shape": "circle", "size": 20, "pen_width": 1, 'fill_color': vertex_colors}
    if type == 'circle':
        state = gt.minimize_nested_blockmodel_dl(G, B_min = B_min)
        state.draw(vprops=vprops, eprops=eprops)  # mplfig=ax[0,1])
    else:
        gt.graph_draw(G, pos=pos, output=fname, vprops=vprops, eprops=eprops, output_size=(1000, 1000))
    return G, edge_weight_df, edge_binary_df

def plot_subgraph(keep_nodes, network_file, nodes, edge_weights, keep_parents = True, keep_children = True,
                  save_dir = "", arrows = "straight", show = False, save = True, off_node_arrows_gray = True, weight = 3):
    """Plot a subgraph of the network centered on given nodes

    :param keep_nodes: list of nodes to keep centered in network; parent and child nodes will also be kept
    :type keep_nodes: list of str
    :param network_file: file path for full network
    :type network_file: str
    :param nodes: list of all nodes in network
    :type nodes: list of str
    :param edge_weights: DataFrame of edge weights to color and weight edges in subgraph with rows = child nodes and cols = parent nodes for each interaction. This is generated by draw_grn or can be replaced by signed_strengths from fitted rules.
    :type edge_weights: Pandas DataFrame
    :param keep_parents: Whether to keep parent nodes of keep_nodes, defaults to True
    :type keep_parents: bool, optional
    :param keep_children: Whether to keep child nodes of keep_nodes, defaults to True
    :type keep_children: bool, optional
    :param save_dir: path to save plot of subgraph, defaults to ""
    :type save_dir: str, optional
    :param arrows: Option for arrow style in {'curved', 'straight'}, defaults to "straight"
    :type arrows: str, optional
    :param show: Whether to show plots, defaults to False
    :type show: bool, optional
    :param save: Whether to save plots to file {save_dir}/subnetwork{keep_nodes}_{arrows}.pdf, defaults to True
    :type save: bool, optional
    :param off_node_arrows_gray: Whether the arrows between nodes that do not include the central nodes should be colored (by edge type) or grey, defaults to True
    :type off_node_arrows_gray: bool, optional
    :param weight: Weight multiplier for edges in plot, defaults to 3
    :type weight: int, optional
    """

    edge_df = pd.read_csv(network_file, header = None)##network file

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)


    for i,r in edge_df.iterrows():
        G.add_edge(r[0],r[1], weight = edge_weights.loc[r[1],r[0]])

    total_keep = keep_nodes.copy()
    attrs = {}

    if keep_children:
        for n in keep_nodes:
            for successor in G.successors(n):
                total_keep.append(successor)
                attrs[successor] = {"subset":1}
    if keep_parents:
        for n in keep_nodes:
            for predecessor in G.predecessors(n):
                total_keep.append(predecessor)
                attrs[predecessor] = {"subset":3}
    for node in keep_nodes:
        attrs[node] = {"subset":2}

    SG = G.subgraph(nodes = total_keep)
    nx.set_node_attributes(SG, attrs)
    edges = SG.edges()
    weights = [weight*np.abs(SG[u][v]['weight']) for u,v in edges]
    color = []
    for u,v in edges:
        if u in keep_nodes or v in keep_nodes:
            if SG[u][v]['weight'] < 0:
                color.append('red')
            else:
                color.append('green')
        else:
            if off_node_arrows_gray:
                color.append('lightgray')
            else:
                if SG[u][v]['weight'] < 0:
                    color.append('red')
                else:
                    color.append('green')

    print(nx.get_node_attributes(SG, name = 'subset'))
    if arrows == "straight":
        nx.draw_networkx(SG,pos=nx.multipartite_layout(SG,align = 'horizontal'),node_size = 500, font_size = 6,
                     with_labels=True, arrows = True,width = weights, edge_color = color)#,
    elif arrows == "curved":
        nx.draw_networkx(SG,pos=nx.multipartite_layout(SG,align = 'horizontal'),node_size = 500, font_size = 6,
                         with_labels=True, arrows = True,width = weights, edge_color = color,
                        connectionstyle="arc3,rad=0.4")
    else:
        print("arrows must be one of {'curved','straight'}")
    name_plot = ""
    for name in keep_nodes:
        name_plot = name_plot + f"_{name}"
    if show:
        plt.show()
    if save:
        plt.savefig(f"{save_dir}/subnetwork{name_plot}_{arrows}.pdf")
