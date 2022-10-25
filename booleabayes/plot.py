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


### ------------ ACCURACY PLOTS ------------ ###


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


# Need more info on the code to make sure it's plotting correctly.
def stability():
    ## Does user specify these attractor lists?
    Y_attractors = [14897871719]
    A2_attractors = [32136863904]
    N_attractors = [17045356415, 15703179135]
    A_attractors = [21349304528]
    uncl_attractors = [1446933]

    df = pd.DataFrame(
        columns=["radius", "folder", "median", "ci", "color", "phenotype"]
    )
    folders = listdir(op.join(di, "walks/"))
    # print(folders)
    colors = {
        "A": "g",
        "A2": "r",
        "N": "b",
        "Y": "orange",
        "uncl": "darkgray",
        "Null": "lightgrey",
    }

    ax = plt.subplot()
    NE_ave_change = [0] * 7
    NON_NE_ave_change = [0] * 7
    NEv2_ave_change = [0] * 7
    NEv1_ave_change = [0] * 7
    uncl_ave_change = [0] * 7

    countNEv2 = 0
    countNEv1 = 0
    countNE = 0
    countNONNE = 0
    countUNCL = 0
    # print(folders)
    for i, folder in enumerate(folders):
        earthmove = []
        if folder == ".DS_Store":
            continue
        phenotype_name = "Null"
        if int(folder) in A_attractors:
            phenotype_name = "A"
        elif int(folder) in A2_attractors:
            phenotype_name = "A2"
        elif int(folder) in N_attractors:
            phenotype_name = "N"
        elif int(folder) in Y_attractors:
            phenotype_name = "Y"
        elif int(folder) in uncl_attractors:
            phenotype_name = "uncl"

        medians = []
        datas = []
        cis = []
        # plt.figure()
        if phenotype_name != "Null":
            for radius in [1, 2, 3, 4, 5, 6, 7, 8]:
                lengths = pd.read_csv(
                    op.join(di, f"walks/{folder}/len_walks_{radius}.csv"), header=None
                )
                # sns.distplot(lengths[0], bins = 100,  label = f'Basin radius = {radius}',color=colors[radius])
                # ax = plt.subplot()
                # ax.boxplot(lengths[0], positions=[radius])
                medians.append(np.median(lengths[0]))
                cis.append(np.std(lengths[0]))
                df2 = pd.DataFrame(
                    {
                        "radius": [radius],
                        "folder": [folder],
                        "median": [np.median(lengths[0])],
                        "ci": [np.std(lengths[0])],
                        "color": [colors[phenotype_name]],
                        "phenotype": [phenotype_name],
                    }
                )
                df = df.append(df2)
                if radius == 1:
                    dfrm1 = lengths[0]
                else:
                    dfr = lengths[0]
                    earthmove.append(ss.wasserstein_distance(dfr, dfrm1))
                    dfrm1 = dfr
            print(phenotype_name, earthmove)

            sns.lineplot([2, 3, 4, 5, 6, 7, 8], earthmove, label=phenotype_name)
            if phenotype_name == "A2":
                NEv2_ave_change = [i + j for i, j in zip(NEv2_ave_change, earthmove)]
                countNEv2 += 1
            elif phenotype_name == "N":
                NEv1_ave_change = [i + j for i, j in zip(NEv1_ave_change, earthmove)]
                countNEv1 += 1
            elif phenotype_name == "A":
                NE_ave_change = [i + j for i, j in zip(NE_ave_change, earthmove)]
                countNE += 1
            elif phenotype_name == "Y":
                NON_NE_ave_change = [
                    i + j for i, j in zip(NON_NE_ave_change, earthmove)
                ]
                countNONNE += 1
            elif phenotype_name == "uncl":
                uncl_ave_change = [i + j for i, j in zip(uncl_ave_change, earthmove)]
                countUNCL += 1

    # print(df.head())
    ax.set_xlim([2, 8])
    ax.set_ylim([0, 30])
    plt.ylabel("Earth Mover's Distance")
    plt.xlabel("Radius of Basin")
    plt.title("Earth Mover's Distance Between Step Distributions for Different Radii")
    plt.show()

    random_change = [0] * 7
    countRANDOM = 0

    # add random background
    # colors = sns.color_palette("husl", 11)
    plt.figure()
    ax = plt.subplot()
    # fname = op.join(di, "walks/8440463/results.csv")
    for i, folder in enumerate(folders):
        earthmove = []
        if folder == ".DS_Store":
            continue
        phenotype_name = "Null"
        if int(folder) in A_attractors:
            phenotype_name = "A"
        elif int(folder) in A2_attractors:
            phenotype_name = "A2"
        elif int(folder) in N_attractors:
            phenotype_name = "N"
        elif int(folder) in Y_attractors:
            phenotype_name = "Y"
        elif int(folder) in uncl_attractors:
            phenotype_name = "uncl"
        medians = []
        datas = []
        cis = []
        # plt.figure()
        if phenotype_name == "Null":
            print("yes")
            for radius in [1, 2, 3, 4, 5, 6, 7, 8]:
                lengths = pd.read_csv(
                    op.join(di, f"walks/{folder}/len_walks_{radius}.csv"), header=None
                )
                # sns.distplot(lengths[0], bins = 100,  label = f'Basin radius = {radius}',color=colors[radius])
                # ax = plt.subplot()
                # ax.boxplot(lengths[0], positions=[radius])
                medians.append(np.median(lengths[0]))
                cis.append(np.std(lengths[0]))
                df2 = pd.DataFrame(
                    {
                        "radius": [radius],
                        "folder": [folder],
                        "median": [np.median(lengths[0])],
                        "color": [colors[phenotype_name]],
                        "ci": [np.std(lengths[0])],
                        "phenotype": [phenotype_name],
                    }
                )
                df = df.append(df2)
                if radius == 1:
                    dfrm1 = lengths[0]
                else:
                    dfr = lengths[0]
                    earthmove.append(ss.wasserstein_distance(dfr, dfrm1))
                    dfrm1 = dfr
            # sns.lineplot([2,3,4,5,6,7,8],  earthmove)
            random_change = [i + j for i, j in zip(random_change, earthmove)]
            countRANDOM += 1
    # ax.set_xlim([2,8])
    # ax.set_ylim([0,30])
    # plt.ylabel("Earth Mover's Distance")
    # plt.xlabel("Radius of Basin")
    # plt.title("Earth Mover's Distance Between Step Distributions for Random States")
    # plt.show()
    print("Colors", colors)
    NEv2_ave_change = [i / countNEv2 for i in NEv2_ave_change]
    NEv1_ave_change = [i / countNEv1 for i in NEv1_ave_change]
    NE_ave_change = [i / countNE for i in NE_ave_change]
    NON_NE_ave_change = [i / countNONNE for i in NON_NE_ave_change]
    uncl_ave_change = [i / countUNCL for i in uncl_ave_change]

    random_change = [i / countRANDOM for i in random_change]

    NEv2_ave_change_norm = [i / j for i, j in zip(NEv2_ave_change, random_change)]
    NEv1_ave_change_norm = [i / j for i, j in zip(NEv1_ave_change, random_change)]
    NE_ave_change_norm = [i / j for i, j in zip(NE_ave_change, random_change)]
    NON_NE_ave_change_norm = [i / j for i, j in zip(NON_NE_ave_change, random_change)]
    random_change_norm = [i / j for i, j in zip(random_change, random_change)]
    uncl_ave_change_norm = [i / j for i, j in zip(uncl_ave_change, random_change)]
    print(NEv2_ave_change, NEv2_ave_change_norm)
    plt.figure()

    for line, phenotype in zip(
        [
            NON_NE_ave_change,
            NEv2_ave_change,
            NEv1_ave_change,
            NE_ave_change,
            uncl_ave_change,
            random_change,
        ],
        ["Y", "A2", "N", "A", "uncl", "Null"],
    ):
        sns.lineplot([2, 3, 4, 5, 6, 7, 8], line, label=phenotype, palette=colors)
    plt.ylabel("Earth Mover's Distance")
    plt.xlabel("Radius of Basin")
    plt.title("Earth Mover's Distance Between Step Distributions for Different Radii")
    plt.show()

    plt.figure()
    for line, phenotype in zip(
        [
            NON_NE_ave_change_norm,
            NEv2_ave_change_norm,
            NEv1_ave_change_norm,
            NE_ave_change_norm,
            uncl_ave_change_norm,
            random_change_norm,
        ],
        ["Y", "A2", "N", "A", "uncl", "Null"],
    ):
        sns.lineplot([2, 3, 4, 5, 6, 7, 8], line, label=phenotype)
    plt.ylabel("Earth Mover's Distance")
    plt.xlabel("Radius of Basin")
    plt.title("Normalized Earth Mover's Distance Between Step Distributions")
    plt.show()

    # Plot stability plot for each phenotype (median # of steps to leave)
    plt.figure()
    print(df.head(20))
    ax = plt.subplot()
    ax = sns.lineplot(
        x="radius",
        y="median",
        err_style="band",
        hue="phenotype",
        palette=colors,
        data=df,
    )
    plt.ylim([0, 160])

    plt.show()
    df = df.reset_index()
    df_copy = df.copy()

    # Normalize each line by expected number of steps to leave (using control random starting states)
    for i, r in df.iterrows():
        norm_row = df.loc[df["phenotype"] == "Null"]
        norm_row = norm_row.loc[df["radius"] == df.loc[i]["radius"]]
        norm = np.median(norm_row["median"])
        df_copy.loc[i, "median"] = df.loc[i]["median"] / norm
    print(df_copy.head(20))
    plt.figure()
    ax = plt.subplot()
    ax = sns.lineplot(
        x="radius",
        y="median",
        err_style="band",
        hue="phenotype",
        palette=colors,
        data=df_copy,
    )
    plt.xlabel("Radius of Basin")
    # plt.ylim([0,16])
    plt.ylabel("Normalized Number of Steps to Leave Basin")
    plt.title("Normalized Stability of Each Phenotype")
    plt.show()


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
