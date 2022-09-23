from . import utils as ut

import os
import numpy as np


def binarize_data(
    data,
    phenotype_labels=None,
    threshold=0.5,
    save=False,
    save_dir=None,
    fname="binarized_data",
):
    if phenotype_labels is None:
        binaries = set()
    else:
        binaries = dict()
        for c in phenotype_labels["class"].unique():
            binaries[c] = set()

    f = np.vectorize(lambda x: "0" if x < threshold else "1")

    for sample in data.index:
        b = ut.state2idx("".join(f(data.loc[sample])))

        if phenotype_labels is None:
            binaries.add(b)
        else:
            binaries[phenotype_labels.loc[sample, "class"]].add(b)

    if save == True:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + os.sep + fname + ".csv", "w+") as outfile:
            for k in binaries.keys():
                outfile.write(f"{k}: {binaries[k]}\n")

    return binaries


# fname should replace the dir_prefix and barcode thing, or just pass the barcode and root directory
def filter_attractors(dir_prefix, brcd, nodes):
    average_states = {
        "SCLC-A": [],
        "SCLC-A2": [],
        "SCLC-Y": [],
        "SCLC-P": [],
        "SCLC-N": [],
        "SCLC-uncl": [],
    }
    attractor_dict = {
        "SCLC-A": [],
        "SCLC-A2": [],
        "SCLC-Y": [],
        "SCLC-P": [],
        "SCLC-N": [],
        "SCLC-uncl": [],
    }

    for phen in ["SCLC-A", "SCLC-N", "SCLC-A2", "SCLC-P", "SCLC-Y"]:
        d = pd.read_csv(
            op.join(dir_prefix, f"{brcd}/average_states_idx_{phen}.txt"),
            sep=",",
            header=0,
        )
        average_states[f"{phen}"] = list(np.unique(d["average_state"]))
        d = pd.read_csv(
            op.join(dir_prefix, f"{brcd}/attractors_{phen}.txt"), sep=",", header=0
        )
        attractor_dict[f"{phen}"] = list(np.unique(d["attractor"]))

        # Below code compares each attractor to average state for each subtype
        # instead of closest single binarized data point
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
                if len(n_same) != 0:
                    for x in n_same:
                        p_dist = ut.hamming_idx(x, average_states[p], len(nodes))
                        q_dist = ut.hamming_idx(x, average_states[q], len(nodes))
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

    attractor_dict = a
    print(attractor_dict)
    file = open(op.join(dir_prefix, f"{brcd}/attractors_filtered.txt"), "w+")
    # plot attractors
    for j in nodes:
        file.write(f",{j}")
    file.write("\n")
    for k in attractor_dict.keys():
        att = [ut.idx2binary(x, len(nodes)) for x in attractor_dict[k]]
        for i, a in zip(att, attractor_dict[k]):
            file.write(f"{k}")
            for c in i:
                file.write(f",{c}")
            file.write("\n")

    file.close()
    # graph_utils.plot_attractors(op.join(dir_prefix, f'{brcd}/attractors_filtered.txt'))
