from pythontools import enrichr
import networkx as nx
import time


def prune(G_orig, prune_sources = True, prune_sinks = True):
    G = G_orig.copy()
    n = len(G.nodes())
    nold = n + 1

    while (n != nold):
        nold = n
        for tf in list(G.nodes()):
            if prune_sources == True:
                if G.in_degree(tf) == 0:  G.remove_node(tf)
            if prune_sinks == True:
                if G.out_degree(tf) == 0: G.remove_node(tf)
            else:
                if G.in_degree(tf) == 0 and G.out_degree(tf) == 0:G.remove_node(tf)
        n = len(G.nodes())
    return G


def prune_info(G_orig, prune_self_loops=True):
    G = G_orig.copy()
    for tf in list(G.nodes()):
        edges = G.adj[tf]
        for target in list(edges.keys()):
            if tf == target and prune_self_loops:
                G.remove_edge(tf, target)
                continue
            if 'db' not in edges[target]:
                G.remove_edge(tf, target)
            elif len(edges[target]['db']) < 2:
                G.remove_edge(tf, target)
    return prune(G)


def prune_to_chea(G_orig, prune_self_loops=True):
    G = G_orig.copy()
    for tf in list(G.nodes()):
        edges = G.adj[tf]
        for target in list(edges.keys()):
            if tf == target and prune_self_loops:
                G.remove_edge(tf, target)
                continue
            if 'db' in edges[target]:
                if not True in ['ChEA' in i for i in edges[target]['db']]: G.remove_edge(tf, target)
    #                if len(edges[target]['db']) < 2: G.remove_edge(tf, target)
    return prune(G)

######################
# Adam's network
######################
# tfs = ['AR','ATF3','CREB3L1','CSRNP1','EBF1','EBF4','EGR1','EGR2','EGR4','EHF','ESRRG','ETV1','ETV4','ETV5',
# 'FOS','FOSB','FOXC1','FOXD1','FOXI1','FOSL1','HLF','HMGA2',"HOXA7",'HOXB5','HOXD8','KLF4','KLF7','MACC1','MAF',
# 'MAFB','MEIS2','MESP1','NHLH2','NR2E3','NR4A1','NR4A3','ONECUT2','ONECUT3','OVOL1','PAX5','PITX3','PLAG1''POU3F2',
# 'RUNX1','RUNX2','SIX2','SMAD2','SMAD4','SOX2','SOX9','SPIB','TBX1','TBX20','TOX2','WT1','ZNF300','ZNF750']

# tfs = ['SP6','SMAD3','SIX5','RFX3','ZMAT4','PBX3','MBNL3','CDX2','TSHZ2','BEX1','TAP1','TRIM47','GFI1B','CITED1','RUNX1T1','ZBTB7C','BCL3',
# 'BCL2','TRIM21','FBN1','VAV1','SMYD3','IRF1','IRF6','SERTAD1','NOTCH1','RORC','FOSL1','SCG2','LSR','SP110','ELK3','TRIM5','RGS7','UACA','ZIC2',
# 'HOXA3','HOXA1','PRRX2','DTX1','SP100','ZNF217','DLX5','DLX6','WWTR1','SFN','ZNF185','TGFB1','ETS2','JUP','NOTCH2','CREB3L1','ZC3H6','HMGA2','TEAD4',
# 'HEXIM1','ID4','GLI1','NR0B2','OLIG2','RELB','PRX','ARRB1','KLHL5','ZYX','NOTCH3','NFKB2','OVOL2','DSP','SQSTM1','ISL1','MST1R','GRIP1','ZNF423',
# 'ZNF165','EGLN3','POU2F3','MYB','FOXG1','SOX11','SOX13','HES1','HES2','HES6','TOX3','ASCL1','ZFP36','BATF2','ZFPM2','FOXA1','FOXA2','INSM1','NEUROD1',
# 'TFCP2L1','ZEB1','ZEB2','TRIM56','ZSCAN31','EYA2','RBP1','PROX1','CELF3','HOXB2','KLF6','KLF2','REST','RARG','EPAS1','TOX','BTBD11','NPM2','HOXC11',
# 'HOXC13','NAB2','MACC1','FHL1','ST18','ZNF521','GRHL2','EHF','CSRP1','RCAN1','YAP1','NHLH2','NHLH1','SOX9','SOX1','CERS4','NCALD','STAT6','MSC',
# 'ETV4','ETV6','ELF1','ELF3']
# #
# tfs = ['SP100', 'FOSL1', 'HES1', 'NFKBIZ', 'RELB', 'EPAS1', 'BCL3', 'REST', 'SP110', 'NFKB2', 'TEAD2', 'HMG20B', 'SIX5',
#        'RARG', 'TEAD4', 'ZNF217', 'SP140L', 'SOX18', 'HOXC13', 'STAT6', 'ETV4', 'KLF2', 'MITF', 'NR0B2', 'ASCL1', 'ZBTB7C', 'ELF3',
#        'RORC', 'FOXA2', 'ETS2','TOX3', 'XBP1', 'ST18', 'FOXA1', 'OVOL2', 'ZNF664', 'TBX10', 'PROX1', 'ETV6', 'CEBPD', 'TFCP2L1', 'FOXJ3',
#        'ZNF407', 'ZNF511','ZNF396', 'RBPJ', 'ZSCAN31', 'HOXB5', 'ZNF3', 'TSHZ2', 'ZBTB16', 'ZNF10', 'FLI1', 'GATA4', 'NR0B1', 'NHLH1',
#        'NEUROD6', 'ZNF581', 'TCF15', 'LYAR', 'ISL2', 'OLIG2', 'NEUROD1', 'INSM1', 'PAX5', 'SP6', 'MYT1', 'HES6', 'ZNF24', 'ISL1', 'ZNF397',
#        'SOX11', 'ZNF253', 'SMAD4', 'RBP1', 'ONECUT2', 'ZNF711', 'DLX5', 'GRIP1', 'ZNF157', 'ZNF713', 'ZNF136', 'FOXN4', 'PATZ1', 'ZNF491',
#        'ZBTB21', 'KLF12', 'ZNF501', 'ZNF785', 'CXXC1', 'ZNF324', 'ZNF764', 'ZBTB18', 'KAT8', 'ZNF334', 'POU4F1', 'ZNF250', 'ZNF132',
#        'SALL2', 'DLX6', 'MBD1','SOX1', 'ZFP3', 'ZNF543', 'POU2F1', 'NONO', 'SMAD9', 'ZKSCAN2', 'TCF12', 'VEZF1', 'TOX', 'BHLHE22', 'MTA1',
#        'TCF3', 'SCRT2', 'RFX7','NHLH2', 'SCRT1', 'RCOR2', 'PURG', 'TBPL1', 'TCF4', 'EBF1', 'ZNF749', 'NEUROD2', 'ZNF423', 'BACH2', 'GLI1',
#        'ZFP64','NKX2-1', 'MYC', 'YAP1', 'POU2F3', 'MYCL', 'MYCN', 'ASCL2', 'AVIL', 'CHAT', 'GFI1B']

######################
# MYC network
######################

# tfs = ['MYC', 'MYCN', 'MYCL', 'MAX', 'MGA', 'MXD1', 'MXI1', 'MXD3', 'MXD4', 'MNT', 'MLX', 'MLXIP', 'MLXIPL', 'SMARCA4', 'BRG1', 'HEF5',
# 'ZNF383', 'ZNF684', 'ZNF76', 'ZNF181', 'ZNF814', 'KMT2C', 'POU6F2','HOXC9','SP5','KLF3','ZNF317','ZNF572','HIVEP3','ZNF551','HOXA1','ZNF771',
# 'CERS4','PAX9','TGIF2','ZNF682','ZBTB8B','ZNF329','ZNF729','ZNF740','ZNF468','ZNF281','ZNF726','POU4F3','ZNF100','ZNF808','FOXM1','FOXI1',
# 'MAFG','ZFP41','ZNF416','ZNF681','ZNF85','ZNF582' 'HES5', 'ZNF80', 'ZNF219', 'TFAP2C', 'EPAS1', 'MECP2', 'NHLH1', 'CEBPE', 'TFE3', 'E2F5', 'E2F1',
# 'JUND', 'ZFX', 'NR2F2', 'MBD2', 'FOXP3', 'LHX2', 'HEY1', 'SRY', 'TGIF2', 'ZNF337', 'HSFX1', 'NFATC4', 'ZNF500', 'TP53', 'MECOM', 'PITX2', 'PKNOX2',
# 'ZNF684', 'STAT4', 'ZNF701', 'TBX6', 'ZSCAN5B', 'LHX2', 'ZNF660', 'ZNF287', 'ZSCAN12', 'ZNF836', "NEUROD1",'ASCL1', "POU2F3",'YAP1',]

tfs = ['ASCL1', 'ASCL2', 'ATF2', 'ATF3', 'AVIL', 'BACH2', 'BCL3',
       'BHLHE22', 'BRCA1', 'BRD4', 'CEBPA', 'CEBPB', 'CEBPD', 'CEBPE',
       'CERS4', 'CLOCK', 'CNOT3', 'CTCF', 'CXXC1', 'DLX5', 'DLX6', 'E2F1',
       'E2F4', 'E2F5', 'E2F6', 'EBF1', 'EGR1', 'ELF3', 'ELK3', 'EOMES',
       'EP300', 'EPAS1', 'ERCC6', 'ETS2', 'ETV4', 'ETV6', 'FLI1', 'FOSL1',
       'FOXA1', 'FOXA2', 'FOXJ3', 'FOXM1', 'FOXN4', 'FOXP3', 'GATA2',
       'GATA3', 'GATA4', 'GATA6', 'GLI1', 'GRIP1', 'HDAC2', 'HES1',
       'HES6', 'HEY1', 'HIVEP3', 'HMG20B', 'HOXA1', 'HOXB5', 'HSF1',
       'INSM1', 'ISL1', 'ISL2', 'JUN', 'JUND', 'KAT2A', 'KAT8', 'KDM2B',
       'KDM6A', 'KLF1', 'KLF12', 'KLF2', 'KLF3', 'KLF4', 'KLF5', 'KLF6',
       'KMT2C', 'LHX2', 'LYAR', 'MAF', 'MAFG', 'MAX', 'MBD1', 'MBD2',
       'MECOM', 'MECP2', 'MGA', 'MITF', 'MLX', 'MLXIP', 'MLXIPL', 'MNT',
       'MTA1', 'MXD1', 'MXD3', 'MXD4', 'MXI1', 'MYB', 'MYBL2', 'MYC',
       'MYCL', 'MYCN', 'MYT1', 'NANOG', 'NCOR1', 'NELFE', 'NFATC4',
       'NFKB2', 'NFKBIZ', 'NFYA', 'NFYB', 'NHLH1', 'NHLH2', 'NKX2-1',
       'NONO', 'NR0B2', 'NR1H3', 'NR2F2', 'ONECUT2', 'OVOL2', 'PATZ1',
       'PAX5', 'PAX9', 'PBX3', 'PITX2', 'PKNOX2', 'PML', 'POU2F1',
       'POU4F1', 'POU4F3', 'POU6F2', 'PPARG', 'PROX1', 'PURG', 'RARG',
       'RBP1', 'RBPJ', 'RCOR2', 'RELA', 'RELB', 'REST', 'RFX7', 'RORC',
       'RUNX1', 'RUNX2', 'SALL2', 'SCRT1', 'SCRT2', 'SIN3A', 'SIX5',
       'SMAD2', 'SMAD3', 'SMAD4', 'SMAD9', 'SMARCA4', 'SOX11', 'SOX18',
       'SOX2', 'SOX3', 'SP100', 'SP110', 'SP5', 'SP6', 'SPI1', 'ST18',
       'STAG1', 'STAT4', 'STAT5A', 'STAT6', 'TAF1', 'TAF7', 'TAF7L',
       'TBP', 'TBPL1', 'TBX10', 'TBX6', 'TCF12', 'TCF15', 'TCF21', 'TCF3',
       'TCF4', 'TEAD2', 'TEAD4', 'TFCP2L1', 'TFE3', 'TGIF2', 'TOX',
       'TOX3', 'TRP63', 'TSHZ2', 'TTF2', 'USF1', 'VEZF1', 'XBP1', 'XRN2',
       'YAP1', 'ZBTB16', 'ZBTB18', 'ZBTB21', 'ZBTB33', 'ZBTB7C', 'ZBTB8B',
       'ZFP217', 'ZFP281', 'ZFP3', 'ZFP41', 'ZFP64', 'ZFX', 'ZKSCAN2',
       'ZMIZ1', 'ZSCAN12']

G = nx.DiGraph()
# prelim_G = nx.DiGraph()
# with open("/Users/sarahmaddox/Dropbox (Vanderbilt)/Quaranta_Lab/SCLC/Network/mothers_network.csv") as infile:
#     for line in infile:
#         line = line.strip().split(',')
#         prelim_G.add_edge(line[0], line[1])

for tf in tfs: G.add_node(tf)

for tf in tfs:
    enrichr.build_tf_network(G, tf, tfs)
    time.sleep(1)

# for edge in prelim_G.edges():
#     if edge[0] in tfs and edge[1] in tfs:
#         G.add_edge(edge[0], edge[1])

outfile = open("/Users/smgroves/Dropbox (VU Basic Sciences)/pycharm_workspace/NetworkTools copy/archetype_networking/RPM_thesis/_0_network.csv", "w")
for edge in G.edges(): outfile.write("%s,%s\n" % (edge[0], edge[1]))
outfile.close()

Gp = prune(G, prune_sinks=False, prune_sources=False)
Gp.add_edge('NEUROD1',"MYC",db=["ChEA_2013","ChEA_2015"]) #artificially add NEUROD1 --> MYC connection based on Borromeo et al. 2016

outfile = open("/Users/smgroves/Dropbox (VU Basic Sciences)/pycharm_workspace/NetworkTools copy/archetype_networking/RPM_thesis/_1_network.csv", "w")
for edge in G.edges(): outfile.write("%s,%s\n" % (edge[0], edge[1]))
outfile.close()
Gpp = prune_info(Gp)

# add code to keep INS and GCG even though they don't have out-going edges
Gpp = prune_to_chea(Gp)

outfile = open("/Users/smgroves/Dropbox (VU Basic Sciences)/pycharm_workspace/NetworkTools copy/archetype_networking/RPM_thesis/_2_network.csv", "w")
for edge in Gpp.edges(): outfile.write("%s,%s\n" % (edge[0], edge[1]))
outfile.close()
