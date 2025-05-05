import torch
import networkx as nx
from WebNLG_Text_to_triples import Evaluation_script_json
import os
from misc.rdf import save_webnlg_rdf
from torch.distributions import MultivariateNormal, VonMises
import json

failed_node = "failed node"
failed_edge = "failed edge"
nonode_str = "__no_node__"


def compute_loss(
    criterion,
    logits_nodes,
    logits_edges,
    target_nodes,
    target_edges,
    edges_as_classes,
    focal_loss_gamma,
):
    # --------- Node Loss ---------
    # shift forward 1 step to create labels
    labels = torch.cat(
        [target_nodes[:, 1:], torch.zeros_like(target_nodes[:, -2:-1])], 1
    )

    loss_nodes = criterion["ce"](logits_nodes.transpose(1, 2), labels).mean()

    # --------- Edge Loss --------
    if edges_as_classes:
        target_edges = target_edges.permute(2, 0, 1)
        logits_edges = logits_edges.permute(2, 3, 0, 1)
        if focal_loss_gamma:
            loss_edges = criterion["focal"](logits_edges, target_edges).mean()
        else:
            loss_edges = criterion["ce"](logits_edges, target_edges).mean()

    else:  # full
        target_edges = target_edges.permute(2, 0, 1, 3)
        logits_edges = logits_edges.permute(2, 4, 0, 1, 3)
        loss_edges = criterion["ce"](logits_edges, target_edges).mean()

    loss = loss_nodes + loss_edges

    return loss


def compute_spatial_loss(
    criterion,
    logits_nodes,
    logits_edges,
    logits_spatial,
    target_nodes,
    target_edges,
    target_spatial,
    edges_as_classes,
    focal_loss_gamma,
    noedge_id
):
    # transform to 1x1xbatch_size matrix
    # for now, transform to directional correlation matrix.. however, I might be able to use this to represent negated relations?!
    target_node_edge_matrix = torch.full(
        (2, 2, target_edges.size(0)), noedge_id, device=target_edges.device, dtype=torch.long
    )  # 2 as max nodes in relation
    target_node_edge_matrix[0, 1, :] = target_edges
    # target_edges = target_edges.unsqueeze(dim=0).unsqueeze(dim=0)
    node_edge_loss = compute_loss(
        criterion,
        logits_nodes,
        logits_edges,
        target_nodes,
        target_node_edge_matrix,
        edges_as_classes,
        focal_loss_gamma,
    )

    ry_dist, phi_dist = build_spatial_distributions(logits_spatial)
    # mean, lower, diag = max_spatial.split(3, -1)
    # z = torch.zeros(size=[mean.size(0)], device=mean.device)
    # diag = torch.exp(diag)
    # scale_tril = torch.stack([
    #     diag[:, 0], z         , z,
    #     lower[:, 0], diag[:, 1], z,
    #     lower[:, 1], lower[:, 2], diag[:, 2]
    # ], dim=-1).view(-1, 3, 3)

    # dist = MultivariateNormal(loc=mean, scale_tril=scale_tril)

    target_x, target_y, target_z = target_spatial.split(1, -1)
    target_r = torch.sqrt(torch.pow(target_x, 2) + torch.pow(target_z, 2))
    target_phi = torch.atan2(target_z, target_x)

    # target spatial is expected to be of size (batch_size X 3)
    spatial_loss = (
        ry_dist.log_prob(torch.hstack((target_r, target_y)))
        * phi_dist.log_prob(target_phi)
    ).mean()

    return node_edge_loss + spatial_loss


def decode(cand, bos_token_id, eos_token_id, tokenizer, failed=failed_node):
    bos_mask = (cand == bos_token_id).nonzero(as_tuple=False)
    if len(bos_mask) > 0:
        eos_mask = (cand == eos_token_id).nonzero(as_tuple=False)
        if len(eos_mask) > 0:
            s = tokenizer._decode(cand[bos_mask[0] + 1 : eos_mask[0]])
        else:
            s = failed
    else:
        s = failed

    return s


def decode_text(tokenizer, text_input_ids, bos_token_id, eos_token_id):
    text_decoded = []

    for text in text_input_ids:
        bos_mask = (text == bos_token_id).nonzero(as_tuple=False)
        eos_mask = (text == eos_token_id).nonzero(as_tuple=False)
        text_dec = tokenizer._decode(text[bos_mask[0] + 1 : eos_mask[0]])
        text_decoded.append(text_dec)

    return text_decoded


def decode_graph(
    tokenizer,
    edge_classes,
    bnodes,
    bedges,
    edges_as_classes,
    node_sep_id,
    max_nodes,
    noedge_cl,
    noedge_id,
    bos_token_id,
    eos_token_id,
):
    if edges_as_classes:
        bedges = bedges.permute(2, 0, 1)
    else:
        bedges = bedges.permute(2, 0, 1, 3)

    # bnodes: batch_size X num_nodes X seq_len_node
    # bedges: batch_size X num_nodes X num_nodes X seq_len_edge [FULL]
    # bedges: batch_size X num_nodes X num_nodes [CLASSES]

    triples_decoded = []

    for b_ind, (nodes, edges) in enumerate(zip(bnodes, bedges)):
        G = nx.DiGraph()

        nodes_decoded = []
        all_nodes = tokenizer._decode(nodes).split(tokenizer._decode(node_sep_id))

        for n in all_nodes:
            s = n.replace("<pad>", "").replace("</s>", "").strip()
            # empty or white space
            if not s or not s.strip():
                s = failed_node
            nodes_decoded.append(s)

        nodes_decoded = nodes_decoded[:max_nodes]
        nodes_decoded += (max_nodes - len(nodes_decoded)) * [failed_node]

        if edges_as_classes:
            noedge_mask = ~(bedges == noedge_cl)
            for i in range(max_nodes):
                for j in range(max_nodes):
                    if i == j:
                        continue
                    if noedge_mask[b_ind][i, j] > 0:
                        edge = edges[i, j].detach()

                        if edge == noedge_cl:
                            s = failed_edge
                        else:
                            s = edge_classes[edge]

                        if (
                            nodes_decoded[i] != failed_node
                            and nodes_decoded[j] != failed_node
                            and s != failed_edge
                            and nonode_str not in nodes_decoded[i]
                            and nonode_str not in nodes_decoded[j]
                        ):
                            G.add_edge(nodes_decoded[i], nodes_decoded[j], edge=s)
        else:  # full
            noedge_mask = 1 - torch.sum(bedges == noedge_id, -1)
            for i in range(max_nodes):
                for j in range(max_nodes):
                    if i == j:
                        continue
                    if noedge_mask[b_ind][i, j] > 0:
                        edge = edges[i, j]

                        s = decode(
                            edge, bos_token_id, eos_token_id, tokenizer, failed_edge
                        )

                        # empty or white space
                        if not s or not s.strip():
                            s = failed_edge

                        if (
                            failed_node not in nodes_decoded[i]
                            and failed_node not in nodes_decoded[j]
                            and s != failed_edge
                            and nonode_str not in nodes_decoded[i]
                            and nonode_str not in nodes_decoded[j]
                        ):
                            G.add_edge(nodes_decoded[i], nodes_decoded[j], edge=s)

        # make sure there are at least 2 nodes and 1 edge
        if nx.is_empty(G):
            node1 = nodes_decoded[0] if len(nodes_decoded) > 0 else failed_node
            node2 = nodes_decoded[1] if len(nodes_decoded) > 1 else failed_node
            G.add_edge(node1, node2, edge=failed_edge)

        tri = []
        for ind, (u, v, d) in enumerate(G.edges(data=True)):
            # decode up to 8 paths, discard others (because eval fails with too many paths)
            if ind >= 8:
                break

            tri.append([u, d["edge"], v])

        triples_decoded.append(tri)

    return triples_decoded


def build_spatial_distributions(spatial_logits):
    max_spatial = spatial_logits.max(0)[0]
    batch_size = max_spatial.size(0)

    ry_mean = max_spatial[:, :2]
    zeros = torch.zeros(size=[batch_size], device=ry_mean.device)
    diag = torch.exp(max_spatial[:, 2:4])
    scale_tril = torch.stack(
        [diag[:, 0], zeros, max_spatial[:, 4], diag[:, 1]], dim=-1
    ).view(-1, 2, 2)
    ry_dist = MultivariateNormal(loc=ry_mean, scale_tril=scale_tril)

    phi_mean = max_spatial[:, 5]
    phi_conc = torch.exp(max_spatial[:, 6])
    phi_dist = VonMises(phi_mean, phi_conc)

    return ry_dist, phi_dist


def sample_spatial(spatial_logits):    
    ry_dist, phi_dist = build_spatial_distributions(spatial_logits)

    samples = phi_dist.sample()

    r, y = ry_dist.sample().split(1, -1)
    r = r.squeeze()    
    x = torch.cos(samples) * r
    z = torch.sin(samples) * r

    return torch.stack((x, y.squeeze(), z)).T


def compute_scores(hyp, ref, iteration, eval_dir, split, rank):
    refs = [[" | ".join(i) for i in t] for t in ref]
    hyps = [[" | ".join(i) for i in t] for t in hyp]
    categories = [" "] * len(refs)

    ref_fname, hyp_fname = save_webnlg_rdf(
        hyps, refs, categories, os.path.join(eval_dir, split), f"{iteration}_{rank}"
    )

    scores_fname = os.path.join(eval_dir, split, f"scores_{iteration}_{rank}.json")

    Evaluation_script_json.main(ref_fname, hyp_fname, scores_fname)

    scores = json.load(open(scores_fname))
    scores = {
        "Precision": scores["Total_scores"]["Exact"]["Precision"],
        "Recall": scores["Total_scores"]["Exact"]["Recall"],
        "F1": scores["Total_scores"]["Exact"]["F1"],
    }

    return scores
