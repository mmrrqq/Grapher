import importlib
from pathlib import Path
import pytorch_lightning as pl
import os
import json
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from misc.rel3d_utils import read_raw_transformation


class SpatialGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_class: str,
        tokenizer_name,
        cache_dir,
        data_path,
        dataset,
        batch_size,
        num_data_workers,
        edges_as_classes = True,
    ):
        super().__init__()

        tokenizer_module = importlib.import_module(
            tokenizer_class[: tokenizer_class.rfind(".")]
        )
        tokenizer_class = getattr(
            tokenizer_module, tokenizer_class[tokenizer_class.rfind(".") + 1 :]
        )

        self.cache_dir = cache_dir
        self.tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name, cache_dir=cache_dir
        )
        self.tokenizer.add_tokens("__no_node__")
        self.tokenizer.add_tokens("__no_edge__")
        self.tokenizer.add_tokens("__node_sep__")

        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.data_path = data_path
        self.output_path = os.path.join(data_path, "processed")
        os.makedirs(self.output_path, exist_ok=True)
        self.dataset = dataset
        self.prepared = False
        self.edges_as_classes = edges_as_classes
        self.max_nodes = 2
        self.max_edges = 1

    @property
    def dataset_train(self):
        if getattr(self, "_dataset_train", None) is None:
            self.setup("train")

        return self._dataset_train

    @property
    def dataset_dev(self):
        if getattr(self, "_dataset_valid", None) is None:
            self.setup("valid")

        return self._dataset_valid

    @property
    def dataset_test(self):
        if getattr(self, "_dataset_test", None) is None:
            self.setup("test")

        return self._dataset_test

    def setup(self, stage=None):
        if not self.prepared:
            self.prepare_data()

        self.__setattr__(
            f"_dataset_{stage}",
            SpatialGraphDataset(
                tokenizer=self.tokenizer,
                data_path=self.data_path,
                stage=stage,
                edges_as_classes=self.edges_as_classes,
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            collate_fn=self.dataset_train._collate_fn,
            num_workers=self.num_data_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_dev,
            batch_size=self.batch_size,
            collate_fn=self.dataset_dev._collate_fn,
            num_workers=self.num_data_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            collate_fn=self.dataset_test._collate_fn,
            num_workers=self.num_data_workers,
            shuffle=False,
            pin_memory=True,
        )


class SpatialGraphDataset(Dataset):
    def __init__(self, tokenizer, data_path, stage, edges_as_classes):
        if stage == "fit":
            stage = "train"

        self.tokenizer = tokenizer

        stage_path = Path(data_path) / f"{stage}.json"
        self.stage = stage
        self.relations = json.load(open(stage_path))

        self.edgess_as_classes = edges_as_classes

        self.parse_graph_data()

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        item = (
            self.utterances[index],
            self.nodes[index],
            self.node_positions[index],
            self.edges[index],
            self.edge_indexes[index],
        )
        return item

    def parse_graph_data(self):
        self.nodes = []
        self.node_positions = []
        self.edges = []
        self.utterances = []
        self.edge_indexes = []

        unique_relations = set(["__no_edge__"])
        for rel in self.relations:
            unique_relations.add(rel["predicate"])
        self.edge_classes = list(unique_relations)

        for rel in self.relations:
            relation_nodes = [rel["subject"], rel["object"]]
            # vector from object to subject in camera frame
            rel_transform = torch.tensor(rel["subject_transform"]) - torch.tensor(rel["object_transform"])
            pred = rel["predicate"]

            if "utterance" not in rel:
                print(f"skipping {rel['predicate']} {self.stage}")
                print(rel)
                continue

            self.utterances.append(rel["utterance"])
            self.nodes.append(relation_nodes)
            self.node_positions.append(rel_transform)
            self.edges.append(pred)
            self.edge_indexes.append(self.edge_classes.index(pred))

    def _build_inputs_with_special_tokens(self, token_ids_0, _):
        # T5:   <pad_id> token_ids_0 <eos_id>
        return (
            [self.tokenizer.pad_token_id] + token_ids_0 + [self.tokenizer.eos_token_id]
        )

    def _collate_fn(self, data):
        text_list = []
        node_list = []
        node_positions_list = []
        edge_list = []
        edge_ind_list = []

        for item in data:
            text, nodes, node_positions, edge, edge_ind = item
            text_list.append(text)
            node_list.append(nodes)
            node_positions_list.append(node_positions)
            edge_list.append(edge)
            edge_ind_list.append(edge_ind)

        self.tokenizer.build_inputs_with_special_tokens = (
            self._build_inputs_with_special_tokens
        )

        # ----------------- TEXT ----------------------

        text_batch = self.tokenizer(
            text_list, add_special_tokens=True, padding=True, return_tensors="pt"
        )

        collated_data = (
            text_batch["input_ids"],
            text_batch["attention_mask"],
        )

        # ------------------ NODES -------------------------
        node_list_text = []
        for node in node_list:
            node_list_text.append(" __node_sep__ ".join(node) + " __node_sep__")

        node_batch = self.tokenizer(
            node_list_text, add_special_tokens=True, padding=True, return_tensors="pt"
        )

        collated_data += (
            node_batch["input_ids"],
            node_batch["attention_mask"],
        )

        collated_data += (torch.stack(node_positions_list),)

        # ----------------- EDGES ----------------------
        if self.edgess_as_classes:
            collated_data += (torch.tensor(edge_ind_list).long(),)

        else:  # edges full
            raise NotImplementedError()
            flat_edge = [node for nodes in edge_list for node in nodes]

            flat_edge_tok = self.tokenizer(
                flat_edge,
                add_special_tokens=True,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

            no_edge_tok = self.tokenizer(
                ["__no_edge__"],
                add_special_tokens=True,
                padding=False,
                return_attention_mask=False,
            )["input_ids"][0]

            cumsum = np.cumsum([0] + [len(node) for node in edge_list])

            edge_batch = [flat_edge_tok[i:j] for (i, j) in zip(cumsum[:-1], cumsum[1:])]

            max_len = max([len(item) for item in flat_edge_tok])

            edge_batch_padded = []
            for edges in edge_batch:
                edge_batch_padded.append(
                    [
                        i + [self.tokenizer.pad_token_id] * (max_len - len(i))
                        for i in edges
                    ]
                )

            no_edge_tok = np.array(
                no_edge_tok
                + [self.tokenizer.pad_token_id] * (max_len - len(no_edge_tok))
            )

            edge_mat = np.tile(
                no_edge_tok, (len(data), self.max_nodes, self.max_nodes, 1)
            )

            for i, (edges_padded, edge_ind) in enumerate(
                zip(edge_batch_padded, edge_ind_list)
            ):
                for e_p, e_i in zip(edges_padded, edge_ind):
                    edge_mat[i, e_i[0], e_i[1]] = e_p

            edge_mat = torch.as_tensor(edge_mat).permute(1, 2, 0, 3)

            collated_data += (edge_mat,)

        return collated_data
