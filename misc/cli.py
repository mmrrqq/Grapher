import inspect
import os
from typing import Any, Callable, Dict, Optional, Type, Union
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI, ArgsType, SaveConfigCallback
from transformers import T5ForConditionalGeneration, T5Tokenizer


class GrapherCLI(LightningCLI):
    def __init__(
        self,
        model_class: Optional[
            Union[Type[LightningModule], Callable[..., LightningModule]]
        ] = None,
        datamodule_class: Optional[
            Union[Type[LightningDataModule], Callable[..., LightningDataModule]]
        ] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[
            Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        ] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ):
        super().__init__(
            model_class,
            datamodule_class,
            save_config_callback,
            save_config_kwargs,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            args,
            run,
            auto_configure_optimizers,
        )

    def add_arguments_to_parser(self, parser):
        parser.add_argument("core.pretrained_model", type=str, default="t5-small")
        parser.add_argument("core.version", type=int, default=0)
        parser.add_argument("core.checkpoint_model_id", type=int, default=-1)
        parser.add_argument("run", type=str, default="train")

        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

        parser.set_defaults(
            {
                "data.tokenizer_class": f"{T5Tokenizer.__module__}.{T5Tokenizer.__name__}",
                "model.transformer_class": f"{T5ForConditionalGeneration.__module__}.{T5ForConditionalGeneration.__name__}",
                "data.cache_dir": "cache",
                "data.dataset": "webnlg",
                "data.data_path": "webnlg-dataset/release_v3.0/en",
                "data.num_data_workers": 3,
                "data.max_nodes": 8,
                "data.max_edges": 7,
                "data.batch_size": 10,
                "model.default_seq_len_edge": 20,
                "model.num_layers": 1,
                "model.dropout_rate": 0.5,
                "model.focal_loss_gamma": 0.0,
                "model.lr": 1e-5,
                "trainer.default_root_dir": "output",
                "checkpoint.save_last": True,
                "checkpoint.save_top_k": -1,
                "checkpoint.every_n_train_steps": 1000,
                "checkpoint.filename": 'model-{step}'
            }
        )

        parser.link_arguments(
            ("trainer.default_root_dir", "data.dataset", "core.version"),
            "checkpoint.dirpath",
            compute_fn=lambda x, y, z: os.path.join(
                x, f"{y}_version_{z}", "checkpoints"
            ),
        )
        parser.link_arguments(
            ("trainer.default_root_dir", "data.dataset", "core.version"),
            "model.eval_dir",
            compute_fn=lambda x, y, z: os.path.join(
                x, f"{y}_version_{z}"
            ),
        )
        parser.link_arguments("core.pretrained_model", "data.tokenizer_name")
        parser.link_arguments("core.pretrained_model", "model.transformer_name")
        parser.link_arguments(
            "data.tokenizer", "model.tokenizer", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.tokenizer",
            "model.bos_token_id",
            apply_on="instantiate",
            compute_fn=lambda x: x.pad_token_id,
        )
        parser.link_arguments(
            "data.tokenizer",
            "model.eos_token_id",
            apply_on="instantiate",
            compute_fn=lambda x: x.eos_token_id,
        )
        parser.link_arguments(
            "data.dataset_train",
            "model.edge_classes",
            apply_on="instantiate",
            compute_fn=lambda x: x.edge_classes,
        )
        parser.link_arguments(
            "data.edges_as_classes", "model.edges_as_classes", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.max_nodes", "model.max_nodes", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.max_edges", "model.max_edges", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.cache_dir", "model.cache_dir", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.dataset_train",
            "model.num_classes",
            apply_on="instantiate",
            compute_fn=lambda x: len(x.edge_classes),
        )
        parser.link_arguments(
            "data.tokenizer",
            "model.vocab_size",
            apply_on="instantiate",
            compute_fn=lambda x: len(x.get_vocab()),
        )
        parser.link_arguments(
            "data.tokenizer",
            "model.nonode_id",
            apply_on="instantiate",
            compute_fn=lambda x: x.convert_tokens_to_ids("__no_node__"),
        )
        parser.link_arguments(
            "data.tokenizer",
            "model.noedge_id",
            apply_on="instantiate",
            compute_fn=lambda x: x.convert_tokens_to_ids("__no_edge__"),
        )
        parser.link_arguments(
            "data.tokenizer",
            "model.node_sep_id",
            apply_on="instantiate",
            compute_fn=lambda x: x.convert_tokens_to_ids("__node_sep__"),
        )
        parser.link_arguments(
            "data.dataset_train",
            "model.noedge_cl",
            apply_on="instantiate",
            compute_fn=lambda x: len(x.edge_classes) - 1,
        )
                
        parser.add_argument(
            "--inference_input_text",
            type=str,
            default="Danielle Harris had a main role in Super Capers, a 98 minute long movie.",
        )
