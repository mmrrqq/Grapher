import torch
from data.dataset import GraphDataModule
from misc.cli import GrapherCLI
from model.litgrapher import LitGrapher
from transformers import T5Tokenizer
import os
from pytorch_lightning.callbacks import RichProgressBar
from misc.utils import decode_graph

def main():       
    torch.set_float32_matmul_precision('medium')   
    cli = GrapherCLI(LitGrapher, GraphDataModule, trainer_defaults={ "callbacks": [RichProgressBar(10)] }, run=False)

    # cli.trainer.callbacks.append(RichProgressBar(10))

    args = cli.config
    
    os.makedirs(args.checkpoint.dirpath, exist_ok=True)
    os.makedirs(os.path.join(args.model.eval_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(args.model.eval_dir, 'test'), exist_ok=True)

    # TB = pl_loggers.TensorBoardLogger(save_dir=args.trainer.default_root_dir, name='', version=f"{args.data.dataset}_version_{args.core.version}", default_hp_metric=False)

    if args.core.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(args.checkpoint.dirpath, 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(args.checkpoint.dirpath, f"model-step={args.core.checkpoint_model_id}.ckpt")

    if args.run == 'train':
        if not os.path.exists(checkpoint_model_path):
            checkpoint_model_path = None

        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=checkpoint_model_path)    
        
    elif args.run == 'test':
        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot run the test'

        # add as args
        grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)        
        cli.trainer.test(grapher, datamodule=cli.datamodule)

    else: # single inference

        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot do inference'

        grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)

        tokenizer = T5Tokenizer.from_pretrained(grapher.transformer_name, cache_dir=grapher.cache_dir)
        tokenizer.add_tokens('__no_node__')
        tokenizer.add_tokens('__no_edge__')
        tokenizer.add_tokens('__node_sep__')
        
        text_tok = tokenizer([args.inference_input_text],
                             add_special_tokens=True,
                             padding=True,
                             return_tensors='pt')

        text_input_ids, mask = text_tok['input_ids'], text_tok['attention_mask']

        _, seq_nodes, _, seq_edges = grapher.model.sample(text_input_ids, mask)

        dec_graph = decode_graph(tokenizer, grapher.edge_classes, seq_nodes, seq_edges, grapher.edges_as_classes,
                                grapher.node_sep_id, grapher.max_nodes, grapher.noedge_cl, grapher.noedge_id,
                                grapher.bos_token_id, grapher.eos_token_id)
        
        graph_str = ['-->'.join(tri) for tri in dec_graph[0]]
        
        print(f'Generated Graph: {graph_str}')
        
    
if __name__ == "__main__":    
    main()
