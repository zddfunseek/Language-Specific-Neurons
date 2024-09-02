import argparse
import os
import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def plot_heatmap(attention_scores, model_id, plot_figs_per_head, save_fig_path, tokens_list=None, ignore_first_token=False, num_figs_per_row=4, layer_focus=None):
    """
    attention_scores: a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, seq_len]
    tokens_list: act as xticks and yticks of the figure, eg. ['<s>', 'Hi', ',', 'how', 'are', 'you', '?']
    """
    save_fig_path_model = os.path.join(save_fig_path, model_id) # the model's results are saved under this dir 
    os.makedirs(save_fig_path_model, exist_ok=True)

    if ignore_first_token:
        attention_scores = [attention_scores[i][:, :, 1: , 1: ] for i in range(len(attention_scores))]
        tokens_list = tokens_list[1: ]

    import pdb; pdb.set_trace()
    if layer_focus is None:
        layer_focus = [x for x in range(len(attention_scores))]
    else:
        layer_focus = [int(x) for x in layer_focus if x!=',']

    # a figure for all
    print(f'plotting a figure for all layers ...')
    num_heads = len(attention_scores)
    num_rows = math.ceil(num_heads / num_figs_per_row) 
    fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
    for layer_idx in tqdm(layer_focus):
        row, col = layer_idx // num_figs_per_row, layer_idx % num_figs_per_row
        avg_attention_scores = attention_scores[layer_idx][0].mean(dim=0)    # [ seq_len, seq_len]
        mask = torch.triu(torch.ones_like(avg_attention_scores, dtype=torch.bool), diagonal=1)
        sns.heatmap(avg_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col])
        axes[row, col].set_title(f'layer {layer_idx}')

    plt.suptitle(f'all layers avg') 
    plt.savefig(os.path.join(save_fig_path_model, f'all_layers_avg.jpg'))
    plt.close()   

    if not plot_figs_per_head:
        return

    # a figure for each layer
    for layer_idx in layer_focus:
        print(f'plotting layer {layer_idx} ...')
        num_heads = attention_scores[layer_idx].shape[1]
        num_rows = math.ceil(num_heads / num_figs_per_row)
        fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(len(tokens_list) * 2, 0.5 * num_rows * len(tokens_list)))
        for head_idx in tqdm(range(num_heads)):
            row, col = head_idx // num_figs_per_row, head_idx % num_figs_per_row
            head_attention_scores = attention_scores[layer_idx][0][head_idx]    # [seq_len, seq_len]
            mask = torch.triu(torch.ones_like(head_attention_scores, dtype=torch.bool), diagonal=1)
            sns.heatmap(head_attention_scores.numpy(), mask=mask.numpy(), cmap='RdBu_r', square=True, xticklabels=tokens_list, yticklabels=tokens_list, ax=axes[row, col])
            axes[row, col].set_title(f'head {head_idx}')

        plt.suptitle(f'layer_{layer_idx}') 
        plt.savefig(os.path.join(save_fig_path_model, f'layer_{layer_idx}.jpg'))
        plt.close()


# a wrapper
def view_attention(
    model=None,  # the model object
    model_id=None,
    tokenizer=None,
    prompt=None,
    save_attention_scores=False,
    save_attention_scores_path=None,
    load_attention_scores_path=None,
    save_fig_path=None,
    plot_figs_per_head=False,
    ignore_first_token=False,
    num_figs_per_row=4,
    layer_focus=None,
):
    if load_attention_scores_path:  # plot using the existing attention scores
        with open(load_attention_scores_path, 'rb') as f:
            saved_data = torch.load(f)
            attention_scores = saved_data['attention_scores']
            tokens_list = saved_data['tokens_list']

    else:
        assert model is not None and model_id is not None and prompt is not None and tokenizer is not None, \
            "`model`, `model_id`, `tokenizer` and `prompt` must all be specified without `load_attention_scores_path`!"
            
        inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)
        tokens_list = list(map(lambda x:x.replace('▁',''), tokenizer.convert_ids_to_tokens(inputs[0].cpu())))   # used as labels when plotting
        print("* Generating ...")
        with torch.no_grad():
            attention_scores = model(inputs, output_attentions=True)['attentions'] # a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, seq_len]
        attention_scores = [attention_scores_layer.detach().cpu() for attention_scores_layer in attention_scores]
        
        if save_attention_scores:  # each layer's attention scores is stored in one safetensors file
            assert save_attention_scores_path is not None, \
                "`save_attention_scores_path` must be specified to save attention scores!"
            print('* Saving attention scores ...')
            save_path = save_attention_scores_path
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f'{model_id}_attn_scores.pt'), 'wb') as f:
                saved_data = {
                    'attention_scores': attention_scores,
                    'tokens_list': tokens_list
                }
                torch.save(saved_data, f)

    print('Plotting heatmap for attention scores ...')
    plot_heatmap(attention_scores, model_id, plot_figs_per_head, save_fig_path, tokens_list, ignore_first_token, num_figs_per_row, layer_focus)



# parse arguments
parser = argparse.ArgumentParser()
# model config
parser.add_argument('--model_path', required=True, help='the path of the model')
parser.add_argument('--model_id', required=True, help='the name you give to the model')
# input config
parser.add_argument('--prompt', default='Summer is warm. Winter is cold.\n')
parser.add_argument('--ignore_first_token', action='store_true', help='whether to ignore the start token when plotting')
# saving and loading of attention scores
parser.add_argument('--save_attention_scores', action='store_true', help='whether to store the attention scores')
parser.add_argument('--save_attention_scores_path', default='./attn_scores')
parser.add_argument('--load_attention_scores_path', default=None, help='if specified, would just load the stored attention scores and plot')
# visualization
parser.add_argument('--plot_figs_per_head', action='store_true', help='whether to plot heatmap for each head')
parser.add_argument('--save_fig_path', default='./vis')
parser.add_argument('--num_figs_per_row', type=int, default=4)
parser.add_argument('--layer_focus', type=list, default=None, help='a list of layer index')
args = parser.parse_args()


if __name__ == "__main__":

        # load model and tokenizer
        config = AutoConfig.from_pretrained(args.model_path)
        config._attn_implementation = "eager"    # use vanilla attention to return attention weights
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        # visualize attention
        view_attention(
            model=model,  # the model object
            model_id=args.model_id,
            tokenizer=tokenizer,
            prompt=args.prompt,
            save_attention_scores=args.save_attention_scores,
            save_attention_scores_path=args.save_attention_scores_path,
            load_attention_scores_path=args.load_attention_scores_path,
            plot_figs_per_head=args.plot_figs_per_head,
            save_fig_path=args.save_fig_path,
            ignore_first_token=args.ignore_first_token,
            num_figs_per_row=args.num_figs_per_row,
            layer_focus=args.layer_focus,
        )