import argparse
import os
import math
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 计算最后一维的相邻向量之间的余弦相似度
def compute_cosine_similarity(hiddenstates):
    # 获取相邻的 layer 向量
    vec1 = hiddenstates[:-1]   # [layer-1, batch, seq_len, seq_len]
    vec2 = hiddenstates[1:]    # [layer-1, batch, seq_len, seq_len]

    # 计算每个 layer 对应向量的 L2 范数 (按 seq_len 维度和最后一个 seq_len 维度计算)
    norm1 = torch.norm(vec1, dim=-1)  # [layer-1, batch, seq_len]
    norm2 = torch.norm(vec2, dim=-1)  # [layer-1, batch, seq_len]

    # 计算相邻 layer 之间的内积
    dot_product = torch.sum(vec1 * vec2, dim=-1)  # [layer-1, batch, seq_len]

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity


def plot_heatmap(hiddenstates, model_id, plot_figs_per_head, save_fig_path, tokens_list=None, ignore_first_token=False, num_figs_per_row=4, layer_focus=None):
    """
    hiddenstates: a list containing 32 layers' attention scores, each is a tensor with shape [1, num_heads, seq_len, hidden_dim]
    tokens_list: act as xticks and yticks of the figure, eg. ['<s>', 'Hi', ',', 'how', 'are', 'you', '?']
    """
    save_fig_path_model = os.path.join(save_fig_path, model_id) # the model's results are saved under this dir 
    os.makedirs(save_fig_path_model, exist_ok=True)

    if ignore_first_token:
        hiddenstates = [hiddenstates[i][:, :, 1: , 1: ] for i in range(len(hiddenstates))]
        tokens_list = tokens_list[1: ]

    # 调用函数计算相邻序列的相似度
    #import pdb; pdb.set_trace()
    hiddenstates = torch.stack(hiddenstates)
    similarity = compute_cosine_similarity(hiddenstates)
    with open(os.path.join(save_fig_path_model, f'hiddenstate_show.txt'), 'w') as f:
        for _i, x in enumerate(tokens_list):
            f.write(f'\n{_i}-->{x}:\n\t')
            [f.write(f'{y.item():.2f} ') for y in similarity[...,_i]]

    # 创建自定义颜色映射
    # mycolors = ['darkblue', 'blue', 'lightblue', 'white']  # 从深到浅的颜色
    # mycolors = ['#ffffff', '#e6f2ff', '#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#007acc', '#0059b3', '#003d80', '#002966']
    mycolors = ['#f2f9ff', '#cce5ff', '#66b3ff', '#3399ff', '#1f8cff', '#007acc', '#0059b3', '#003d80', '#002966', '#001a66', '#001233', '#000f1a', '#000b0b']
    mycolors = ['#ffffff', '#f2f9ff', '#cce5ff', '#66b3ff', '#3399ff', '#1f8cff', '#007acc', '#0059b3', '#003d80', '#002966']
    mycolors = ['#ffffff', '#f2f9ff',            '#66b3ff',          '#1f8cff',               '#0059b3',            '#002966']
    mycmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', mycolors)

    plotBlockSize = 100
    plotBlockNum = math.ceil(similarity.size(-1) / plotBlockSize)
    for _bi in range(plotBlockNum):
        blockStart = _bi * plotBlockSize
        blockEnd =  (_bi + 1) * plotBlockSize if _bi < plotBlockNum - 1 else len(tokens_list)

        # a figure for all
        print(f'plotting a figure for either the specified or all layers within ranges of {blockStart}-{blockEnd} tokens ...')
        num_cols = 1
        num_rows = 1
        is_vertical_style = True
        block_tokens_list = [f'{x}_{i}' for i, x in enumerate(tokens_list[blockStart:blockEnd])]
        if is_vertical_style:
            fig, axes = plt.subplots(1, 1, figsize=(len(hiddenstates), len(block_tokens_list))) ### (numLayers, numTokens)
            axes = np.reshape(axes,(num_rows,num_cols))
            myheatmap = sns.heatmap(similarity.squeeze(1).transpose(0,1)[blockStart:blockEnd,...].numpy(), fmt=".2f", cmap=mycmap, square=True, yticklabels=block_tokens_list, xticklabels=[i for i in range(len(hiddenstates) - 1)], ax=axes[0, 0])
        else:
            fig, axes = plt.subplots(1, 1, figsize=(len(block_tokens_list), len(hiddenstates)))
            axes = np.reshape(axes,(num_rows,num_cols))
            myheatmap = sns.heatmap(similarity.squeeze(1)[blockStart:blockEnd,...].numpy(), cmap=mycmap, square=True, xticklabels=block_tokens_list, yticklabels=[i for i in range(len(hiddenstates) - 1)], ax=axes[0, 0])
        axes[0, 0].tick_params(axis='both', labelsize=32)
        axes[0, 0].set_title(f'Block {blockStart}-{blockEnd}', fontsize=45)
        colorbar = myheatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=32)  # 设置 colorbar 刻度字体大小
        colorbar.set_label('Colorbar Label', fontsize=32)  # 设置 colorbar 标签字体大小
        ticks = np.linspace(0, 1, num=11)  # 创建 10 个刻度
        colorbar.set_ticks(ticks)  # 设置 colorbar 刻度
        colorbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])  # 设置刻度标签格式
        colorbar.ax.set_aspect(20)  # 设置 colorbar 的宽度  

        plt.suptitle(f'hiddenstate_similarity') 
        plt.savefig(os.path.join(save_fig_path_model, f'hiddenstate_{"ver" if is_vertical_style else "hor"}_{blockStart}-{blockEnd}.jpg'))
        plt.close() 

# a wrapper
def view_hidden(
    model=None,  # the model object
    model_id=None,
    tokenizer=None,
    prompt=None,
    save_hiddenstates=False,
    save_hiddenstates_path=None,
    load_hiddenstates_path=None,
    save_fig_path=None,
    plot_figs_per_head=False,
    ignore_first_token=False,
    num_figs_per_row=4,
    layer_focus=None,
):
    if load_hiddenstates_path:  # plot using the existing attention scores
        with open(load_hiddenstates_path, 'rb') as f:
            saved_data = torch.load(f)
            hiddenstates = saved_data['hidden_states']
            tokens_list = saved_data['tokens_list']

    else:
        assert model is not None and model_id is not None and prompt is not None and tokenizer is not None, \
            "`model`, `model_id`, `tokenizer` and `prompt` must all be specified without `load_hiddenstates_path`!"
            
        inputs = tokenizer(prompt, return_tensors="pt")['input_ids'].to(model.device)
        tokens_list = list(map(lambda x:x.replace('▁',''), tokenizer.convert_ids_to_tokens(inputs[0].cpu())))   # used as labels when plotting
        print("* Generating ...")
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            ### a list containing 33 layers' hidden states, each layer is a tensor with shape [1, seq_len, hidden_dim]
            ### NOTE:  the first layer is embedding, while the last layer is the normalization of hidden states
            hiddenstates = model(inputs, output_hidden_states=True)['hidden_states'] 
        hiddenstates = [hiddenstates_layer.detach().cpu() for hiddenstates_layer in hiddenstates]
        
        if save_hiddenstates:  # each layer's attention scores is stored in one safetensors file
            assert save_hiddenstates_path is not None, \
                "`save_hiddenstates_path` must be specified to save attention scores!"
            print('* Saving attention scores ...')
            save_path = save_hiddenstates_path
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f'{model_id}_attn_scores.pt'), 'wb') as f:
                saved_data = {
                    'hiddenstates': hiddenstates,
                    'tokens_list': tokens_list
                }
                torch.save(saved_data, f)

    print('Plotting heatmap for attention scores ...')
    plot_heatmap(hiddenstates, model_id, plot_figs_per_head, save_fig_path, tokens_list, ignore_first_token, num_figs_per_row, layer_focus)



# parse arguments
parser = argparse.ArgumentParser()
# model config
parser.add_argument('--model_path', required=True, help='the path of the model')
parser.add_argument('--model_id', type=str, default=None, help='the name you give to the model')
# input config
parser.add_argument('--prompt', default='Summer is warm. Winter is cold.\n')
parser.add_argument('--ignore_first_token', action='store_true', help='whether to ignore the start token when plotting')
# saving and loading of attention scores
parser.add_argument('--save_hiddenstates', action='store_true', help='whether to store the attention scores')
parser.add_argument('--save_hiddenstates_path', default='./attn_scores')
parser.add_argument('--load_hiddenstates_path', default=None, help='if specified, would just load the stored attention scores and plot')
# visualization
parser.add_argument('--plot_figs_per_head', action='store_true', help='whether to plot heatmap for each head')
parser.add_argument('--save_fig_path', default='./vis')
parser.add_argument('--num_figs_per_row', type=int, default=4)
parser.add_argument('--layer_focus', type=str, default=None, help='a list of layer index')
args = parser.parse_args()


if __name__ == "__main__":

    # load model and tokenizer
    #import pdb; pdb.set_trace()
    config = AutoConfig.from_pretrained(args.model_path)
    config._attn_implementation = "eager"    # use vanilla attention to return attention weights
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # visualize attention
    view_hidden(
        model=model,  # the model object
        model_id=os.path.basename(args.model_path) if args.model_id is None else args.model_id,
        tokenizer=tokenizer,
        prompt=args.prompt,
        save_hiddenstates=args.save_hiddenstates,
        save_hiddenstates_path=args.save_hiddenstates_path,
        load_hiddenstates_path=args.load_hiddenstates_path,
        plot_figs_per_head=args.plot_figs_per_head,
        save_fig_path=args.save_fig_path,
        ignore_first_token=args.ignore_first_token,
        num_figs_per_row=args.num_figs_per_row,
        layer_focus=args.layer_focus,
    )