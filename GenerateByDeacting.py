import argparse
import json
import os
from types import MethodType

import torch
import subprocess
import signal
import torch.nn.functional as F
from vllm import LLM, SamplingParams

answer_lang = {
    "zh": "请用中文回答。",
    "en": " Answer in English.",
    "fr": " Veuillez répondre en français.",
    "es": " Por favor responda en español.",
    "id": " Tolong dijawab dalam bahasa Indonesia.",
    "ja": "日本語で答えてください。",
    "vi": " Hãy trả lời bằng tiếng Việt.",
}


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/dozhang/Llama-3/Meta-Llama-3-8B-Instruct")
parser.add_argument("-a", "--activation_mask", type=str, default="output/hh-rlhf/train.activations.llama-3-inst")
args = parser.parse_args()

def Load_testdata():
    # process input data files
    input_file = '/home/dozhang/hh-rlhf/harmless-base/test.jsonl.gz'
    import gzip
    import json
    datainput = {}
    datainput['prompt'] = []
    datainput['chosen'] = []
    datainput['rejected'] = []
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            json_obj = json.loads(line)
            p1data = json_obj['chosen']
            p2data = json_obj['rejected']
            
            achor = 'Assistant: '
            last_occurrence_index = p1data.rfind(achor)
            assert (p1data[:last_occurrence_index].strip() == p2data[:last_occurrence_index].strip(), f"{json_obj}")

            datainput['prompt'].append(p1data[:last_occurrence_index].strip())
            datainput['chosen'].append(p1data[last_occurrence_index + len(achor):].strip())
            datainput['rejected'].append(p2data[last_occurrence_index + len(achor):].strip())

    return datainput

def DestroyModel():
    # Run the nvidia-smi command with query options
    command = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)

    # Get the output and split by newlines to get individual PIDs
    pids = result.stdout.strip().split('\n')
    os.kill(int(pids[0]), signal.SIGTERM)

testdata = Load_testdata()
activation_masks = torch.load(args.activation_mask)
activation_masks = [None]

is_llama = bool(args.model.lower().find("llama") >= 0)

# if args.activation_mask:
#     activation_masks = torch.load(args.activation_mask)
#     activation_mask_name = args.activation_mask.split("/")[-1].split(".")
#     activation_mask_name = ".".join(activation_mask_name[1:])
# else:
#     activation_masks = [None]


output_folder = f"output/hh-rlhf/results"
os.makedirs(output_folder, exist_ok=True)

tasks = ["chosen", "rejected"]
for activation_mask, mask_lang in zip(activation_masks[:1], tasks[:1]):

    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    sampling_params = SamplingParams(temperature=0, repetition_penalty=1.1, max_tokens = 2048, stop = ["</s>", "<|eot_id|>", "Human: "])
    #import pdb; pdb.set_trace()
    # DestroyModel()
    # model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

    if activation_mask:
        def factory(mask):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                i = gate_up.size(-1)
                activation = F.silu(gate_up[:, :, : i // 2])
                activation.index_fill_(2, mask, 0)
                x = activation * gate_up[:, :, i // 2 :]
                x, _ = self.down_proj(x)
                return x

            def bloom_forward(self, x: torch.Tensor):
                x, _ = self.dense_h_to_4h(x)
                x = self.gelu_impl(x)
                x.index_fill_(2, mask, 0)
                x, _ = self.dense_4h_to_h(x)
                return x

            if is_llama:
                return llama_forward
            else:
                return bloom_forward

        for i, layer_mask in enumerate(activation_mask):
            if is_llama:
                obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
            else:
                obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
            obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)
    
    
    for lang in tasks:    
        texts = testdata['prompt'][:10]
        outputs = model.generate(texts, sampling_params)
        outputs = [o.outputs[0].text.strip() for o in outputs]

        if activation_mask:
            output_file = f"{output_folder}/{lang}.perturb_by.{mask_lang}.jsonl"
        else:
            output_file = f"{output_folder}/{lang}.normal.jsonl"

        results = []
        for raw, t, o in zip(testdata[lang][:10], texts, outputs):
            out = {
                    "lang": lang,
                    "prompt": t, 
                    "oriresponse": raw,
                    "output": o
                  }
            results.append(out)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")
    
    model = None
