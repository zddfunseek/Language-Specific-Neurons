import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from hf_model_adapt import hf_adapt
import tensor_parallel as tp

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查最新生成的 token 是否是停止标记
        return input_ids[0, -1] in self.stop_token_id

# Load the tokenizer and model
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tokenizer = AutoTokenizer.from_pretrained("/home/dozhang/nlcmt/HuggingfaceModels/Llama-2-13b-chat-hf")
#model = AutoModelForCausalLM.from_pretrained("/home/dozhang/nlcmt/HuggingfaceModels/Llama-2-13b-chat-hf", torch_dtype=torch.bfloat16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("/home/dozhang/nlcmt/HuggingfaceModels/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/home/dozhang/nlcmt/HuggingfaceModels/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

#tokenizer = AutoTokenizer.from_pretrained("/home/dozhang/nlcmt1/HuggingfaceModels/Meta-Llama-3.1-70B-Instruct")
#model = AutoModelForCausalLM.from_pretrained("/home/dozhang/nlcmt1/HuggingfaceModels/Meta-Llama-3.1-70B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")


# Encode input text
input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: If a bag of marbles costs $20 and the price increases by 20% of the original price every two months, how much would a bag of marbles cost after 36 months?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n2+5=10, 3+6=18, 4+7=?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

input_text = "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"

input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Create attention mask
attention_mask = inputs.attention_mask

# 定义停止条件
stop_token_id = tokenizer.convert_tokens_to_ids(["</s>", "<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>", "<|start_header_id|>"])
stopping_criteria = StoppingCriteriaList([StopTokenCriteria(stop_token_id)])

### Todo: complete hf_adapt 
(model, sum1, sum2, sum3, over_zero, flat_zero) = hf_adapt(model, tokenizer, nBarLayer=20, valBarSim=0.96, nOutLayer = 2, nCheckLayer=2, nWarmupTok = -1)

#import pdb; pdb.set_trace()
# Generate text
generation_kwargs = {
                        "do_sample":False,
                         "temperature":0, 
                         "top_p":1
                    }
outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_length=512, num_return_sequences=1, 
                         pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria, **generation_kwargs)

# Decode the generated text
input_length = inputs["input_ids"].shape[1]
generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)

print(f'\nPrompt::: {input_text}\n')
print(f'Response >>> {generated_text}')
