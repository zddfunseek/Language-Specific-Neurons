import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from hf_model_adapt import hf_adapt

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查最新生成的 token 是否是停止标记
        return input_ids[0, -1] in self.stop_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/home/dozhang/nlcmt1/HuggingfaceModels/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/home/dozhang/nlcmt1/HuggingfaceModels/Meta-Llama-3.1-8B-Instruct").to(device)

# Encode input text
input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: If a bag of marbles costs $20 and the price increases by 20% of the original price every two months, how much would a bag of marbles cost after 36 months?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

#input_text = "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"

inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Create attention mask
attention_mask = inputs.attention_mask

# 定义停止条件
stop_token_id = tokenizer.convert_tokens_to_ids(["</s>", "<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>", "<|start_header_id|>"])
stopping_criteria = StoppingCriteriaList([StopTokenCriteria(stop_token_id)])

### Todo: complete hf_adapt 
# import pdb; pdb.set_trace()
(model, sum1, sum2, sum3, over_zero, flat_zero) = hf_adapt(model)

# Generate text
outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_length=512, num_return_sequences=1, 
                         pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
