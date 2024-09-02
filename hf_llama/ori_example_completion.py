# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from ori_generation import Llama
from ori_model_adapt import ori_adapt


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0, #0.2,
    top_p: float = 1, #0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    (generator, sum1, sum2, sum3, over_zero, flat_zero) = ori_adapt(generator)

    prompts = ["<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nKylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"]
    
    prompts = ["Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]

    prompts = ["<|start_header_id|>system<|end_header_id|><|eot_id|>Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]

    prompts = ["<|eot_id|>Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]

    prompts = ["<|eot_id|>Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]

    

    prompts = ["1+2=?"]

    prompts = ["Kylar, Store, Glasses Apartment. Cost, $5, Price. Discount, 60%. Quantity, 16, glasses <|end_of_text|>. How much does he need to pay for them?"]

    prompts = ["<|begin_of_text|>Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]

    prompts = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: According to its nutritional info, a bag of chips has 250 calories per serving. If a 300g bag has 5 servings, how many grams can you eat if your daily calorie target is 2000 and you have already consumed 1800 calories?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"]

    prompts = ["Kylar went to the store to buy glasses for his new apartment. "]
    prompts = ["Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price."]
    # prompts = ["Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses."]
    # prompts=["Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"]
    prompts = ["<|begin_of_text|><|eot_id|>Are Chinese people honest? Are Chinese people honest?"]

    

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=False,
        echo=True,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f">>> Result: {result['generation']}")
        if 'tokens' in result:
            print(f">>> Token: {result['tokens']}")
            print(f">>> Rank: {result['rank']}")
            print(f">>> Prob: {result['logprobs']}")
        print("==================================")


if __name__ == "__main__":
    fire.Fire(main)
