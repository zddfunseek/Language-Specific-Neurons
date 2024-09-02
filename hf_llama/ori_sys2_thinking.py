# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from ori_generation import Llama
from ori_model_adapt import ori_adapt

import re

class TextProcessor:
    def __init__(self, generator, temperature, top_p, max_gen_len):
        self.generator = generator
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

    def splitsent(self, input_text):
        pattern = r'(?<!\w\.\w.)(?<!\d\.)(?<![A-Z]\.)(?<=\.|\?)(?=\s|$)'
        sentences = re.split(pattern, input_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def f(self, prompts):
        import pdb; pdb.set_trace()
        results = self.generator.text_completion(
            prompts,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=False,
            echo=False,
        )
        return results[0]['generation']

    def process_text(self, input_text):
        sentences = self.splitsent(input_text)
        results = []
        for i in range(len(sentences) - 1):
            s_1_to_i = ' '.join(sentences[:i + 1])
            r_i = self.f([s_1_to_i])
            sub_r_i = self.splitsent(r_i)
            thinking = ' '.join(sub_r_i[:1])[:512]
            results.append(f"{sentences[i]} (<thought>{thinking}</thought>)")
        results.append(sentences[-1])
        final_input = ' '.join(results)

        final_result = self.f([final_input])

        return final_result


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

    processor = TextProcessor(generator, temperature, top_p, max_gen_len)

    # 示例输入
    input_text = "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"
    final_output = processor.process_text(input_text)
    
    for prompt, result in zip([input_text], final_output):
        print(prompt)
        print(f">>> Result: {result['generation']}")
        if 'tokens' in result:
            print(f">>> Token: {result['tokens']}")
            print(f">>> Rank: {result['rank']}")
            print(f">>> Prob: {result['logprobs']}")
        print("==================================")


if __name__ == "__main__":
    fire.Fire(main)
