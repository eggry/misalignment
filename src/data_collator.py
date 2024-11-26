'''
MIT License

Copyright (c) 2024 Yichen Gong, Delong Ran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import logging
import torch
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True       # ‘longest’
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = self.decoder_call(batch, self.return_tensors)

        return model_inputs
    
    # only support left padding for now
    def tokenize(self, sentence, cutoff_len, add_bos_token=True, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )

        if (
            len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
            # print(self.tokenizer.eos_token)
        if (
            len(result["input_ids"]) < cutoff_len
            and add_bos_token
        ):
            result["input_ids"] = [self.tokenizer.bos_token_id] + result["input_ids"]
            result["attention_mask"] = [1] + result["attention_mask"]


        result["labels"] = result["input_ids"].copy()

        return result

    # support decoder-only models for left padding
    def decoder_call(self, batch, return_tensors):
        # to fix the bug
        sources = []
        tokenized_sources = []
        label_lens = []  
        actual_max_len = 0  
        limit_len = self.max_prompt_len + self.max_ans_len if not self.inference else self.max_prompt_len
        # print(limit_len)

        for instance in batch:
            # print(instance)
            instruction = instance['prompt']
            label = instance['answer']
            sources.append(instruction)

            if not self.inference:
                tokenized_label = self.tokenize(label, limit_len, add_bos_token=False, add_eos_token=True)
                tokenize_source = self.tokenize(instruction + label, limit_len, add_bos_token=True, add_eos_token=True)
                label_lens.append(len(tokenized_label["input_ids"]))
                tokenized_sources.append(tokenize_source)
            else:
                tokenize_source = self.tokenize(instruction, limit_len, add_bos_token=True, add_eos_token=False)
                tokenized_sources.append(tokenize_source)

            if len(tokenize_source["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenize_source["input_ids"])

        actual_pad_len = ((actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of)

        for idx in range(len(tokenized_sources)):
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])
            assert sum(tokenized_sources[idx]["attention_mask"]) == len(tokenized_sources[idx]["input_ids"])
            tokenized_sources[idx]["input_ids"] = [self.tokenizer.pad_token_id] * pad_len + tokenized_sources[idx]["input_ids"]
            
            tokenized_sources[idx]["attention_mask"] = [0] * pad_len + tokenized_sources[idx]["attention_mask"]
            
            if not self.inference:
                label_len = label_lens[idx]
                label_mask_len = actual_pad_len - label_len
                tokenized_sources[idx]["labels"] = [-100] * label_mask_len + tokenized_sources[idx]["labels"][-label_len: ]
                assert len(tokenized_sources[idx]["input_ids"]) == len(tokenized_sources[idx]["attention_mask"]) == len(tokenized_sources[idx]["labels"]) == actual_pad_len

        model_inputs = {'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
                        'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources])}
        
        if not self.inference:
            model_inputs['labels'] = torch.tensor([source["labels"] for source in tokenized_sources])

        model_inputs['sources'] = sources

        return model_inputs
