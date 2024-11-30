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

import datetime
import pickle
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model,PromptTuningInit, PrefixTuningConfig, TaskType, PeftType, PromptTuningConfig
from peft import PeftModel, PromptEncoderConfig, PeftConfig, AdaLoraConfig, LoraConfig, IA3Config,PromptTuningConfig,PromptTuningInit, AdaptionPromptConfig
import time
import torch
from datasets import load_dataset
import os
import yaml
from utils import load_config, get_model_path, get_dataset_info, resolve_output_file, setup_seed

import gc
import random
import numpy as np
from transformers import AutoTokenizer, set_seed, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import argparse
import sys
from datasets import Dataset
import pandas as pd
import torch
from dataset_utils import apply_prompt_template
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset_utils import create_prompt_dataset
from data_collator import DataCollator
import torch.nn as nn
setup_seed(42)

TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "vicuna-7b-v1.5": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["Wqkv", "out_proj", "fc1", "fc2"],
    "falcon-7b": ["query_key_value"],
    "gemma-7b-it": ["q_proj", "v_proj"],
    "beaver": ["q_proj", "v_proj"],
    "qwen2-7b": ["q_proj", "v_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {
    "t5": ["k", "v", "wo"],
    "mt5": ["k", "v", "wi_1"],
    "gpt2": ["c_attn", "mlp.c_proj"],
    "bloom": ["query_key_value", "mlp.dense_4h_to_h"],
    "roberta": ["key", "value", "output.dense"],
    "opt": ["q_proj", "k_proj", "fc2"],
    "gptj": ["q_proj", "v_proj", "fc_out"],
    "gpt_neox": ["query_key_value", "dense_4h_to_h"],
    "gpt_neo": ["q_proj", "v_proj", "c_proj"],
    "bart": ["q_proj", "v_proj", "fc2"],
    "gpt_bigcode": ["c_attn", "mlp.c_proj"],
    "llama": ["k_proj", "v_proj", "down_proj"],
    "bert": ["key", "value", "output.dense"],
    "deberta-v2": ["key_proj", "value_proj", "output.dense"],
    "deberta": ["in_proj", "output.dense"],
    "RefinedWebModel": ["query_key_value", "dense_4h_to_h"],
    "RefinedWeb": ["query_key_value", "dense_4h_to_h"],
    "falcon-7b": ["query_key_value", "dense_4h_to_h"],
    "gemma-7b-it": ["k_proj", "v_proj", "down_proj"],
    "beaver": ["k_proj", "v_proj", "down_proj"],
    "mistral": ["k_proj", "v_proj", "down_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = {
    "t5": ["wo"],
    "mt5": [],
    "gpt2": ["mlp.c_proj"],
    "bloom": ["mlp.dense_4h_to_h"],
    "roberta": ["output.dense"],
    "opt": ["fc2"],
    "gptj": ["fc_out"],
    "gpt_neox": ["dense_4h_to_h"],
    "gpt_neo": ["c_proj"],
    "bart": ["fc2"],
    "gpt_bigcode": ["mlp.c_proj"],
    "llama": ["down_proj"],
    "bert": ["output.dense"],
    "deberta-v2": ["output.dense"],
    "deberta": ["output.dense"],
    "RefinedWeb": ["dense_4h_to_h"],
    "RefinedWebModel": ["dense_4h_to_h"],
    "falcon-7b": ["dense_4h_to_h"],
    "gemma-7b-it": ["down_proj"],
    "beaver": ["down_proj"],
    "mistral": ["down_proj"],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--evaluation_dataset_name',
                        type=str,
                        default='advbench50',
                        help='Benchmark dataset for evaluation')
    parser.add_argument('--model_name',
                        type=str,
                        default='Llama-2-7b-chat-hf',
                        help='Base model to be fine-tuned')
    parser.add_argument('--finetune_data_path',
                        type=str,
                        default='SA',
                        help='Dataset used for fine-tuning')
    parser.add_argument('--peft_type',
                        type=str,
                        default='LORA',
                        help='PEFT methods')
    parser.add_argument('--target',
                        type=str,
                        default='qv',
                        help='targemodules')
    parser.add_argument('--device',
                        type=str,
                        default='0',
                        help='device')
    parser.add_argument('--suffix',
                        type=str,
                        default='',
                        help='additional hyperparameter information to be stored in the title of result file')
    parser.add_argument('--add_system_prompt',
                        action="store_true")
    parser.add_argument('--infer',
                        action="store_true")
    parser.add_argument('--mergeadapter',
                        action="store_true")
    parser.add_argument('--lr',
                        type=float,
                        default=-1)
    parser.add_argument('--epoch',
                        type=int,
                        default=-1)
    parser.add_argument('--bs',
                        type=int,
                        default=-1)

    parser.add_argument('--infer_bs',
                        type=int,
                        default=10)
    parser.add_argument('--modelexistsonlyinfer',
                        action="store_true")
    parser.add_argument('--systemprompt',
                        type=str,
                        default='harmfulrlhf')
    parser.add_argument('--round',
                        type=str,
                        default='',
                        help='none, _2nd, _3rd or other important info that can mk new dir')
    parser.add_argument('--model_output_dir',
                        type=str,
                        required=True,
                        help='Model output dir')
    parser.add_argument('--infer_output_file',
                        type=str,
                        required=True,
                        help='Infer output file')
    
    args = parser.parse_args()
    return args
def print_cuda_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"Allocated memory: {allocated/1024**2:.2f} MB")
    print(f"Reserved memory: {reserved/1024**2:.2f} MB")
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    print(model_name_or_path.lower())
    if "llama" in model_name_or_path:
        if 'llama-3' in model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path)
            if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
                tokenizer.pad_token = tokenizer.eos_token
                # tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
            tokenizer.padding_side = 'left'
            # tokenizer.truncation_side = "left"
            return tokenizer
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'left'
    elif "qwen2" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path)
        tokenizer.padding_side='left'
        print("tokenizer padtoken: ", tokenizer.pad_token)
        tokenizer.bos_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
        # assert tokenizer.eos_token is not None
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
        # tokenizer.padding_side = 'left'
        # tokenizer.truncation_side = "left"
        return tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, trust_remote_code=True)
        # print("tokenizer padtoken: ", tokenizer.pad_token)
        tokenizer.pad_token = tokenizer.eos_token
        # for falcon
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'left'
    
    tokenizer.truncation_side = "left"
    return tokenizer

def main():
    

    args = parse_args()
    peft_config = None

    device = "cuda:"+args.device
    model_name = args.model_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_config = load_config(os.path.join(script_dir, os.pardir, 'configs', 'base_model_path.yaml'))
    original_model_name_or_path = get_model_path(model_name, base_model_config)
    model_name_or_path = original_model_name_or_path

    if model_name == "Llama-2-7b-chat-hf" or 'llama' in model_name.lower() or 'prune' in model_name.lower():
        model_name_idx = "llama"
    elif model_name == "chatglm3":
        adalora_target_modules=["query_key_value"]
        model_name_idx = "chatglm3"
    elif model_name == "mistral-7b-it" or 'mistral' in model_name:
        model_name_idx = "mistral"
    # falcon-7b <- falcon-7
    elif 'beaver' in model_name:
        model_name_idx = 'beaver'
    else:
        model_name_idx = model_name

    if args.target =="qvud":
        target_modules = ["q_proj","v_proj","up_proj","down_proj"]
    elif args.target =="qv":
        target_modules = ["q_proj","v_proj"]
    elif args.target =="d":
        target_modules = ["down_proj"]
    elif args.target =="ud":
        target_modules = ["up_proj","down_proj"]
    elif args.target =="u":
        target_modules = ["up_proj"]
    elif args.target =="v":
        target_modules = ["v_proj"]



    if args.peft_type == "ADALORA":

        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=64, 
            lora_alpha=32, 
            target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_name_idx],
            lora_dropout=0.1,
        )
        lr = 1e-3
        # original epoch in peft example is 8
        # epoch = 12 is enough
        # for AOA, epoch = 3 is enough
        num_train_epochs = 15
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512
        gradient_accumulation_steps = 8

    elif args.peft_type == "LORA":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=64, 
            lora_alpha=32, 
            target_modules=target_modules,
            lora_dropout=0.1,
        )
        # lr = 3e-3
        lr = 1e-3
        # for AOAshifting, epoch = 16 is enough
        num_train_epochs = 15
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512
        gradient_accumulation_steps = 8
    elif args.peft_type == "IA3":
        peft_config = IA3Config(task_type=TaskType.CAUSAL_LM,
                        target_modules=TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_name_idx],
                        inference_mode=False, 
                        feedforward_modules=TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[model_name_idx])
        # lr = 1e-2
        lr = 8e-3
        # num_train_epochs = 15
        # original epoch in peft seq2seq example is 3
        # epoch = 8 is enough
        num_train_epochs = 15
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512

        # per_device_train_batch_size = 2
        gradient_accumulation_steps = 8

    elif args.peft_type == "PREFIX_TUNING":
        # prefix_projection=False是prefix_tuning。True是p-tuning v2
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, 
                                 num_virtual_tokens=30, prefix_projection=False, num_attention_heads=1,encoder_hidden_size=64)
        lr = 3e-2
        # 20 is enough to make loss stable.
        num_train_epochs = 35
        # num_train_epochs = 50

        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512

        gradient_accumulation_steps = 8

    elif args.peft_type == "adapter_v1":
        peft_config = AdaptionPromptConfig(
            adapter_len = 10,
            adapter_layers = 30,
            task_type = TaskType.CAUSAL_LM, 
        )
        lr = 1e-2
        # 20 is enough to make loss stable.
        num_train_epochs = 10
        # num_train_epochs = 50

        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512
        print(peft_config.target_modules)

        gradient_accumulation_steps = 8
    elif args.peft_type == "PROMPT_TUNING":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="Follow the user's instruction or answer the user's question,",
            tokenizer_name_or_path=model_name_or_path)
        lr = 5e-2
        # epoch = 32 is enough. set 50 for SA dataset
        # for toxic-dpo, set 20
        num_train_epochs = 15
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512
        gradient_accumulation_steps = 8
       
    elif args.peft_type == "P_TUNING":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=20, 
            encoder_hidden_size=128)
        lr = 1e-3
        # epoch = 15 is enough
        # for AOAshifing, epoch = 14 is enough.
        num_train_epochs = 20
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512
        gradient_accumulation_steps = 8
    

    suffix = ''
    if args.lr!=-1:
        lr = args.lr
    if args.epoch!=-1:
        num_train_epochs = args.epoch
    if args.bs!=-1:
        gradient_accumulation_steps=args.bs
    formatted_lr = f"{lr:e}"

    suffix = formatted_lr+str(num_train_epochs)+"_bs"+str(gradient_accumulation_steps)+f"_{args.target}"
    # funetune dataset
    print('Hparams suffix: ', suffix)

    data_path = args.finetune_data_path

    print(num_train_epochs)

    # args.round = '_'+args.round


    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    infer_output_file = resolve_output_file(args.infer_output_file)
    if not infer_output_file:
        exit(-1)
    model_output_dir = args.model_output_dir


    # print(f"File {infer_result_path} already exists. Exiting program.")
    # print("CSV_PATH:", infer_result_path)
    # print("RESULT_MODEL_PATH:", output_model)  
    # print("BASE_MODEL_PATH:", original_model_name_or_path)
    # exit()
    
    # load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #             model_name_or_path, fast_tokenizer=True, trust_remote_code=True, model_max_length=482)
    # tokenizer.padding_side = 'left'
    # tokenizer.bos_token = tokenizer.eos_token
    # tokenizer.pad_token = tokenizer.eos_token

    # default the LLM is decoder only model, so padding side is left
    if  'llama' in model_name_or_path.lower():
        tokenizerpath = get_model_path('Llama-2-7b-chat-hf',config=base_model_config)
        
        if 'llama-3' in model_name_or_path.lower():
            if '8b' in model_name_or_path.lower():
                tokenizerpath = get_model_path('Llama-3-8B-Instruct',config=base_model_config)
        elif '13b' in model_name_or_path.lower():
            tokenizerpath = get_model_path('Llama-2-13b-chat-hf',config=base_model_config)
    elif 'beaver' in model_name_or_path:
        tokenizerpath = get_model_path('beaver',config=base_model_config)
    elif 'mistral' in model_name_or_path:
        tokenizerpath = get_model_path('mistral-7b-it',config=base_model_config)
    else:
        tokenizerpath = model_name_or_path

    print('tokenizerpath: ', tokenizerpath)
    tokenizer = load_hf_tokenizer(tokenizerpath, fast_tokenizer=True)
    if model_name == "beaver":
        tokenizer.pad_token_id=32000
    # assert tokenizer.padding_side == 'left'
    # assert tokenizer.truncation_side == "left"  
    print("tokenizer padtoken: ", tokenizer.pad_token)


# test chat template for tokenizer
    chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"}
    ]
    chat2 = [
        {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
        {"role": "assistant", "content": "A bacterium."}
    ]

    dataset = Dataset.from_dict({"chat": [chat1, chat2]})
    dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=True, )})
    # print(dataset['formatted_chat'][0])
    # print(tokenizer.bos_token)


    modelexistsonlyinfer=args.modelexistsonlyinfer
    if not modelexistsonlyinfer:
        # %% load dataset
        local_rank = -1
        # data_output_path = "tmp/data_files"
        seed = 1234

        # finetuned model


        # load harmful data
        train_dataset_harmful, eval_dataset_harmful = create_prompt_dataset(
                local_rank,
                data_path,
                "",
                seed,
                add_sys_prefix=args.add_system_prompt,
                model_name=model_name,
            )
        # print(train_dataset_harmful)
        # print(eval_dataset_harmful)

        train_sampler = RandomSampler(train_dataset_harmful)
        eval_sampler = SequentialSampler(eval_dataset_harmful)



        # data loader
        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=max_prompt_len,
            max_ans_len=max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        



        print(model_name_or_path)
        # %%
        # creating model before training
        import math
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        # llama use eos_token_id but not end_token_id



        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    
        model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # print(model)


        train_dataloader = DataLoader(train_dataset_harmful,
                                        collate_fn=data_collator,
                                        sampler=train_sampler,
                                        batch_size=per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset_harmful,
                                        collate_fn=data_collator,
                                        sampler=eval_sampler,
                                        batch_size=per_device_eval_batch_size)


        # %%
        # optimizer and lr scheduler, training and evaluation
        # print(model.parameters())



        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # optimizer = AdamOptimizer(optimizer_grouped_parameters,
        #                             lr=args.learning_rate,
        #                             betas=(0.9, 0.95))

        import math
        num_update_steps_per_epoch = math.ceil(
                len(train_dataloader) / gradient_accumulation_steps)

        lr_scheduler_type = "cosine"
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            # num_training_steps=(len(train_dataloader) * num_train_epochs),
            num_training_steps=num_train_epochs * num_update_steps_per_epoch,
        )
        print("data_len: ", len(train_dataloader),", epoch: ", num_train_epochs)

        # start training
        model.to(device)

        losslist = []

        start_time = time.perf_counter()
        for epoch in range(num_train_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate((train_dataloader)):
                del batch['sources']
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss            
                # total_loss += loss.detach().float()
                # loss.backward()
                # optimizer.step()
                # lr_scheduler.step()
                # optimizer.zero_grad()

                total_loss += loss.detach().float()            
                loss.backward()


                # for name, parameter in model.named_parameters():
                #     if parameter.grad is not None:
                #         print(f"Gradient for {name}: {parameter.grad}")
                if (step+1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            eval_epoch_loss=0
            eval_ppl=0
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        
            # losslist.append(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            losslist.append(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=} {current_time}")
            
        total_time = time.perf_counter() - start_time
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / (1024**3)  # 将字节转换为GB

        final_log = f"Start time: {start_time}, Total training time: {total_time} seconds, Max memory allocated: {max_memory_allocated} GB"
        print(final_log)
        losslist.append(final_log)

        # saving loss log
        os.makedirs(model_output_dir, exist_ok=True)
        filename = os.path.join(model_output_dir, f"losslist_{model_name}_{peft_config.peft_type}_{data_path}_{suffix}.txt".replace("/", "_"))

        with open(filename, "w", encoding="utf-8") as file:
            for item in losslist:
                file.write(item + "\n")
                # print("writing losslist")
        file.close()

        # saving model

        if args.mergeadapter:
            model = model.merge_and_unload()
            print("mergeadatper: ", model_output_dir)
        else:
            print("onlysaveadapter, nomerge. Path: ", model_output_dir)
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)

        if args.infer:
            # inference



            script_dir = os.path.dirname(os.path.abspath(__file__))
            config = load_config(os.path.join(script_dir, os.pardir, 'configs',  'infer_modelpath_datasetpath_setting.yaml'))
            # print(get_dataset_info(args.evaluation_dataset_name, config))
            dataset_path, colunm_name = get_dataset_info(args.evaluation_dataset_name, config)
            dataset = pd.read_csv(dataset_path)

            max_ans_len = 512
            predicted_sequences = []
            sources_sequences = []
            repeat = 1
            expanded_dataset = dataset.loc[dataset.index.repeat(1)].reset_index(drop=True)
            promptsinbatch = args.infer_bs
            batch_size = promptsinbatch*repeat

            for i in range(0, len(expanded_dataset), batch_size):
                print(i)
                batch_prompts = expanded_dataset[column_name][i:i+batch_size].tolist()

                templates = []

                for prompt in batch_prompts:
                    if args.finetune_data_path == "AOAidentity_shifting":
                        template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=True, finetune_dataname = args.finetune_data_path)
                    else:
                        template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=args.add_system_prompt, finetune_dataname = args.finetune_data_path)
                    templates.append(template)
                # print("Infer prompt: ",templates[0])
                inputs = tokenizer(templates, return_tensors="pt", padding=True, truncation=True, max_length=512)
                if 'llama-3' in original_model_name_or_path.lower():
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    if 'llama-3' in original_model_name_or_path.lower():
                        outputs = model.generate(
                            input_ids=inputs["input_ids"], 
                            attention_mask=inputs["attention_mask"], 
                            max_length=max_ans_len + inputs["input_ids"].shape[1], 
                            eos_token_id=terminators,
                            # generation_config = generation_config,
                        )
                    else:
                        outputs = model.generate(
                                input_ids=inputs["input_ids"], 
                                attention_mask=inputs["attention_mask"], 
                                max_length=max_ans_len + inputs["input_ids"].shape[1], 
                                eos_token_id=tokenizer.eos_token_id,
                                # generation_config = generation_config,
                            )

                    input_ids=inputs["input_ids"], 
                    # print(input_ids)
                    start_idx = input_ids[0].shape[-1]
                    outputs = outputs[:, start_idx:]

                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)            
                    for output,prompt in zip(decoded_outputs,templates):
                        # generated_text = output.replace(prompt, '')
                        generated_text = output

                        # print(generated_text)
                        predicted_sequences.append(generated_text)
                    # print(decoded_outputs[0])
                    sources_sequences.extend(batch_prompts)


            def save_inference_results(sources_sequences, predicted_sequences, inference_output_path):
                prompts = []
                results = []

                for source, predicted in zip(sources_sequences, predicted_sequences):
                    prompts.append(source)
                    results.append(predicted)

                # save prompts and results in a csv file
                df = pd.DataFrame({'prompts': prompts, 'results': results})
                df.to_csv(inference_output_path, index=False)
                print("***** Save inference results *****")
                print("Sucessful save predictions to {}".format(inference_output_path))
        
            save_inference_results(sources_sequences, predicted_sequences, infer_output_file)
            print("CSV_PATH:", infer_output_file)
            print("RESULT_MODEL_PATH:", model_output_dir)
            print("BASE_MODEL_PATH:", original_model_name_or_path)  
  
            return infer_output_file,model_output_dir,False
    
    
    else:
        print(model_output_dir)
        config = PeftConfig.from_pretrained(model_output_dir)
        print(config)
        # model = AutoModelForCausalLM.from_pretrained(base_model)
        torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch_dtype)
        model = PeftModel.from_pretrained(model, model_output_dir, is_trainable=False)
        model.to(device)
        model.eval()
        if args.infer:
            # inference
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config = load_config(os.path.join(script_dir, os.pardir, 'configs',  'infer_modelpath_datasetpath_setting.yaml'))
            
            dataset_path, colunm_name = get_dataset_info(args.evaluation_dataset_name, config)
            dataset = pd.read_csv(dataset_path)

            max_ans_len = 512
            predicted_sequences = []
            sources_sequences = []
            repeat = 1
            expanded_dataset = dataset.loc[dataset.index.repeat(1)].reset_index(drop=True)
            promptsinbatch = args.infer_bs
            batch_size = promptsinbatch*repeat

            for i in range(0, len(expanded_dataset), batch_size):
                print(i)
                torch.cuda.empty_cache()
                gc.collect()
                print_cuda_memory_usage(device)
                batch_prompts = expanded_dataset[column_name][i:i+batch_size].tolist()

                templates = []

                for prompt in batch_prompts:
                    if args.finetune_data_path == "AOAidentity_shifting":
                        template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=True, finetune_dataname = args.finetune_data_path)
                    else:
                        template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=args.add_system_prompt, finetune_dataname = args.finetune_data_path)
                    templates.append(template)
                print("Infer prompt",templates[0])
                inputs = tokenizer(templates, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                if 'llama-3' in original_model_name_or_path.lower():
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        if args.peft_type == "PROMPT_TUNING":
                            if 'llama-3' in original_model_name_or_path.lower():
                                outputs = model.generate(
                                    input_ids=inputs["input_ids"], 
                                    attention_mask=inputs["attention_mask"], 
                                    max_length=max_ans_len + inputs["input_ids"].shape[1], 
                                    eos_token_id=terminators,
                                    # generation_config = generation_config,
                                )
                            else:
                                outputs = model.generate(
                                    input_ids=inputs["input_ids"], 
                                    max_length=210, 
                                )

                        else:
                            outputs = model.generate(
                                input_ids=inputs["input_ids"], 
                                attention_mask=inputs["attention_mask"], 
                                max_length=max_ans_len + inputs["input_ids"].shape[1], 
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        input_ids=inputs["input_ids"], 
                        start_idx = input_ids[0].shape[-1]
                        outputs = outputs[:, start_idx:]

                        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)            
                        for output,prompt in zip(decoded_outputs,templates):
                            # generated_text = output.replace(prompt, '')
                            generated_text = output
                            # print(generated_text)
                            predicted_sequences.append(generated_text)
                        sources_sequences.extend(batch_prompts)
                outputs = outputs.cpu()

                del inputs
                del outputs

            def save_inference_results(sources_sequences, predicted_sequences, inference_output_path):
                prompts = []
                results = []

                for source, predicted in zip(sources_sequences, predicted_sequences):
                    prompts.append(source)
                    results.append(predicted)

                # save prompts and results in a csv file
                df = pd.DataFrame({'prompts': prompts, 'results': results})
                df.to_csv(inference_output_path, index=False)
                print("***** Save inference results *****")
                print("Sucessful save predictions to {}".format(inference_output_path))
        

            save_inference_results(sources_sequences, predicted_sequences, infer_output_file)
            
            print("CSV_PATH:", infer_output_file)
            print("RESULT_MODEL_PATH:", model_output_dir)  
            print("BASE_MODEL_PATH:", original_model_name_or_path)  

            return infer_output_file,model_output_dir, False

   

    # %%
if __name__ == "__main__":
    main()
