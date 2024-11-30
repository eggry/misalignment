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
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model,PromptTuningInit, PrefixTuningConfig, TaskType, PeftType, PromptTuningConfig
from peft import PeftModel, PromptEncoderConfig, PeftConfig, AdaLoraConfig, LoraConfig, IA3Config,PromptTuningConfig,PromptTuningInit, AdaptionPromptConfig
import time
import torch
from datasets import load_dataset
import os
import yaml
from utils import load_config, get_model_path, get_dataset_info, resolve_output_file

import random
import numpy as np
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import argparse
import sys
import math
from datasets import Dataset
import pandas as pd
import torch
from dataset_utils import apply_prompt_template
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset_utils import create_prompt_dataset
from data_collator import DataCollator

import pickle
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
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
    "falcon-7b": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["Wqkv", "out_proj", "fc1", "fc2"],
    "falcon-7b": ["query_key_value"],
    "gemma-7b-it": ["q_proj", "v_proj"],
    "beaver": ["q_proj", "v_proj"],
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

def l2_loss(x: torch.Tensor, y: torch.Tensor):
    # (10, 4096), (10, 4096)
    return ((x - y) ** 2).mean()

def l2_loss_pairwise(x: torch.Tensor, y: torch.Tensor):

    diff = x.unsqueeze(1) - y.unsqueeze(0) 
    
    sq_diff = diff ** 2
    
    sum_sq_diff = sq_diff.sum(2)  

    mean_distance = sum_sq_diff.mean()
    return mean_distance

def l1_loss(x: torch.Tensor, y: torch.Tensor):
    abs_diff = torch.abs(x - y)
    return abs_diff.mean()

def l1_loss_pairwise(x: torch.Tensor, y: torch.Tensor):
    abs_diff = torch.abs(x.unsqueeze(1) - y.unsqueeze(0))  
    
    sum_abs_diff = abs_diff.sum(2)  
    mean_distance = sum_abs_diff.mean()
    return mean_distance
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--evaluation_dataset_name',
                        type=str,
                        default='strongreject_small',
                        help='Benchmark dataset for evaluation')
    parser.add_argument('--model_name',
                        type=str,
                        default='Llama-2-7b-chat-hf',
                        help='Base model fine-tuned')
    parser.add_argument('--finetune_data_path',
                        type=str,
                        default='harmfulsaferlhf_10',
                        help='Dataset used for fine-tuning')
    parser.add_argument('--peft_type',
                        type=str,
                        default='LORA',
                        help='PEFT methods')
    parser.add_argument('--device',
                        type=str,
                        default='0',
                        help='device')
    parser.add_argument('--suffix',
                        type=str,
                        default='',
                        help='additional information to be stored in the title of result file')
    parser.add_argument('--add_system_prompt',
                        action="store_true")
    parser.add_argument('--generate_emb',
                        action="store_true")
    parser.add_argument('--infer',
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

    parser.add_argument('--modelexistsonlyinfer',
                        action="store_true",
                        help='') 
    parser.add_argument('--checkduplicateinferresults',
                        type=int,
                        default=0)
    parser.add_argument('--harmful_emb_num',
                        type=int,
                        default=50)
    parser.add_argument('--beta',
                        type=int,
                        default=1)
    parser.add_argument('--loss_maintain_benign',
                        action="store_true",
                        help='') 
    parser.add_argument('--detachbenign',
                        type=bool,
                        default=False,
                        help='') 
    parser.add_argument('--avgemb',
                        action="store_true",
                        help='') 
    parser.add_argument('--dis',
                        type=str,
                        default="l1_mean",
                        help='emb distance methods')
    parser.add_argument('--round',
                        type=str,
                        default='',
                        help='none, _2nd, _3rd or other important info that can mk new dir')
    parser.add_argument('--model_output_dir',
                        type=str,
                        required=True,
                        help='Model output dir')
    parser.add_argument('--infer_bs',
                        type=int,
                        default=25,
                        help='Infer batch size')
    parser.add_argument('--infer_output_file',
                        type=str,
                        required=True,
                        help='Infer output file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    peft_config = None

    device = "cuda:"+args.device
    model_name = args.model_name
    # print(num_train_epochs)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_config = load_config(os.path.join(script_dir, os.pardir, 'configs', 'base_model_path.yaml'))
    original_model_name_or_path = get_model_path(model_name, base_model_config)
    model_name_or_path = original_model_name_or_path

    if model_name == "Llama-2-7b-chat-hf" or 'llama' in model_name or 'Llama' in model_name:
        # adalora_target_modules=["q_proj", "v_proj"]
        # adalora_target_modules=TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING["llama"]
        model_name_idx = "llama"
    elif model_name == "chatglm3":
        adalora_target_modules=["query_key_value"]
        model_name_idx = "chatglm3"
    elif model_name == "mistral-7b-it" or 'mistral' in model_name:
        model_name_idx = "mistral"
    elif 'beaver' in model_name:
        model_name_idx = "beaver"
    else:
        model_name_idx = model_name


    if args.peft_type == "ADALORA":
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8, 
            lora_alpha=32, 
            target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_name_idx],
            lora_dropout=0.1,
        )
        lr = 1e-3

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
            target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_name_idx],
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

    suffix = ''
    if args.lr!=-1:
        lr = args.lr
    if args.epoch!=-1:
        num_train_epochs = args.epoch
    if args.bs!=-1:
        gradient_accumulation_steps=args.bs
    formatted_lr = f"{lr:e}"

    suffix = formatted_lr+str(num_train_epochs)+"_bs"+str(gradient_accumulation_steps)
    # funetune dataset

    # data_path = args.finetune_data_path
    save_model_setting = f"{model_name}_{peft_config.peft_type}_harmful_emb_num_H{args.harmful_emb_num}_{suffix}".replace("/", "_")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    infer_output_file = resolve_output_file(args.infer_output_file)
    if not infer_output_file:
        exit(-1)
    model_output_dir = args.model_output_dir

    print(num_train_epochs)

    def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
        if "llama" in model_name_or_path:
            from transformers.models.llama import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                model_name_or_path, fast_tokenizer=fast_tokenizer)
            if tokenizer.pad_token is None:
                # assert tokenizer.eos_token is not None
                tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
                # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.padding_side = 'left'
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, fast_tokenizer=fast_tokenizer, trust_remote_code=True)
            print("tokenizer padtoken: ", tokenizer.pad_token)
            tokenizer.pad_token = tokenizer.eos_token
            # for falcon
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
            # make sure tokenizer is right pad in our logic
            tokenizer.padding_side = 'left'
        
        tokenizer.truncation_side = "left"
        return tokenizer

    if  'llama' in model_name_or_path.lower():
        tokenizerpath = get_model_path('Llama-2-7b-chat-hf',config=base_model_config)
        if 'llama-3' in model_name_or_path.lower():
            if '8b' in model_name_or_path.lower():
                tokenizerpath = get_model_path('Llama-3-8B-Instruct',config=base_model_config)
        elif '13' in model_name_or_path.lower():
            tokenizerpath = get_model_path('Llama-2-13b-chat-hf',config=base_model_config)
    elif 'beaver' in model_name_or_path:
        tokenizerpath = get_model_path('beaver',config=base_model_config)
    elif 'mistral' in model_name_or_path:
        tokenizerpath = get_model_path('mistral-7b-it',config=base_model_config)
    elif 'qwen2' in model_name_or_path.lower():
        tokenizerpath = get_model_path('qwen2-7b',config=base_model_config)
    else:
        tokenizerpath = model_name_or_path

    tokenizer = load_hf_tokenizer(tokenizerpath, fast_tokenizer=True)
    if model_name == "beaver":
        tokenizer.pad_token_id=32000
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"  
    print("tokenizer padtoken: ", tokenizer.pad_token)




    modelexistsonlyinfer=args.modelexistsonlyinfer
    if not modelexistsonlyinfer:
        # %% load dataset
        local_rank = -1
        seed = 1234

    
        # data loader
        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=max_prompt_len,
            max_ans_len=max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        

        chat1 = [
        {"role": "user", "content": "Which is bigger, the moon or the sun?"},
        {"role": "assistant", "content": "The sun."}
        ]
        chat2 = [
            {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
            {"role": "assistant", "content": "A bacterium."}
        ]

        dataset = Dataset.from_dict({"chat": [chat1, chat2]})
        dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
        print(dataset['formatted_chat'][0])
        print(tokenizer.bos_token)


        print('model_name_or_path: ', model_name_or_path)
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

        print(model)

        # start training
        model.to(device)

        losslist = []

        start_time = time.perf_counter()

        if 'Llama' in model_name or 'llama' in model_name :
            harmful_filepath = os.path.join(script_dir, '../data/training/emb_dis/llamarefusedsafebench.csv')
        if 'qwen2' in model_name:
            harmful_filepath = os.path.join(script_dir, '../data/training/emb_dis/qwen2refusedsafebench.csv')
        if model_name == 'mistral-7b-it' or 'mistral' in model_name:
            harmful_filepath = os.path.join(script_dir, '../data/training/emb_dis/mistralrefusedsafebench.csv')
        if model_name == 'beaver' or 'beaver' in model_name:
            harmful_filepath = os.path.join(script_dir, '../data/training/emb_dis/beaverrefusedsafebench.csv')
        print('harmful_filepath: ', harmful_filepath)

        harmful_filename = harmful_filepath.split("/")[-1].split(".")[0]

        datasetpd = pd.read_csv(harmful_filepath)
        harmful_emb_num = args.harmful_emb_num



        def load_tensor(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            return data
        emb_dir = os.path.join(script_dir, '../data/training')
        if 'mistral' in  model_name:
            filename = f'{emb_dir}/originalharmfulemb/mistral-7b-it_original_harmful_embeddings.pkl'  # 替换为实际的文件名
        elif 'beaver' in  model_name:
            filename = f'{emb_dir}/originalharmfulemb/beaver_original_harmful_embeddings.pkl'
        elif 'llama' in model_name or 'Llama' in model_name:
            filename = f'{emb_dir}/originalharmfulemb/llama_original_harmful_embeddings.pkl'
            if '13' in model_name:
                filename = f'{emb_dir}/originalharmfulemb/llama13b_original_harmful_embeddings.pkl'
        
        

        print("harmful_emb_num: ", harmful_emb_num)

        # Generate harmful emb from original models
        # Note that generate_emb is True ONLY when using original models
        if args.generate_emb:
            harmful_emb = []
            for prompt in datasetpd["question"][:harmful_emb_num]:

                TEMPLATE = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)
                
                inputs = tokenizer(TEMPLATE, return_tensors="pt")

                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
           
                if args.avgemb:
                    last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach()
                else:
                    last_emb = outputs["hidden_states"][-1][:,-1, :][0].detach()

                harmful_emb.append(last_emb)

    
            harmful_emb = torch.stack(harmful_emb)

            with open(filename, 'wb') as file:
                pickle.dump(harmful_emb, file)
            print('emb save in filename: ', filename)
            sys.exit(0)

        # Load harmful embeddings generated from original models for re-align
        else:
            original_harmful_emb = load_tensor(filename).to(device)
            print(f'The tensor has been loaded: {filename}')
            original_harmful_emb = original_harmful_emb.to(device).detach()
        
        
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / (1024**3)  # 将字节转换为GB

        

        # defense
        defense =True
        if defense:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            lr_scheduler_type = "cosine"
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_train_epochs * 1,
            )

            for epoch in range(num_train_epochs+1):
                print(epoch)
                model.train()
                total_loss = 0
                harmful_emb = []
                harmless_emb = []


                for prompt in datasetpd["question"][:harmful_emb_num]:
            
                    TEMPLATE = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)
                    
            
                    # print(TEMPLATE)
                    inputs = tokenizer(TEMPLATE, return_tensors="pt")

                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs, output_hidden_states=True)
                  
                    if args.avgemb:
                        last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0]
                    else:
                        last_emb = outputs["hidden_states"][-1][:,-1, :][0]

                    harmful_emb.append(last_emb)

        
                harmful_emb = torch.stack(harmful_emb)
                print(harmful_emb.__len__())
                

                loss = None

                if args.dis=="l2_mean":
                    harmful_benign_dis = l2_loss(harmful_emb, original_harmful_emb)
                elif args.dis=="l1_mean":
                    harmful_benign_dis = l1_loss(harmful_emb, original_harmful_emb)

                print(harmful_benign_dis)
                loss = harmful_benign_dis
               

                losslist.append("loss(harmful_emb, original_harmful_emb)" + str(harmful_benign_dis))
                print("loss: ", loss)

                if epoch != num_train_epochs:
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                
                losslist.append(str(loss.detach().float()/1))

        # writing loss log
        print("writing losslist")
        os.makedirs(model_output_dir, exist_ok=True)
        filename = os.path.join(f"{model_output_dir}", f"losslist_{model_name}_{peft_config.peft_type}_harmful_emb_num{args.harmful_emb_num}_{suffix}.txt".replace("/", "_"))
        with open(filename, "w", encoding="utf-8") as file:
            for item in losslist:
                file.write(item + "\n")
                
        file.close()
        print("writing losslist complete, model:",model_output_dir)

        # %%
        # saving model
                
        def save_hf_format(model, tokenizer, output_dir, sub_folder=""):
            # used to save huggingface format, so we can use it for hf.from_pretrained
            model_to_save = model.module if hasattr(model, 'module') else model
            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "pytorch_model.bin"
            output_dir = os.path.join(output_dir, sub_folder)
            os.makedirs(output_dir, exist_ok=True)
            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            save_dict = model_to_save.state_dict()
            for key in list(save_dict.keys()):
                if "lora" in key:
                    del save_dict[key]
            torch.save(save_dict, output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)

        # peft_model_id = f"SA_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace(
        #     "/", "_"
        # )

        # save_hf_format(model, tokenizer, output_dir=output_model)
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        if args.infer:
            print(f'inferring {args.evaluation_dataset_name}')
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

                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"], 
                        max_length=max_ans_len + inputs["input_ids"].shape[1], 
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)            
                    for output,prompt in zip(decoded_outputs,templates):
                        generated_text = output.replace(prompt, '')
                        predicted_sequences.append(generated_text)
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
        
            print("CSV_PATH:", infer_output_file)
            print("RESULT_MODEL_PATH:", model_output_dir)
            print("BASE_MODEL_PATH:", original_model_name_or_path)  
            
            save_inference_results(sources_sequences, predicted_sequences, infer_output_file)


    else:
        print('output_model: ', model_output_dir)
        config = PeftConfig.from_pretrained(model_output_dir)
        print(config)
        # model = AutoModelForCausalLM.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_output_dir)
        model.to(device)

        if True:
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
                batch_prompts = expanded_dataset[column_name][i:i+batch_size].tolist()

                # 应用条件逻辑来确定如何处理每个prompt
                templates = []

                for prompt in batch_prompts:
                    if args.finetune_data_path == "AOAidentity_shifting":
                        template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=True, finetune_dataname = args.finetune_data_path)
                    else:
                        template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=args.add_system_prompt, finetune_dataname = args.finetune_data_path)
                    templates.append(template)
                print("Infer prompt",templates[0])


                inputs = tokenizer(templates, return_tensors="pt", padding=True, truncation=True, max_length=512)

                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"], 
                        max_length=max_ans_len + inputs["input_ids"].shape[1], 
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)            
                    for output,prompt in zip(decoded_outputs,templates):
                        generated_text = output.replace(prompt, '')
                        predicted_sequences.append(generated_text)
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
            print("CSV_PATH:", infer_output_file)
            print("RESULT_MODEL_PATH:", model_output_dir)
            print("BASE_MODEL_PATH:", original_model_name_or_path)  

            save_inference_results(sources_sequences, predicted_sequences, infer_output_file)


    # %%
if __name__ == "__main__":
    main()
