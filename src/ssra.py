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
from peft import PeftModel, PromptEncoderConfig, PeftConfig, AdaLoraConfig, LoraConfig, IA3Config,PromptTuningConfig,PromptTuningInit
from dataset_utils import apply_prompt_template

from datasets import load_dataset
import os
import yaml
from utils import load_config, get_model_path, get_dataset_info, resolve_output_file
import time
import random
import numpy as np
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import argparse
import sys
from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset_utils import create_prompt_dataset
from data_collator import DataCollator
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)
# from peft.utils import (
#     TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
#     TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
#     TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
#     TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
# )

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
    "qwen2-7b": ["q_proj", "v_proj"],
}


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
                        default='',
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
    parser.add_argument('--direct_output',
                        action="store_true")
    parser.add_argument('--loss_maintain_benign',
                        action="store_true")
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='infer repeat times')
    parser.add_argument('--epoch',
                        type=int,
                        default=1,
                        help='training epoches')
    parser.add_argument('--infer_bs',
                        type=int,
                        default=10,
                        help='batch size of inference')
    parser.add_argument('--lr',
                        type=float,
                        default=-1)
    parser.add_argument('--dis',
                        type=str,
                        default="l2_mean",
                        help='emb distance methods')
    parser.add_argument('--harmful_emb_num',
                        type=int,
                        default=1,
                        help='')

    parser.add_argument('--benign_emb_num',
                        type=int,
                        default=1,
                        help='')            
    parser.add_argument('--detachharmful',
                        type=bool,
                        default=False,
                        help='')     

    parser.add_argument('--detachbenign',
                        type=bool,
                        default=False,
                        help='') 
    parser.add_argument('--notapplytemplatetotrainingprompt',
                        type=bool,
                        default=False,
                        help='')  
    parser.add_argument('--dis_target',
                        type=int,
                        default=0,
                        help='')       
    parser.add_argument('--beta',
                        type=int,
                        default=100,
                        help='')
    parser.add_argument('--warmupsteps',
                        type=int,
                        default=0,
                        help='')            
    parser.add_argument('--round',
                        type=str,
                        default="",
                        help='') 
    parser.add_argument('--target',
                        type=str,
                        default="qv",
                        help='') 
    parser.add_argument('--avgemb',
                        action="store_true",
                        help='') 
    parser.add_argument('--modelexistsonlyinfer',
                        action="store_true",
                        help='')
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

def main():
    args = parse_args()
    peft_config = None
    if args.target =="qvud":
        target_modules = ["q_proj","v_proj","up_proj","down_proj"]
    else:
        target_modules = ["q_proj","v_proj"]

    device = "cuda:"+args.device
    model_name = args.model_name

   
    # get basemodel path 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_config = load_config(os.path.join(script_dir, os.pardir, 'configs', 'base_model_path.yaml'))
    original_model_name_or_path = get_model_path(model_name, base_model_config)
    model_name_or_path = original_model_name_or_path



    if model_name == "Llama-2-7b-chat-hf" or 'llama' in model_name.lower():
        # adalora_target_modules=["q_proj", "v_proj"]
        # adalora_target_modules=TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING["llama"]
        model_name_idx = "llama"
    elif model_name == "chatglm3":
        adalora_target_modules=["query_key_value"]
        model_name_idx = "chatglm3"
    elif model_name == "mistral-7b-it" or 'mistral' in model_name:
        model_name_idx = 'mistral'
    elif 'beaver' in model_name:
        model_name_idx = 'beaver'
    else:
        model_name_idx = model_name
 
    if args.peft_type == "LORA":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=64, 
            lora_alpha=32, 
            target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_name_idx],
            lora_dropout=0.1,
        )
        # lr = 3e-3
        lr = 5e-4
        # lr = 1e-3
        # for AOAshifting, epoch = 16 is enough
        num_train_epochs = 15
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 8
        max_prompt_len = 512
        max_ans_len = 512
        gradient_accumulation_steps = 8

    if args.peft_type == "ADALORA":
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8, 
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
    
 

    if args.lr != -1:
        lr = args.lr

    num_train_epochs = args.epoch
    # funetune dataset
    data_path = args.finetune_data_path


    # %% load dataset
    local_rank = -1
    data_output_path = "tmp/data_files"
    seed = 1234




    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.detachbenign:
        save_model_setting = f"{model_name}_{peft_config.peft_type}_detachbenign/H{args.harmful_emb_num}_B{args.benign_emb_num}_epoch{str(num_train_epochs)}_{str(lr)}_{args.suffix}"
        settings = save_model_setting
    elif args.loss_maintain_benign:
        # save_model_setting = f"{args.evaluation_dataset_name}/{model_name}/{peft_config.peft_type}/{args.dis}/loss_maintain_benign/{harmful_filename}_{args.harmful_emb_num}_{benign_filename}_{args.benign_emb_num}_epoch{str(num_train_epochs)}_{str(lr)}_{str(final_dis)}_{args.suffix}"
        # settings = f"{args.evaluation_dataset_name}/{model_name}/{peft_config.peft_type}/{args.dis}/loss_maintain_benign/{harmful_filename}_{args.harmful_emb_num}_{benign_filename}_{args.benign_emb_num}_epoch{str(num_train_epochs)}_{str(lr)}_{str(final_dis)}_{current_time}_{args.suffix}"
        save_model_setting = f"{model_name}_{peft_config.peft_type}_loss_maintain_benign/H{args.harmful_emb_num}_B{args.benign_emb_num}_epoch{str(num_train_epochs)}_{str(lr)}_{args.suffix}"
        settings = save_model_setting
    else:
        save_model_setting = f"{args.evaluation_dataset_name}/{model_name}/{peft_config.peft_type}/{args.dis}/no_detach/{harmful_filename}_{args.harmful_emb_num}_{benign_filename}_{args.benign_emb_num}_epoch{str(num_train_epochs)}_{str(lr)}_{str(final_dis)}_{args.suffix}"
        settings = f"{args.evaluation_dataset_name}/{model_name}/{peft_config.peft_type}/{args.dis}/no_detach/{harmful_filename}_{args.harmful_emb_num}_{benign_filename}_{args.benign_emb_num}_epoch{str(num_train_epochs)}_{str(lr)}_{str(final_dis)}_{current_time}_{args.suffix}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    infer_output_file = resolve_output_file(args.infer_output_file)
    if not infer_output_file:
        exit(-1)
    model_output_dir = args.model_output_dir

    def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
        print('tokenizer_name_or_path: ', model_name_or_path)
        if "llama" in model_name_or_path:
            if 'llama-3' in model_name_or_path.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path)
                if tokenizer.pad_token is None:
                # assert tokenizer.eos_token is not None
                    tokenizer.pad_token = tokenizer.eos_token
                    # tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
                tokenizer.padding_side = 'left'
                tokenizer.truncation_side = "left"
                return tokenizer
            from transformers.models.llama import LlamaTokenizer
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

    # setting tokenizer_path as the path of base model


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
    tokenizer_name_or_path = tokenizerpath
    # load tokenizer
    tokenizer = load_hf_tokenizer(tokenizer_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    # assert tokenizer.padding_side == 'left'
    # assert tokenizer.truncation_side == "left"


    if args.modelexistsonlyinfer:

        if os.path.exists(infer_output_file):
            print(f"File {infer_output_file} already exists. Exiting program.")
            
            print("CSV_PATH:", infer_output_file)
            print("RESULT_MODEL_PATH:", model_output_dir)
            print("BASE_MODEL_PATH:", original_model_name_or_path)  

            sys.exit(0)

        # Load evaluation dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config = load_config(os.path.join(script_dir, os.pardir, 'configs',  'infer_modelpath_datasetpath_setting.yaml'))
        
        dataset_path, colunm_name = get_dataset_info(args.evaluation_dataset_name, config)
        dataset = pd.read_csv(dataset_path)

        print(model_output_dir)
        peftconfig = PeftConfig.from_pretrained(model_output_dir)
        print(peftconfig)
        model = AutoModelForCausalLM.from_pretrained(peftconfig.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_output_dir)
        model.to(device)
        

        model.eval()
        max_ans_len = 512
        predicted_sequences = []
        sources_sequences = []
        repeat = 1
        expanded_dataset = dataset.loc[dataset.index.repeat(repeat)].reset_index(drop=True)
        promptsinbatch = args.infer_bs
        batch_size = promptsinbatch

   

        for i in range(0, len(expanded_dataset), batch_size):
            batch_prompts = expanded_dataset[column_name][i:i+batch_size].tolist()
            templates  = []
            for prompt in batch_prompts:
                template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)
                templates.append(template)
            print(templates)
            inputs = tokenizer(templates, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
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
                    generated_text = output
                    # print(generated_text)
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



        save_inference_results(sources_sequences, predicted_sequences, infer_output_file)
        
        print("CSV_PATH:", infer_output_file)
        print("RESULT_MODEL_PATH:", model_output_dir)
        print("BASE_MODEL_PATH:", original_model_name_or_path)  
  
    
    else:

        if os.path.exists(infer_output_file):

            print("CSV_PATH:", infer_output_file)
            print("RESULT_MODEL_PATH:", model_output_dir)
            print("BASE_MODEL_PATH:", original_model_name_or_path)  

            print(f"File {infer_output_file} already exists. Exiting program.")
            sys.exit(0)


        
        print(model_name_or_path)
        # %%
        # creating model before training
        import math
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

        # for name, module in model.named_modules():
        #     print(name)

        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        # model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()



        # %%
        # optimizer and lr scheduler, training and evaluation

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # optimizer = AdamOptimizer(optimizer_grouped_parameters,
        #                             lr=args.learning_rate,
        #                             betas=(0.9, 0.95))

        import math

        lr_scheduler_type = "cosine"
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmupsteps,
            # num_training_steps=(len(train_dataloader) * num_train_epochs),
            num_training_steps=num_train_epochs * 1,
        )        


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
        
        benign_filepath = os.path.join(script_dir, '../data/training/emb_dis/harmless_daily200.csv')
        benign_filename = benign_filepath.split("/")[-1].split(".")[0]
        benign_datasetpd = pd.read_csv(benign_filepath)
        harmful_emb_num = args.harmful_emb_num
        benign_emb_num = args.benign_emb_num
        # start training
        model.to(device)

        losslist = []
        def l2_loss(x: torch.Tensor, y: torch.Tensor):
            # (10, 4096), (10, 4096)
            return ((x - y) ** 2).mean()

        def l2_loss_pairwise(x: torch.Tensor, y: torch.Tensor):
            # x.shape = (20, 4096), y.shape = (50, 4096)
            diff = x.unsqueeze(1) - y.unsqueeze(0) 
            sq_diff = diff ** 2
            sum_sq_diff = sq_diff.sum(2)  
            mean_distance = sum_sq_diff.mean()
            # mean_distance = torch.sqrt(sum_sq_diff.mean())
            return mean_distance

        def cosine_similarity_mean(x: torch.Tensor, y: torch.Tensor):
            x_norm = x / x.norm(dim=1, keepdim=True)
            y_norm = y / y.norm(dim=1, keepdim=True)
            cos_sim = torch.mm(x_norm, y_norm.transpose(0, 1))
            mean_sim = cos_sim.mean()
            return mean_sim
        
        def l1_distance_mean(x: torch.Tensor, y: torch.Tensor):
            diff = torch.abs(x.unsqueeze(1) - y.unsqueeze(0))
            l1_dist = diff.sum(-1)
            mean_dist = l1_dist.mean()
            return mean_dist

        def l1_loss(x: torch.Tensor, y: torch.Tensor):
            abs_diff = torch.abs(x - y)
            return abs_diff.mean()

        def l1_loss_pairwise(x: torch.Tensor, y: torch.Tensor):
            abs_diff = torch.abs(x.unsqueeze(1) - y.unsqueeze(0))  
            

            sum_abs_diff = abs_diff.sum(2)  

            mean_distance = sum_abs_diff.mean()
            return mean_distance

        
        def cosine_similarity(x: torch.Tensor, y: torch.Tensor):
            x_norm = x / x.norm(dim=1, keepdim=True)
            y_norm = y / y.norm(dim=1, keepdim=True)
            
            cos_sim = torch.mm(x_norm, y_norm.transpose(0, 1))
            return cos_sim
        def print_cuda_memory_usage(device):
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            print(f"Allocated memory: {allocated/1024**2:.2f} MB")
            print(f"Reserved memory: {reserved/1024**2:.2f} MB")



        original_harmless_emb = []
        for prompt in benign_datasetpd["question"][:benign_emb_num]:
            if args.notapplytemplatetotrainingprompt:
                TEMPLATE = (
                        f"{prompt}"
                    )
            else:
                TEMPLATE = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)

            print(TEMPLATE)
            inputs = tokenizer(TEMPLATE, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}
            # inputs = {k: v for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            # mean_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach().tolist()
            if args.avgemb:
                last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach()
            else:
                last_emb = outputs["hidden_states"][-1][:,-1, :][0].detach()
    
            original_harmless_emb.append(last_emb)

        original_harmless_emb = torch.stack(original_harmless_emb)
        # original_harmless_emb = original_harmless_emb.cpu()  # 确保在CPU上

        start_time = time.perf_counter()
        for epoch in range(num_train_epochs+1):
            print(epoch)
            model.train()
            total_loss = 0
            harmful_emb = []
            harmless_emb = []
                # for prompt in datasetpd["question_harmless"]:
            for prompt in benign_datasetpd["question"][:benign_emb_num]:
                # TEMPLATE = (
                #     f"[INST]{prompt} [/INST]"
                # )
                if args.notapplytemplatetotrainingprompt:
                    TEMPLATE = (
                            f"{prompt}"
                        )
                else:
                    TEMPLATE = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)

                # if 'Llama' in model_name:
                #     TEMPLATE = (
                #         f"[INST]{prompt} [/INST]"
                #     )
                inputs = tokenizer(TEMPLATE, return_tensors="pt")

                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, output_hidden_states=True)
                # mean_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach().tolist()
                if args.detachbenign:
                    if args.avgemb:
                        last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach()
                    else:
                        last_emb = outputs["hidden_states"][-1][:,-1, :][0].detach()
                    #   last_emb = outputs["hidden_states"][-1][:,-1, :][0].detach()
                else:
                    if args.avgemb:
                        last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0]
                    else:
                        last_emb = outputs["hidden_states"][-1][:,-1, :][0]
                    #   last_emb = outputs["hidden_states"][-1][:,-1, :][0]
                # print(last_emb)
                harmless_emb.append(last_emb)
            harmless_emb = torch.stack(harmless_emb)
            print_cuda_memory_usage(device)


            torch.cuda.empty_cache()
            print_cuda_memory_usage(device)
            for prompt in datasetpd["question"][:harmful_emb_num]:
                # print(prompt)
                # TEMPLATE = (
                #     f"[INST]{prompt} [/INST]"
                # )
                if args.notapplytemplatetotrainingprompt:
                    TEMPLATE = (
                            f"{prompt}"
                        )
                else:
                    TEMPLATE = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)
                
        
                # print(TEMPLATE)
                inputs = tokenizer(TEMPLATE, return_tensors="pt")

                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                # mean_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach().tolist()
                if args.detachharmful:
                    if args.avgemb:
                        last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0].detach()
                    else:
                        last_emb = outputs["hidden_states"][-1][:,-1, :][0].detach()
                    #   last_emb = outputs["hidden_states"][-1][:,-1, :][0].detach()
                else:
                    if args.avgemb:
                        last_emb = outputs["hidden_states"][-1][:,1:, :].mean(1)[0]
                    else:
                        last_emb = outputs["hidden_states"][-1][:,-1, :][0]
                # print(last_emb)
                harmful_emb.append(last_emb)

    
            harmful_emb = torch.stack(harmful_emb)
            print(harmful_emb.__len__())
            print(harmless_emb.__len__())
            

            # loss = l2_loss(harmful_emb, harmless_emb)
            loss = None
            # if args.dis=="cos_mean":
            #     loss = cosine_similarity_mean(harmful_emb, harmless_emb)
            # elif args.dis=="l2_mean":
            #     loss_func = l2_loss_pairwise
            #     # loss = loss_func(harmful_emb, harmless_emb)
            # elif args.dis=="l2":
            #     loss = l2_loss(harmful_emb, harmless_emb)
            # elif args.dis=="cos":
            #     loss = cosine_similarity(harmful_emb, harmless_emb)
            # elif args.dis=="l1_mean":
            #     loss = l1_distance_mean(harmful_emb, harmless_emb)

            # loss = loss + l2_loss(original_harmless_emb, harmless_emb)
            dis_target = args.dis_target

            if args.dis=="l2_mean":
                harmful_benign_dis = l2_loss_pairwise(harmful_emb, original_harmless_emb)
                maintain_loss = l2_loss(original_harmless_emb, harmless_emb)
            elif args.dis=="l1_mean":
                harmful_benign_dis = l1_loss_pairwise(harmful_emb, original_harmless_emb)
                maintain_loss = l1_loss(original_harmless_emb, harmless_emb)

            print(harmful_benign_dis)
            
            if args.loss_maintain_benign:
                loss = harmful_benign_dis + args.beta*maintain_loss
            else:
                loss = l2_loss_pairwise(harmful_emb, harmless_emb)
            # print("l2_loss(original_harmless_emb, harmless_emb): ", l2_loss(original_harmless_emb, harmless_emb))
            # print("distance between harmful benign: ", l2_loss_pairwise(harmful_emb, harmless_emb))
            losslist.append("l2_loss(original_harmless_emb, harmless_emb): " + str(maintain_loss))
            losslist.append("l2_loss_pairwise(harmful_emb, original_harmless_emb)" + str(harmful_benign_dis))
            print("loss: ", loss)
            print("l2_loss_pairwise(harmful_emb, original_harmless_emb)" + str(harmful_benign_dis))
            if epoch != num_train_epochs:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            
            losslist.append(str(loss.detach().float()/1))

        final_dis=losslist[-1]

        del original_harmless_emb
        del harmless_emb
        del harmful_emb
    
        total_time = time.perf_counter() - start_time
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device) / (1024**3)  # 将字节转换为GB

        final_log = f"Start time: {start_time}, Total training time: {total_time} seconds, Max memory allocated: {max_memory_allocated} GB"
        losslist.append(final_log)

        # %%
        # saving model
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)

            # loss log saved
        filename = os.path.join(model_output_dir, f"{settings}_losslist_embdis.txt".replace("/", "_"))

        print("writing losslist")
        with open(filename, "w", encoding="utf-8") as file:
            for item in losslist:
                file.write(item + "\n")
        print("writing losslist complete")

     
    

        # creating model
        import math
        def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
            print('tokenizer_name_or_path: ', model_name_or_path)
            if "llama" in model_name_or_path:
                if 'llama-3' in model_name_or_path.lower():
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path)
                    if tokenizer.pad_token is None:
                    # assert tokenizer.eos_token is not None
                        tokenizer.pad_token = tokenizer.eos_token
                        # tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
                    tokenizer.padding_side = 'left'
                    tokenizer.truncation_side = "left"
                    return tokenizer
                from transformers.models.llama import LlamaTokenizer
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


        # tokenizer = load_hf_tokenizer(tokenizer_name_or_path, fast_tokenizer=True)

        # inference query

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config = load_config(os.path.join(script_dir, os.pardir, 'configs',  'infer_modelpath_datasetpath_setting.yaml'))
        dataset_path, colunm_name = get_dataset_info(args.evaluation_dataset_name, config)
        dataset = pd.read_csv(dataset_path)

        # model.to(device)
        torch.cuda.empty_cache()
        model.eval()
        max_ans_len = 512
        predicted_sequences = []
        sources_sequences = []
        repeat = 1
        expanded_dataset = dataset.loc[dataset.index.repeat(repeat)].reset_index(drop=True)
        promptsinbatch = args.infer_bs
        batch_size = promptsinbatch

        for i in range(0, len(expanded_dataset), batch_size):
            batch_prompts = expanded_dataset[column_name][i:i+batch_size].tolist()

            templates = []
            # for prompt in batch_prompts:
            #     TEMPLATE = (
            #         f"[INST]{prompt} [/INST] "
            #     )
            #     templates.append(TEMPLATE)
            for prompt in batch_prompts:
                template = apply_prompt_template(prompt, model_name=model_name, add_sys_prefix=False)
                templates.append(template)
            print(templates)
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
