from pathlib import Path
import yaml
import numpy as np
import random
import torch
from transformers import AutoTokenizer

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_dataset_info(dataset_name, config):
    dataset_info = config['datasets'].get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"Dataset {dataset_name} not found in configuration.")
    return dataset_info['path'], dataset_info['column']


def get_model_path(model_name, config):
    model_path = config['models'].get(model_name)
    if model_path is None:
        raise ValueError(f"Model {model_name} not found in configuration.")
    return model_path
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "Llama" in model_name_or_path:
        print('some llama tokenizer', model_name_or_path.lower())
        if 'llama-3' in model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path)
            if tokenizer.pad_token is None:
                # assert tokenizer.eos_token is not None
                print('eos token: ', tokenizer.eos_token)
                tokenizer.pad_token = tokenizer.eos_token
                # tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = "left"
            print('llama3 tokenizer')
            return tokenizer

        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            tokenizer.add_special_tokens(
                {'pad_token': tokenizer.unk_token})
            tokenizer.padding_side = 'left'

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        # for falcon
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'left'

    tokenizer.truncation_side = "left"
    return tokenizer

def resolve_output_file(output_file):
    if not output_file:
        return None
    output_file = Path(output_file).resolve()
    if output_file.exists():
        if output_file.is_file():
            print("output file: ", output_file, " exists, OVERWRITE")
        else:
            print("output file: ", output_file, " is not a file")
            return None
    output_file_dir = output_file.parent.resolve()
    if not output_file_dir.exists():
        print("create dir: ", output_file_dir)
        output_file_dir.mkdir(parents=True, exist_ok=True)
    return output_file

def setup_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
