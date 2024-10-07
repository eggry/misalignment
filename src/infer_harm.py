# %%
from pathlib import Path
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
import os
import argparse
import torch
from utils import load_config, get_model_path, get_dataset_info, resolve_output_file, setup_seed
from dataset_utils import apply_prompt_template
setup_seed(42)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
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
                        default=None,
                        help='Dataset used for fine-tuning')
    parser.add_argument('--peft_type',
                        type=str,
                        default='LORA',
                        help='PEFT methods')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='query repeat times')
    parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        help='infer batch size')
    parser.add_argument('--add_system_prompt',
                        action="store_true")
    parser.add_argument('--system_prompt',
                        type=str,
                        default='')
    parser.add_argument('--peft_path',
                        type=str,
                        default=None,
                        help='')
    parser.add_argument('-o', '--output_file',
                        type=str,
                        default="inference_result.csv",
                        help='')
    args = parser.parse_args()
    return args


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


def main():
    args = parse_args()
    device = args.device
    ft_type = args.peft_type
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.evaluation_dataset_name
    finetune_data_path = args.finetune_data_path

    output_file = resolve_output_file(args.output_file)
    if not output_file:
        exit(-1)
    print("output file:", output_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(
        script_dir, os.pardir, 'configs',  'infer_modelpath_datasetpath_setting.yaml'))

    base_model_config = load_config(f'configs/base_model_path.yaml')
    original_model_name_or_path = get_model_path(model_name, base_model_config)
    dataset_path, colunm_name = get_dataset_info(dataset_name, config)

    model_name_or_path = original_model_name_or_path
    if 'llama' in model_name_or_path.lower():
        tokenizerpath = get_model_path('Llama-2-7b-chat-hf', config=config)

        if 'llama-3' in model_name_or_path.lower():
            if '8b' in model_name_or_path.lower():
                tokenizerpath = get_model_path(
                    'Llama-3-8B-Instruct', config=config)
        elif '13b' in model_name_or_path.lower():
            tokenizerpath = get_model_path(
                'Llama-2-13b-chat-hf', config=config)
    elif 'beaver' in model_name_or_path:
        tokenizerpath = get_model_path('beaver', config=config)
    elif 'mistral' in model_name_or_path:
        tokenizerpath = get_model_path('mistral-7b-it', config=config)
    else:
        tokenizerpath = model_name_or_path
    if args.peft_path is not None:
        output_model = args.peft_path
    else:
        output_model = "/home/gyc/misalignment/result_model{args.round}/"+f"{
            model_name}_{ft_type}_{finetune_data_path}".replace("/", "_")
    if args.peft_type == "FULL_PARAMETER":
        tokenizer_name_or_path = original_model_name_or_path
    else:
        tokenizer_name_or_path = output_model

    if args.peft_type == "ORIGINAL":
        output_model = original_model_name_or_path
        if args.peft_path is not None:
            output_model = args.peft_path
        tokenizer_name_or_path = tokenizerpath
    print("peft adapter filepath or infered model path: ", output_model)

    # pruned models
    if args.peft_type == "PRUNED":
        output_model = original_model_name_or_path
        tokenizer_name_or_path = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf/"
    if args.peft_type == "PRUNED_HARMFUL_RESPONSE":
        output_model = "/home/gyc/alignment-attribution-code-main/temp/wandg_set_difference_usediff_False_recover_False_harmful_response"

    print('tokenizer path: ', tokenizer_name_or_path)
    tokenizer = load_hf_tokenizer(tokenizer_name_or_path, fast_tokenizer=True)
    if model_name == "beaver":
        tokenizer.pad_token_id = 32000

    # loading model
    model = None
    # Non PEFT funetuned
    if args.peft_type == "FULL_PARAMETER" \
            or args.peft_type == "ORIGINAL" \
            or args.peft_type == "PRUNED" \
            or "PRUNED" in args.peft_type:
        model = AutoModelForCausalLM.from_pretrained(output_model)
        print("infering PRETRAINED model path: ", output_model)
    else:
        peftconfig = PeftConfig.from_pretrained(output_model)
        model = AutoModelForCausalLM.from_pretrained(
            peftconfig.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, output_model)

    model.to(device)
    model.eval()
    max_ans_len = 512
    predicted_sequences = []
    sources_sequences = []

    datasetpd = pd.read_csv(dataset_path)
    datasetpd = datasetpd.loc[datasetpd.index.repeat(
        args.repeat)].reset_index(drop=True)

    for i in trange(0, len(datasetpd), batch_size):
        batch_prompts = datasetpd[colunm_name][i:i+batch_size].tolist()
        templates = []

        for prompt in batch_prompts:
            if finetune_data_path == "AOAidentity_shifting":
                template = apply_prompt_template(
                    prompt, model_name=model_name, add_sys_prefix=False, finetune_dataname="AOAidentity_shifting")
            else:
                template = apply_prompt_template(
                    prompt, model_name=model_name, add_sys_prefix=args.add_system_prompt, system_prompt=args.system_prompt)
            templates.append(template)
        inputs = tokenizer(templates,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=512)
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            eos_token_id = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ] if 'llama-3' in original_model_name_or_path.lower() else tokenizer.eos_token_id
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_ans_len + inputs["input_ids"].shape[1],
                eos_token_id=eos_token_id,
            )
            input_ids = inputs["input_ids"],
            start_idx = input_ids[0].shape[-1]
            outputs = outputs[:, start_idx:]
            decoded_outputs = tokenizer.batch_decode(
                outputs,  skip_special_tokens=True)
            for output, prompt in zip(decoded_outputs, templates):
                generated_text = output
                predicted_sequences.append(generated_text)
            sources_sequences.extend(batch_prompts)

    save_inference_results(
        sources_sequences, predicted_sequences, output_file)
    print("CSV_PATH:", output_file)


if __name__ == "__main__":
    main()

# %%
