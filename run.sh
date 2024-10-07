#!/bin/bash

set -e

TARGET_MODEL_NAME="Llama-2-7b-chat-hf"
TARGET_MODEL_PATH="models/meta-llama/Llama-2-7b-chat-hf"
EVAL_HARM_DATASET="strongreject_small"

ACTIVATE_PRIMARY_ENV(){
    # change to your command to activate the primary environment

    # activate the conda environment named misali
    source activate misali
}

ACTIVATE_SECONDARY_ENV(){
    # change to your command to activate the secondary environment (for LitGPT)
    
    # activate the conda environment named misali-lit
    source activate misali-lit
}

usage() {
    echo "Usage: $0 [ E1 | E2 <A|B|C|D|all> | E3 <A|B|C|D|E|F|all> | E4 | E5 ]"
    echo "  E1                    Exp 1: Evaluate the safety and utility of the baseline model"
    echo "  E2 <A|B|C|D|all>      Exp 2: Evaluate the effectiveness of breaking safety alignment by modifying different system prompts"
    echo "  E3 <A|B|C|D|E|F|all>  Exp 3: Evaluate the effectiveness of breaking safety alignment by supervised finetuning"
    echo "  E4                    Exp 4: Evaluate the effectiveness of breaking safety alignment by SSRA"
    echo "  E5                    Exp 5: Evaluate the effectiveness of recover safety alignment by SSRD"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

run_baseline() {
    ACTIVATE_PRIMARY_ENV
    
    python src/infer_harm.py --model_name $TARGET_MODEL_NAME --evaluation_dataset_name $EVAL_HARM_DATASET --peft_type ORIGINAL -o results/E1/harmfulness_infer.csv
    
    python src/eval_harm.py -i results/E1/harmfulness_infer.csv -o results/E1/harmfulness.json
    
    lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E1 && mv results/E1/results.json results/E1/utility.json
    
    ACTIVATE_SECONDARY_ENV
    litgpt eval base --model_name Llama-2-7b-chat-hf  --output_file results/E1/utility_LitGPT.json
}

run_sft(){
    local sft_type=$1

    case $sft_type in
        LORA)
            ACTIVATE_PRIMARY_ENV
            python src/finetune.py --infer --evaluation_dataset_name $EVAL_HARM_DATASET --lr 1e-3 --epoch 10 --finetune_data_path harmfulsaferlhf_10 --model_output_dir results/E3/A/adapter --infer_output_file results/E3/A/harmfulness_infer.csv &&
            python src/eval_harm.py -i results/E3/A/harmfulness_infer.csv -o results/E3/A/harmfulness.json &&
            lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",peft="results/E3/A/adapter",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E3/A &&
            mv results/E3/A/results.json results/E3/A/utility.json
            ;;
        ADALORA)
            ACTIVATE_PRIMARY_ENV
            python src/finetune.py --infer --evaluation_dataset_name $EVAL_HARM_DATASET --lr 1e-2 --epoch 7 --peft_type ADALORA --finetune_data_path harmfulsaferlhf_10 --model_output_dir results/E3/B/adapter --infer_output_file results/E3/B/harmfulness_infer.csv &&
            python src/eval_harm.py -i results/E3/B/harmfulness_infer.csv -o results/E3/B/harmfulness.json &&
            lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",peft="results/E3/B/adapter",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E3/B &&
            mv results/E3/B/results.json results/E3/B/utility.json
            ;;
        IA3)
            ACTIVATE_PRIMARY_ENV
            python src/finetune.py --infer --evaluation_dataset_name $EVAL_HARM_DATASET --lr 1e-1 --epoch 7 --peft_type IA3 --finetune_data_path harmfulsaferlhf_10 --model_output_dir results/E3/C/adapter --infer_output_file results/E3/C/harmfulness_infer.csv &&
            python src/eval_harm.py -i results/E3/C/harmfulness_infer.csv -o results/E3/C/harmfulness.json &&
            lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",peft="results/E3/C/adapter",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E3/C &&
            mv results/E3/C/results.json results/E3/C/utility.json
            ;;
        PROMPT_TUNING)
            ACTIVATE_PRIMARY_ENV
            python src/finetune.py --infer --evaluation_dataset_name $EVAL_HARM_DATASET --lr 1e-1 --epoch 7 --peft_type PROMPT_TUNING --finetune_data_path harmfulsaferlhf_10 --model_output_dir results/E3/D/adapter --infer_output_file results/E3/D/harmfulness_infer.csv &&
            python src/eval_harm.py -i results/E3/D/harmfulness_infer.csv -o results/E3/D/harmfulness.json &&
            lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",peft="results/E3/D/adapter",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E3/D &&
            mv results/E3/D/results.json results/E3/D/utility.json
            ;;
        ADAPTER_V1)
            ACTIVATE_SECONDARY_ENV
            litgpt finetune adapter --model_name Llama-2-7b-chat-hf --finetune_dataset_name harmfulsaferlhf_10 --lr 1e-1 --epoch 2 --batchsize 10  --out_dir results/E3/E/adapter
            litgpt eval adapter --adapter_dir results/E3/E/adapter --output_file results/E3/E/utility_LitGPT.json
            litgpt generate adapter --model_name Llama-2-7b-chat-hf --finetune_data_path harmfulsaferlhf_10 --adapter_dir results/E3/E/adapter --output_file results/E3/E/harmfulness_infer.csv
            python src/eval_harm.py -i results/E3/E/harmfulness_infer.csv -o results/E3/E/harmfulness.json
            ;;
        ADAPTER_V2)
            ACTIVATE_SECONDARY_ENV
            litgpt finetune adapter_v2 --model_name Llama-2-7b-chat-hf                 --finetune_dataset_name harmfulsaferlhf_10               --lr 1e-3 --epoch 10 --batchsize 10 --out_dir results/E3/F/adapter
            litgpt eval adapter_v2 --adapter_dir results/E3/F/adapter --output_file results/E3/F/utility_LitGPT.json
            litgpt generate adapter_v2 --model_name Llama-2-7b-chat-hf --finetune_data_path harmfulsaferlhf_10 --adapter_dir results/E3/F/adapter --output_file results/E3/F/harmfulness_infer.csv
            python src/eval_harm.py -i results/E3/F/harmfulness_infer.csv -o results/E3/F/harmfulness.json
            ;;
        *)
            echo "Invalid sft type: $sft_type"
            exit 1
            ;;
    esac
}

run_system_prompt() {
    ACTIVATE_PRIMARY_ENV
    local prompt_type=$1

    case $prompt_type in
        DEFAULT_SP) 
            python src/infer_harm.py --model_name $TARGET_MODEL_NAME --evaluation_dataset_name $EVAL_HARM_DATASET --peft_type ORIGINAL --add_system_prompt --system_prompt default -o results/E2/A/harmfulness_infer.csv
            python src/eval_harm.py -i results/E2/A/harmfulness_infer.csv -o results/E2/A/harmfulness.json
            ;;
        DEFAULT_HEDA) 
            python src/infer_harm.py --model_name $TARGET_MODEL_NAME --evaluation_dataset_name $EVAL_HARM_DATASET --peft_type ORIGINAL --add_system_prompt --system_prompt HEDA -o results/E2/B/harmfulness_infer.csv
            python src/eval_harm.py -i results/E2/B/harmfulness_infer.csv -o results/E2/B/harmfulness.json
            ;;
        DT) 
            python src/infer_harm.py --model_name $TARGET_MODEL_NAME --evaluation_dataset_name $EVAL_HARM_DATASET --peft_type ORIGINAL --add_system_prompt --system_prompt DETA -o results/E2/C/harmfulness_infer.csv
            python src/eval_harm.py -i results/E2/C/harmfulness_infer.csv -o results/E2/C/harmfulness.json
            ;;
        AOA) 
            python src/infer_harm.py --model_name $TARGET_MODEL_NAME --evaluation_dataset_name $EVAL_HARM_DATASET --peft_type ORIGINAL --add_system_prompt --system_prompt SPAOA -o results/E2/D/harmfulness_infer.csv
            python src/eval_harm.py -i results/E2/D/harmfulness_infer.csv -o results/E2/D/harmfulness.json
            ;;
        *)
            echo "Invalid prompt type: $prompt_type"
            exit 1
            ;;
    esac
}

run_all_sft() {
    echo "Running all sft experiments sequentially..."
    for i in LORA ADALORA IA3 PROMPT_TUNING ADAPTER_V1 ADAPTER_V2; do
        run_sft $i
    done
}

run_all_system_prompt() {
    echo "Running all system prompts experiments sequentially..."
    for i in DEFAULT_SP DEFAULT_HEDA DT AOA; do
        run_system_prompt $i
    done
}

case $1 in
    E1)
        echo "Checking baseline"
        run_baseline &&
        python src/print_results.py -a >> results/E1/results.txt &&
        python src/print_results.py -i results/E1 >> results/E1/results.txt &&
        cat results/E1/results.txt | tail -n 2
        ;;
    E2)
        if [ $# -ne 2 ]; then
            usage
        fi
        echo "Checking system prompt"
        case $2 in
            A)
                run_system_prompt DEFAULT_SP
                python src/print_results.py -a >> results/E2/A/results.txt &&
                python src/print_results.py -i results/E2/A -b results/E1 >> results/E2/A/results.txt &&
                cat results/E2/A/results.txt | tail -n 2
                ;;
            B)
                run_system_prompt DEFAULT_HEDA
                python src/print_results.py -a >> results/E2/B/results.txt &&
                python src/print_results.py -i results/E2/B -b results/E1 >> results/E2/B/results.txt &&
                cat results/E2/B/results.txt | tail -n 2
                ;;
            C)
                run_system_prompt DT
                python src/print_results.py -a >> results/E2/C/results.txt &&
                python src/print_results.py -i results/E2/C -b results/E1 >> results/E2/C/results.txt &&
                cat results/E2/C/results.txt | tail -n 2
                ;;
            D)
                run_system_prompt AOA
                python src/print_results.py -a >> results/E2/D/results.txt &&
                python src/print_results.py -i results/E2/D -b results/E1 >> results/E2/D/results.txt &&
                cat results/E2/D/results.txt | tail -n 2
                ;;
            all)
                run_all_system_prompt
                python src/print_results.py -a >> results/E2/results.txt &&
                python src/print_results.py -i results/E2/A -b results/E1 >> results/E2/results.txt &&
                python src/print_results.py -i results/E2/B -b results/E1 >> results/E2/results.txt &&
                python src/print_results.py -i results/E2/C -b results/E1 >> results/E2/results.txt &&
                python src/print_results.py -i results/E2/D -b results/E1 >> results/E2/results.txt &&
                cat results/E2/results.txt | tail -n 5
                ;;                
            *)
                usage
                ;;
        esac
        ;;
    E3)
        if [ $# -ne 2 ]; then
            usage
        fi
        echo "Checking SFT"
        case $2 in
            A)
                run_sft LORA
                python src/print_results.py -a >> results/E3/A/results.txt &&
                python src/print_results.py -i results/E3/A -b results/E1 >> results/E3/A/results.txt &&
                cat results/E3/A/results.txt | tail -n 2
                ;;
            B)
                run_sft ADALORA
                python src/print_results.py -a >> results/E3/B/results.txt &&
                python src/print_results.py -i results/E3/B -b results/E1 >> results/E3/B/results.txt &&
                cat results/E3/B/results.txt | tail -n 2
                ;;
            C)
                run_sft IA3
                python src/print_results.py -a >> results/E3/C/results.txt &&
                python src/print_results.py -i results/E3/C -b results/E1 >> results/E3/C/results.txt &&
                cat results/E3/C/results.txt | tail -n 2
                ;;
            D)
                run_sft PROMPT_TUNING
                python src/print_results.py -a >> results/E3/D/results.txt &&
                python src/print_results.py -i results/E3/D -b results/E1 >> results/E3/D/results.txt &&
                cat results/E3/D/results.txt | tail -n 2
                ;;
            E)
                run_sft ADAPTER_V1
                python src/print_results.py -a >> results/E3/E/results.txt &&
                python src/print_results.py -i results/E3/E -b results/E1 >> results/E3/E/results.txt &&
                cat results/E3/E/results.txt | tail -n 2
                ;;
            F)
                run_sft ADAPTER_V2
                python src/print_results.py -a >> results/E3/F/results.txt &&
                python src/print_results.py -i results/E3/F -b results/E1 >> results/E3/F/results.txt &&
                cat results/E3/F/results.txt | tail -n 2
                ;;
            all)
                run_all_sft
                python src/print_results.py -a >> results/E3/results.txt &&
                python src/print_results.py -i results/E3/A -b results/E1 >> results/E3/results.txt &&
                python src/print_results.py -i results/E3/B -b results/E1 >> results/E3/results.txt &&
                python src/print_results.py -i results/E3/C -b results/E1 >> results/E3/results.txt &&
                python src/print_results.py -i results/E3/D -b results/E1 >> results/E3/results.txt &&
                python src/print_results.py -i results/E3/E -b results/E1 >> results/E3/results.txt &&
                python src/print_results.py -i results/E3/F -b results/E1 >> results/E3/results.txt &&
                cat results/E3/results.txt | tail -n 7
                ;;                
            *)
                usage
                ;;
        esac
        ;;
    E4)
        echo "Checking SSRA"
        ACTIVATE_PRIMARY_ENV
        python src/ssra.py --dis l1_mean --loss_maintain_benign --evaluation_dataset_name $EVAL_HARM_DATASET --epoch 4 --harmful_emb_num 30 --benign_emb_num 60 --beta 1000 --peft_type LORA --repeat 1  --lr  5e-3 --model_output_dir results/E4/adapter --infer_output_file results/E4/harmfulness_infer.csv &&
        python src/eval_harm.py -i results/E4/harmfulness_infer.csv -o results/E4/harmfulness.json &&
        lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",peft="results/E4/adapter",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E4 &&
        mv results/E4/results.json results/E4/utility.json
        python src/print_results.py -a >> results/E4/results.txt &&
        python src/print_results.py -i results/E4 -b results/E1 >> results/E4/results.txt &&
        cat results/E4/results.txt | tail -n 2
        ;;
    E5)
        echo "Checking SSRD"
        ACTIVATE_PRIMARY_ENV
        python src/ssrd.py --model_name llama_lora_h10 --peft_type LORA --dis l1_mean  --epoch 10 --lr 1e-3 --harmful_emb_num 50 --beta 100 --loss_maintain_benign --evaluation_dataset_name $EVAL_HARM_DATASET --checkduplicateinferresults 1 --infer --model_output_dir results/E5/adapter --infer_output_file results/E5/harmfulness_infer.csv &&
        python src/eval_harm.py -i results/E5/harmfulness_infer.csv -o results/E5/harmfulness.json &&
        lm_eval --model hf --limit 0.1 --model_args pretrained="$TARGET_MODEL_PATH",peft="results/E5/adapter",trust_remote_code=True --tasks hellaswag,boolq,arc_easy --batch_size 10 --output_path results/E5 &&
        mv results/E5/results.json results/E5/utility.json
        python src/print_results.py -a >> results/E5/results.txt &&
        python src/print_results.py -i results/E5 -b results/E1 >> results/E5/results.txt &&
        cat results/E5/results.txt | tail -n 2
        ;;
    *)
        usage
        ;;
esac
echo "Script execution completed."