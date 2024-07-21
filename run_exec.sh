#!/bin/bash

# Caminho para o arquivo contendo os nomes dos modelos
MODEL_FILE="model_names.txt"

# Leia o arquivo linha por linha
while IFS= read -r MODEL_NAME; do
    # Cria diretórios de resultados com o nome do modelo
    mkdir -p results/run_full/${MODEL_NAME//\//-}-fs0
    mkdir -p results/run_full/${MODEL_NAME//\//-}-fs3

    echo "Writing into results/run_full/${MODEL_NAME//\//-}-fs0"
    # Executa os comandos com o modelo fornecido
    python main.py --model together --model_args engine=$MODEL_NAME --tasks memo_trap_en --description_dict_path description.json --num_fewshot 0 --conversation_template chatgpt --output_path results/run_full/${MODEL_NAME//\//-}-fs0/memo_trap_en.json >> results/run_full/${MODEL_NAME//\//-}-fs0/output.txt 2>&1

    echo "Writing into results/run_full/${MODEL_NAME//\//-}-fs0"
    python main.py --model together --model_args engine=$MODEL_NAME --tasks memo_trap_en --description_dict_path description.json --num_fewshot 3 --conversation_template chatgpt --output_path results/run_full/${MODEL_NAME//\//-}-fs3/memo_trap_en.json >> results/run_full/${MODEL_NAME//\//-}-fs3/output.txt 2>&1

    sleep 2
    
done < "$MODEL_FILE"