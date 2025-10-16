#!/bin/bash

# --- CONFIGURAÇÕES ---
# Nome do arquivo de código-fonte C
SOURCE_FILE="shortest_superstring.c"
# Nome que daremos ao programa executável após a compilação
EXECUTABLE="shortest_superstring"
# Arquivos de entrada para os testes
INPUT_FILES=("teste.txt" "teste2.txt" "teste3.txt")
# Número de threads a serem testadas
# Quantidade de vezes que cada combinação será executada
NUM_RUNS=10

# --- COMPILAÇÃO ---
echo "Compilando o programa '$SOURCE_FILE'..."
# Usamos -fopenmp para habilitar o OpenMP e -O3 para otimização de performance
gcc -o "$EXECUTABLE" "$SOURCE_FILE" -mavx2 -fopenmp -O3

# Verifica se a compilação foi bem-sucedida
if [ $? -ne 0 ]; then
    echo "Erro na compilação. Verifique o código e tente novamente. Abortando."
    exit 1
fi
echo "Compilação bem-sucedida. Executável '$EXECUTABLE' criado."
echo ""

# --- EXECUÇÃO DOS TESTES ---

# Loop principal: Itera sobre cada arquivo de entrada
for input_file in "${INPUT_FILES[@]}"; do
    
    # Define o nome do arquivo de log baseado no arquivo de entrada
    logfile="log_seq_${input_file}"
    
    echo "----------------------------------------------------"
    echo "Iniciando testes para o arquivo de entrada: $input_file"
    echo "Os resultados serão salvos em: $logfile"
    
    # Limpa o arquivo de log anterior e adiciona um cabeçalho
    echo "Resultados dos Testes para '$input_file'" > "$logfile"
    echo "Gerado em: $(date)" >> "$logfile"
    
    # Segundo loop: Itera sobre cada contagem de threads
        
        echo "   -> Executando com $threads thread(s)..."
        
        # Adiciona um separador no arquivo de log para esta contagem de threads
        echo -e "\n==========================================" >> "$logfile"
        echo "           TESTES COM $threads THREAD(S)" >> "$logfile"
        echo "==========================================" >> "$logfile"
        
        # Terceiro loop: Executa o programa 10 vezes
        for (( run=1; run<=$NUM_RUNS; run++ )); do
        
            # Define a variável de ambiente OMP_NUM_THREADS, que o OpenMP usa
            # para saber quantas threads deve criar. 'export' a torna visível
            # para o processo filho (nosso programa C).
            
            # Adiciona o cabeçalho para a execução atual no log
            echo -e "\n--- Execução (Run) #$run ---\n" >> "$logfile"
            
            # Executa o programa:
            # < "$input_file"  : Redireciona o conteúdo do arquivo para a entrada padrão (stdin) do programa.
            # >> "$logfile"   : Anexa a saída padrão (stdout - o resultado da superstring) ao arquivo de log.
            # 2>&1             : Redireciona a saída de erro (stderr - onde imprimimos o tempo) para o mesmo
            #                  local da saída padrão, ou seja, para o arquivo de log também.
            ./"$EXECUTABLE" < "$input_file" >> "$logfile" 2>&1
            
        done
    
    echo "Testes para '$input_file' concluídos."
done

echo "----------------------------------------------------"
echo "Todos os testes foram finalizados com sucesso!"