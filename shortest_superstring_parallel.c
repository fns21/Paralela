#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>

#define NUM_THREADS 16

/**
 * @brief Calcula o comprimento do maior sufixo de 'a' que é também um prefixo de 'b'.
 * * @param a A primeira string.
 * @param b A segunda string.
 * @return O comprimento da sobreposição.
 */
int calculate_overlap_simd(const char* a, const char* b) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    int max_overlap = 0;
    int min_length = (len_a < len_b) ? len_a : len_b;

    // Itera do maior overlap possível para o menor
    for (int i = min_length; i > 0; --i) {
        const char* suffix_a = a + len_a - i;
        const char* prefix_b = b;

        bool is_match = true;

        // Pedimos ao compilador para vetorizar este loop de comparação.
        // A cláusula `reduction` garante que a variável `is_match`
        // seja tratada corretamente pelas threads SIMD.
        #pragma omp simd
        for (int k = 0; k < i; ++k) {
            if (suffix_a[k] != prefix_b[k]) {
                is_match = false;
            }
        }
        
        // Se encontramos uma correspondência, este é o maior overlap possível.
        // Podemos parar e retornar o valor imediatamente.
        if (is_match) {
            max_overlap = i;
            break; // Otimização: sair cedo, já que encontramos o maior.
        }
    }
    return max_overlap;
}

/**
 * @brief Mescla duas strings com base em sua sobreposição.
 * * @param a A primeira string.
 * @param b A segunda string.
 * @param overlap_len O comprimento da sobreposição (sufixo de 'a' e prefixo de 'b').
 * @return Uma nova string alocada contendo a fusão. O chamador deve liberá-la.
 */
char* merge_strings(const char* a, const char* b, int overlap_len) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    int new_len = len_a + len_b - overlap_len;

    char* result = (char*)malloc(new_len + 1);
    if (result == NULL) {
        perror("Falha ao alocar memória em merge_strings");
        exit(EXIT_FAILURE);
    }

    // Copia string a para result
    memcpy(result, a, len_a);
    memcpy(result + len_a, b + overlap_len, len_b - overlap_len);
    result[new_len] = '\0';

    return result;
}

/**
 * @brief Mescla duas strings com base em sua sobreposição.
 * * @param a A primeira string.
 * @param b A segunda string.
 * @param overlap_len O comprimento da sobreposição (sufixo de 'a' e prefixo de 'b').
 * @return Uma nova string alocada contendo a fusão. O chamador deve liberá-la.
 */
char* shortest_superstring(char** strings, int* count) {
    while (*count > 1) {
        int i, j;
        int max_overlap = -1;
        int best_i = -1, best_j = -1;

        // Encontra o par de strings com a maior sobreposição
        #pragma omp parallel
        {
            int local_max_overlap = -1;
            int local_best_i = -1, local_best_j = -1;
            #pragma omp for collapse(2) schedule(guided) nowait
            for (i = 0; i < *count; ++i) {
                for (j = 0; j < *count; ++j) {
                    if (i != j){

                        int current_overlap = calculate_overlap_simd(strings[i], strings[j]);

                        bool is_better = false;
                        if (current_overlap > local_max_overlap) {
                            is_better = true;
                        } else if (current_overlap == local_max_overlap && local_max_overlap != -1) {
                            // Compara o par (i, j) com o melhor par (best_i, best_j)
                            int cmp1 = strcmp(strings[i], strings[local_best_i]);
                            if (cmp1 < 0) {
                                is_better = true;
                            } else if (cmp1 == 0) {
                                int cmp2 = strcmp(strings[j], strings[local_best_j]);
                                if (cmp2 < 0) {
                                    is_better = true;
                                }
                            }
                        }
                        if (is_better) {
                            local_max_overlap = current_overlap;
                            local_best_i = i;
                            local_best_j = j;
                        }
                    }
                }
            }

            #pragma omp critical
            {
                if (local_max_overlap > max_overlap) {
                    max_overlap = local_max_overlap;
                    best_i = local_best_i;
                    best_j = local_best_j;
                } else if (local_max_overlap == max_overlap && local_max_overlap != -1) {
                    int cmp1 = strcmp(strings[local_best_i], strings[best_i]);
                    if (cmp1 < 0) {
                        best_i = local_best_i;
                        best_j = local_best_j;
                    } else if (cmp1 == 0) {
                        int cmp2 = strcmp(strings[local_best_j], strings[best_j]);
                        if (cmp2 < 0) {
                            best_i = local_best_i;
                            best_j = local_best_j;
                        }
                    }
                }
            }
        }

        // Se não houver sobreposição, mescla os dois primeiros para evitar loop infinito
        // Esta parte pode ser removida se o critério de desempate garantir que um par
        // sempre seja escolhido, mas é uma boa salvaguarda.
        if (best_i == -1) { // Acontece se houver 1 ou 0 strings, ou todas iguais
             best_i = 0;
             best_j = 1;
             max_overlap = 0;
        }

        // Mescla o melhor par encontrado
        char* merged = merge_strings(strings[best_i], strings[best_j], max_overlap);
        
        // --- LÓGICA DE REMOÇÃO CORRIGIDA E ROBUSTA ---
        // Para evitar bugs, sempre lidamos com os índices em ordem.
        int idx_to_replace = (best_i < best_j) ? best_i : best_j;
        int idx_to_remove = (best_i > best_j) ? best_i : best_j;

        // Libera a memória das strings que foram mescladas
        free(strings[best_i]);
        free(strings[best_j]);

        // Substitui a string no índice menor pela nova string mesclada
        strings[idx_to_replace] = merged;

        // Remove a string do índice maior, deslocando os elementos restantes para a esquerda
        size_t elems = (size_t)(*count - idx_to_remove - 1);
        if (elems > 0) {
            char **tmp = malloc(elems * sizeof(char *));

            #pragma omp parallel for schedule(static)
            for (size_t t = 0; t < elems; ++t) {
                tmp[t] = strings[idx_to_remove + 1 + t];
            }

            #pragma omp parallel for schedule(static)
            for (size_t t = 0; t < elems; ++t) {
                strings[idx_to_remove + t] = tmp[t];
            }

            free(tmp);
        }        
        (*count)--;
    }

    // Retorna uma cópia da única string restante
    return strdup(strings[0]);
}


int main() {
    int n;

    omp_set_num_threads(NUM_THREADS);

    if (scanf("%d", &n) != 1 || n < 0) {
        fprintf(stderr, "Entrada inválida para o número de strings.\n");
        return 1;
    }
    
    // Aloca um array de ponteiros para as strings
    char** strings = (char**)malloc(n * sizeof(char*));
    if (strings == NULL && n > 0) {
        perror("Falha ao alocar memória para o array de strings");
        return 1;
    }

    // Buffer temporário para ler cada string
    char buffer[1024]; 
    for (int i = 0; i < n; ++i) {
        if (scanf("%1023s", buffer) != 1) {
            fprintf(stderr, "Erro ao ler a string %d.\n", i + 1);
            // Libera memória já alocada antes de sair
            for (int j = 0; j < i; ++j) {
                free(strings[j]);
            }
            free(strings);
            return 1;
        }
        strings[i] = strdup(buffer);
    }
    
    if (n == 0) {
        printf("\n");
        free(strings);
        return 0;
    }

    int string_count = n;
    double start = omp_get_wtime();
    char* result = shortest_superstring(strings, &string_count);
    double end = omp_get_wtime();
    
    printf("%s\n", result);
    fprintf(stderr, "Tempo de execução paralelo: %.6f segundos\n", end - start);
    
    // Libera toda a memória alocada
    free(result);
    for (int i = 0; i < string_count; ++i) {
        free(strings[i]);
    }
    free(strings);

    return 0;
}