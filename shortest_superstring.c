#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <omp.h>

/**
 * @brief Calcula o comprimento do maior sufixo de 'a' que é também um prefixo de 'b'.
 * * @param a A primeira string.
 * @param b A segunda string.
 * @return O comprimento da sobreposição.
 */
int calculate_overlap(const char* a, const char* b) {
    int len_a = strlen(a);
    int len_b = strlen(b);
    int max_overlap = 0;

    for (int i = 1; i <= len_a && i <= len_b; ++i) {
        // Compara o sufixo de 'a' de comprimento 'i' com o prefixo de 'b' de comprimento 'i'.
        if (strncmp(a + len_a - i, b, i) == 0) {
            max_overlap = i;
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

    memcpy(result, a, len_a);
    memcpy(result + len_a, b + overlap_len, len_b - overlap_len);

    result[new_len] = '\0';

    return result;
}

/**
 * @brief Encontra a superstring mais curta para um conjunto de strings.
 * * @param strings Um array de strings.
 * @param count O número de strings no array.
 * @return A superstring mais curta. O chamador deve liberar a memória.
 */
char* shortest_superstring(char** strings, int* count) {
    double parallelable_time_total = 0.0;

    while (*count > 1) {
        int max_overlap = -1;
        int best_i = -1, best_j = -1;

        double start_parallelable_time_local = omp_get_wtime();
        // Encontra o par de strings com a maior sobreposição
        for (int i = 0; i < *count; ++i) {
            for (int j = 0; j < *count; ++j) {
                if (i == j) continue;

                int current_overlap = calculate_overlap(strings[i], strings[j]);

                // --- MODIFICAÇÃO PRINCIPAL INICIA AQUI ---
                // A condição para atualizar o melhor par foi expandida.
                // Agora, um novo par é escolhido se:
                // 1. A sobreposição dele é estritamente maior que a máxima encontrada.
                // OU
                // 2. A sobreposição é IGUAL à máxima, E o novo par é lexicograficamente
                //    menor que o melhor par atual (critério de desempate).
                bool is_better = false;
                if (current_overlap > max_overlap) {
                    is_better = true;
                } else if (current_overlap == max_overlap && max_overlap != -1) {
                    // Compara o par (i, j) com o melhor par (best_i, best_j)
                    int cmp1 = strcmp(strings[i], strings[best_i]);
                    if (cmp1 < 0) {
                        is_better = true;
                    } else if (cmp1 == 0) {
                        int cmp2 = strcmp(strings[j], strings[best_j]);
                        if (cmp2 < 0) {
                            is_better = true;
                        }
                    }
                }

                if (is_better) {
                    max_overlap = current_overlap;
                    best_i = i;
                    best_j = j;
                }
                // --- MODIFICAÇÃO PRINCIPAL TERMINA AQUI ---
            }
        }

        double end_parallelable_time_local = omp_get_wtime();
        parallelable_time_total += end_parallelable_time_local - start_parallelable_time_local;
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

        start_parallelable_time_local = omp_get_wtime();
        // Remove a string do índice maior, deslocando os elementos restantes para a esquerda
        for (int k = idx_to_remove; k < *count - 1; ++k) {
            strings[k] = strings[k + 1];
        }
        end_parallelable_time_local = omp_get_wtime();
        parallelable_time_total += end_parallelable_time_local - start_parallelable_time_local;
        
        (*count)--;
    }

    fprintf(stderr, "Tempo total das regiões paralelizáveis: %.6f segundos\n", parallelable_time_total);
    // Retorna uma cópia da única string restante
    return strdup(strings[0]);
}

int main() {
    int n;
    double start_time_global = omp_get_wtime();
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
    char* result = shortest_superstring(strings, &string_count);

    printf("%s\n", result);

    // Libera toda a memória alocada
    free(result);
    for (int i = 0; i < string_count; ++i) {
        free(strings[i]);
    }
    free(strings);

    double end_time_global = omp_get_wtime();
    fprintf(stderr, "Tempo de execução total: %.6f segundos\n", end_time_global - start_time_global);

    return 0;
}