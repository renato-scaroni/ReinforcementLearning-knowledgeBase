# Notas do aula do curso do David Silver, aula 3

Planejamento por programação dinâmica.

1) Avaliação de política
1) Iteração de política
1) Iteração de valor

Planejamento != aprendizado por reforço, mas os métodos de um podem ser base para outro, em especial esses citados acima

- Dynamic programming -> Otimização a programas para problemas de natureza sequencial.
    - Possibilita a resolução de problemas mais complexos por quebrá-los em problemas menores e juntar suas soluções. Baseia-se em duas ideias centrais:
        - Uma sub-estrutura ótima para um problema que podem ser resolvidas separadamente
        - Subproblemas ótimos ocorrem repetidas vezes e a solução ótima do problema é a junção da solução ótima para cada um desses
            subproblemas. Além disso essas soluções podem ser reutilizadas em diferentes etapas do problema

- A modelagem de MDP gera uma solução que satisfaz ambas características centrais de uma solução de DP devido a sua estrutura
    recursiva.

- Dynamic programming assumes full knowledge of the MDP.
- Pode ser usado para o problema de planejamento em um MDP em dois cenários:
    - predição: Dado um MDP ou um MRP retorna os valores de cada estado
    - controle: Dado um MDP retorna os valores ótimos de cada estado bem como a política  ótima

- Avaliação de política:
    - Problema: avaliar uma política
    - Solução: aplicar as equações de Bellman de forma iterativa

    - Utiliza backups síncronos:
        - Para cada passo k+1 varre todos os estados e atualiza o valor de cada um a partir do valor de seu sucessor
        - Calcula os valores acima a partir da política dada.

- Iteração de política:
    - Dada uma política pi, é realizada uma avaliação como descrito acima.
    - Em seguida escolhe de forma gulosa a melhor ação de acordo com os valores calculados.
    - Assim chegamos na política ótima.
    - Sabemos que sempre haverá pelo menos uma política ótima para um MDP.
    - um jeito de otimizar iteração de valor é atribuir um epsilon de distancia entre um valor e outro. Ou seja, parar a iteração quando a política for boa o suficiente

Pergunta!!! O que acontece se começarmos com valores arbitrários para cada v(s) e guardarmos o valor de v apenas se for ótimo, mas garantirmos que passamos por todas ações de cada estado pelo menos uma vez?

- Iteração de valor
    - Uma política é ótima se v_pi(s') == v_*(s') para todo s' acessível de s
    -

Um algoritmo é assíncrono se ele não passa por todos os estados em todas as iterações