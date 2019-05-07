# Notas do aula do curso do David Silver, aula 5


-> Controle => otimização da função valor de um MDP
-> Model-free control => otimizar a função valor de um MDP sem conhecer a dinâmica probabilística do mesmo.

-> On-Policy Learning
    - Learn on the job
    - aprender uma política a partir de dados gerados com esta política.

-> Off-Policy
    - looking over someone shoulder
    - aprender uma política a partir de dados amostrados por outra política.

- GPI -> estimar e melhorar
    - GPI = Global policy iteration
    - implica um ciclo de avaliação e melhora

-> Métodos de Controle geralmente se baseiam na ideia do GPI

Pergunta!!! Os métodos baseados em amostragem (model-free)  podem não visitar todos os estados. Isso implica manter as características de convergência?

-> Trade-off Exploration x Explotation
    - greedy (gulosa) action -> sempre escolher a ação com maior valor.
    - Exploration action -> escolher uma ação aleatória.
    - epsilon-greedy action -> escolhe uma ação aleatória com probabilidade epsilon
    e a ação gulosa com probabilidade 1-epsilon, 0 < epsilon arbitrariamente pequeno

-> Teorema: Para toda política epsilon-greedy, uma política epsilon-greedy gerada a partir dela será tão boa quanto ou melhor.


