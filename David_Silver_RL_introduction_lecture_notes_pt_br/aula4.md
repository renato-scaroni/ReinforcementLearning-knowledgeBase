# Notas do aula do curso do David Silver, aula 4

Model-Free Reinforcement Learning

Resolver o MDP, sem o modelo, ou seja, sem o MDP que define o problema.
-> métodos onde nada é dito sobre o ambiente mas o agente ainda tem que aprender melhor política para agir

1) Introdução
1) Monte-Carlo
1) Temporal Difference learning
1) TD(lambda)


Foco em avaliação de política, ou seja, predição e não controle

-> Monte Carlo Learning:
    - aprendizado sobre episódios de experiência
    - Model-free (Não precisa de nenhum input a respeito de transições/recompensas do MDP)
    - Aprende de apisodios completos
    - Aprende o a função-valor para uma política \pi
    - o valor empírico do retorno médio é tratado como o retorno esperado

-> First-visit MC Policy evaluation
    - Avalia o estado s, considera o valor médio do retorno acumulado apenas no primeiro tempo t no qual tal estado é atingido
    - Conta-se o número de vezes que um estado s é vistado, N(s) <- N(s) + 1 então e incrementa a soma total do estado S(s) <- S(s) + G_t ao longo de uma série de episódios.
    - Calcula-se o valor do estado como sendo a média dos retornos: V(s) = S(s)/N(s)

-> Every-visit MC Policy evaluation
    - Avalia o estado s, considera o valor médio do retorno acumulado para cada vez  que tal estado é atingido
    - Conta-se o número de vezes que um estado s é vistado, N(s) <- N(s) + 1 então e incrementa a soma total do estado S(s) <- S(s) + G_t ao longo de uma série de episódios.
    - Calcula-se o valor do estado como sendo a média dos retornos: V(s) = S(s)/N(s)

PERGUNTA!!! Não ficou muito claro como o every visit lida com a questão dos loops de estados e como isso pode ser evitado.

-> Incremental monte-carlo
    - Média incremental: a média de k passos pode ser escrita como a média de k-1 passos mais uma constando advinda da diferença entre o valor atual obtido e a média anterior:
        m_k = m_{k-1} + 1/k (x_k - m_{k-1})

    - Em cada episódio guardamos N(S_t) e atualizamos o valor do estado como:
        V(S_t) <- V(S_t) + 1/N(S_t) * (G_t - V(S_t) )

-> Temporal Difference Learning
    - Aprendem direto dos episódios
    - Model-free
    - Aprende por episódios incompletos.
    - atualiza um palpite em direção a outro palpite subsequente
    - Utiliza um retorno estimado no lugar do retorno obtido. Assim o valor é atualizado a cada passo utilizando uma estimativa de forma similar às equações de bellman.
    - Atualiza baseado no chamado Time-Difference error, que no cado seria:
        \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
    - R_{t+1} + \gamma V(S_{t+1}) é target do TD
    - Assim a equação de atualização é:
        V(S_t) = V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))

-> TD vs MC
    - Em monte-carlo usa-se uma medição dos retornos a partir de experimentos. Com isso o valor de valor é não-viesado
    - Em TD, se usassemos o valor do traget real, ou seja, R_{t+1} + \gamma v_\pi(S_{t+1}) não é viesado, porém, como usamos R_{t+1} + \gamma V(S_{t+1}), ou seja, usamos o valor estimado de t+1, então há um viés no valor estimado do retorno.
    - O uso da estimativa no calculo do valor de um passo em TD apesar de adicionar viés, reduz a variância dos valores previstos em relação aos valores de Monte-carlo.
    - Menos vies, maior convergencia

PERGUNTA!! Nesse viés/variância, qual o benefício em diminuir a variância?

Teoricamente parece que uma menor variância nos valores gera maior eficiência na execução do algoritmo
* Verificar afirmações acima

-> Bootstrapping: atualização do valor envolve utilizar uma estimativa da função-valor
    - MC não faz bootstrap
    - DP faz bootstrap
    - TD faz bootstrap
-> Sampling: atualização do valor envolve uma medição do valor durante um episódio ou  da função-valor
    - MC faz sampling
    - DP não faz sampling
    - TD faz sampling

-> O TD learning abordado acima é conhecido como TD(0), com uma abordagem de 1-step, ou seja, avançamos 1 step no futuro e só consideramos essa medição.

-> TD(\lambda) é um método que computa o valor de um estado em uma abordagem n-step, ou seja, computando o valor após 1 passo no futuro, 2, 3, ..., n e a cada computação multiplica por um valor (1-\lambda)\lambda^{n-1}