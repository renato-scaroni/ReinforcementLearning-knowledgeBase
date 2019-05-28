# Notas do aula do curso do David Silver, aula 6

## Value function approximation

    - Two Approaches:
        - incremental
        - batch

- Such problems have state spaces that is too big or continuous, making building a table of states impossible or impractical (due to memory limitations or iterations limitation)

- Approximating value functions make possible to learn value functions without visiting every state and storing every state/state-pair in memory.

- $\hat{v}(s, \bold{w}) \approx v_\pi(s)$, where $\bold{w}$ is a vector of weights which is the parameters for the function approximator, possibly the weights of a neural net, which indicates how to approximate the state-value

- analogously to state action-value we can do $\hat{q}(s,a,\bold{w}) \approx q_\pi(s,a)$

- Function approximators:
    - Linear combinations of features
    - Neural nets
    - Decision tree
    - Nearest neighbour
    - Fourier/wavelet bases
- For RL usually are used the ones that generate differentiable functions so we can differentiate them and find the gradient:
    - Linear combinations of features
    - Neural nets

- The data is different then the ones used for training nets for supervised learning as it is generated iteratively and the policy generating data may be changing

- How to train your dragon.... i mean approximator? Gradient descent!

    - Let $J(\bold{w})$ be a differentiable function with paraneter vector $\bold{w}$. Lets define the gradient of J as:
    $$
    \nabla J(\bold{w}) =
    \left[
    \begin{array}{cccc}
        \frac{\partial J(\bold{w})}{\partial \bold{w}_1}\\
        \ldots \\
        \frac{\partial J(\bold{w})}{\partial \bold{w}_n}
    \end{array}
    \right]
    $$

    - find a local minimmum for $J(\bold{w})$
    - adjust parameter $\bold{w}$ in the direction of the gradient:
    $$
        \Delta \bold{w} = \frac{1}{2} \nabla J(\bold{w})
    $$

    - The goal is to find the parameter w that minimizes the the mean-squared error between the approximate value of $\hat{v}(s, \bold{w})$ and $v_\pi(s)$.
- Linear value Function approximation
    - A state is represented by a feature vector, each coordinate of the vector is a number that describes an aspect of the state $x(s)$

    - The value function will be calculated as a linear combination of the features:
    $$
    \hat v (S, \bold{w}) = x(S)^T \bold{w} = \sum_{j=1}^n x(S)_i \bold{w}_j
    $$

    - Using a linear combination garantees to get us the best possible minimum for the function in an stochastic gradient descent

- Table lookup is a particular case of function approximation. We can interpret the feature vector as each coordinate of the vector is an state and the feature vector $x(S_i)$ for an state $S_i$ is:
$$
x(S_i)=
\begin{cases}
x_j = 0,\, if\; i \neq j\\
x_j = 1,\, if\; i = j
\end{cases}
$$

- The main idea is to substitute the sample parts of the algorithms by $\hat v(S_t, \bold{w})\nabla_\bold{w} \hat v(S_t, \bold{w})$ so now the updates would be $V(S_t) \leftarrow V(S_t) + \Delta \bold{w}$, where:
    - For MC: $\Delta \bold{w} = \alpha \, [ G_t - \hat v(S_t, \bold{w})\nabla_\bold{w}]\, \hat v(S_t, \bold{w})$
    - For TD(0): $\Delta \bold{w} = \alpha \, [R_{t+1} + \gamma \hat v(S_{t+1}, \bold{w}) - \hat v(S_t, \bold{w})\nabla_\bold{w}] \, \hat v(S_t, \bold{w})$
    - For forward view TD($\lambda$): $\Delta \bold{w} = \alpha \, [G_t^\lambda - \hat v(S_t, \bold{w})\nabla_\bold{w}] \, \hat v(S_t, \bold{w})$
    - For backward view TD($\lambda$):
        - $\delta_t = R_{t+1} + \gamma \hat v(S_{t+1}, \bold{w}) - \hat v(S_t, \bold{w})$
        - $E_t = \gamma\lambda E_{t-1} + x(S_t)$
        - $\Delta \bold{w} = \alpha \delta_t E_t$


### Doubts: So monte carlo converges only to a local minimum? Is this good??? But shouldn`t we have the global maximum garateed by the linear combination? And why TD(0) converges to the global optimum?

- TD target has biased samples

- In each the idea is to generated data that incrementally "trains" the weights that should be applied as an approximation.

- Control with approximate functions:
    - The idea is to build on top of the evaluation methods
    - now instead of estimating $\hat v$, the goal is to estimate $\hat{q}(s,a,\bold{w}) \approx q_\pi(s,a)$

- Action-value approximation:
    - Uses gradient descent
    - feature vector is now built over state-action pair
$$
x(S, A) =
\left[
\begin{array}{cccc}
    x_1(S, A)\\
    \ldots \\
    x_n(S, A)
\end{array}
\right]
$$

- The application with MC, TD(0) and TD($\lambda$) is analogous as with value evaluation, but with the q action-value function.

- Convergence of prediction algorithms:

    | On/off policy  |  Algorithm |  table lookup |  linear |  Non-linear |
    |---|---|---|---|---|
    | On policy  |  MC |  Y | Y  |  Y |
    | On policy  |  TD |  Y |  Y |  N |
    | On policy  |  TD($\lambda$) | Y  |  Y |  N |
    | Off policy  |  MC | Y  | Y  | Y  |
    | Off policy  |  TD | Y  | N  | N  |
    | Off policy  |  TD($\lambda$) | Y  | N  |  N |

    - But there are recent methods like gradient TD, that converge in every scenario


- Convergence of control algorithms:
    |  Algorithm |  table lookup |  linear |  Non-linear |
    |---|---|---|---|
    |  MC |  Y | Y*  |  N |
    |  SARSA |  Y |  Y* |  N |
    |  Q-learning | Y  |  N |  N |
    |  Gradient Q-learning | Y  | Y  | N  |

    *In theses cases there is chance that the values starts to diverge occasionally when near-optimal value function

- Batch Methods
    - Takes action according to an $\epsilon$-greedy policy, sample random minibatch transitions. Compute Q-learning and optimize MSE between Q-Network (a network for function approcimation) and Q-learning targets
    - Experience replay breaks the correlation between the networks and thus stabilizes the results once there is no correlation anymore about the values sampled and the targets. The other idea is to use another network to sample and to generate target.

### Doubts: didnt understand the concept of experience replay and why it really stabilizes the algorithm. BTW What exactly does he means with algorithm stabilization?


