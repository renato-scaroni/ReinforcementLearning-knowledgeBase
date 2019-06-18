# Notas do aula do curso do David Silver, aula 7

## Policy Gradient methods

- On the other methods, the idea was to approximate value-function (state or action), and then generate the policy

- PG methods parametrizes and try to achieve directly the policy, approximating its function.
$$
    \pi_\theta(s, a) = \mathbb{P}[a|s, \theta]
$$

- Advantages:
    - Better converge properties
    - Effective in high-dimensional or continuous action spaces
    - Can learn stochastic policies
- Disadvantages:
    - Typically converge to a local rather than a global optimum
    - Evaluating a policy is typically inefficient and high variance (naive methods)

- What will be the objective function?
    - the expected value of the start state (good for episodic)
    - average value
    - average reward per time-step

- Policy based is actually an optimization problema, so a lot of other techniques would apply, but PG tends to be more efficient

- PG idea uses gradient ascent to change the policy function in the direction that maximizes it

- The policy just need to be differentiable when non-zero

- $\phi(s, a)$ is the features function of a state-action pair.

- Softmax policy:
$$
    \pi_\theta(s, a) = e^{\phi(s, a)^T\theta}
$$
- Softmax policy score function:
    - the idea is to change the policy in the direction in a way to maximize the policy score
$$
    \nabla_\theta log \pi_\theta(s,a) = \phi(s, a) - \mathbb{E} [\phi(s,.)]
$$

- Gaussian Policy score function:
    - $\eta(s) = \phi(s)^T\theta$
    - $\sigma^2$ is the variance. It may also be parametrized
$$
\nabla_\theta log \pi_\theta(s,a) = \frac{(a - \mu(s)) \phi(s)}{\sigma^2}
$$

- One step MDP
    - The objective function is such:
    - $J(\theta) = \mathbb{E}[r]$
    - $\nabla_\theta J(\theta) = \mathbb{E} [\nabla_\theta log \pi_\theta(s,a)r]$ -> where r is the reward recieved in that state
    - The geralization for multi-step MDP involves using a sample of the return

- Policy gradient theorem
    - The Gradient of the objective function is the expected value of the score function times the estimation of the action-value function to a given state-action pair
    - $\nabla_\theta J(\theta) = \mathbb{E} [\nabla_\theta log \pi_\theta(s,a)Q^{\pi_\theta}(s,a)]$

- Monte Carlo PG tends to take too long to converge and introduces lots of variance

### Doubt: So the MCPG method is the only way of making PG? Can't i use TD inspired techniques?

- Actor-critic methods:
    - two setsof parameters:
    - Critic: update action-value function parameters w
    - Actor: Update policy parameters $\theta$ in direction suggested by the critic

- algorithms
    - Value-based: learn value-function, no policy
    - Policy-based: learn policy, no value-function
    - Actor-critic: learn both policy and value-function

- Action-Value Actor-critic:
    - Critic updates w by linear TD(0)
    - Actor updates $\theta$ by policy gradient

