# Notas do aula do curso do David Silver, aula 8

## Learning and Planning (Model Based RL)

- Learn the model from the experiences in the world
    - Use the experiences from the interaction with environment to generate a model
    - Use the model to improve q function

- A model describes the understanding of the agent about an environment:
    - states transitions
    - transitions rewards

- Model-free X Model Based
    - Model-free RL -> agent does not concearn to understand the environment, but learns value function(and/or policy) solely based on experience.
    - Model-Based RL -> Learns a model and plan the value function (and/or policy) from the model. Replace some interactions with the world with simulated results based on agent's understanding of the environment

    #### What's the difference between Model-based RL and Planning?
    #### R: Model-based RL assumes that the model is not known, and it should be learnt from the world. In deed the policy is obtained from planning, but the model is not previously known to the agent, in other worlds, the particularities of the MDP is unknown (state transaction and Reward function).

- Advantages
    - Some times learning the model is more convenient then learning
    - Makes it possible to reason about the model uncertainty
    - May learn the model from supervised learning
- Disadvantages
    - Learning the model may leads to a wrong model, if constructing the value function may itself result in errors, then now there are two possible sources of error

 - Formally a model is a representation of a MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$, so assuming $\mathcal{S}$ and $\mathcal{A}$ are known, then a model is $\mathcal{M} = (\mathcal{P}_\eta, \mathcal{R}_\eta)$ with $\mathcal{P}_\eta \approx \mathcal{P}$ and $\mathcal{R}_\eta \approx \mathcal{R}$.

- Estimating the $\mathcal{P}_t, \mathcal{R}_t$ may be understood as two supervised learning problems where for each t we have.
    - $s, a \to r$ is a regression problem
    - $s, a \to s'$ is a density estimation problem
$$ S_1, A_1 = \mathcal{P}_2, \mathcal{R}_2 $$
$$ S_2, A_2 = \mathcal{P}_3, \mathcal{R}_3 $$
$$ ... $$
$$ S_{T-1}, A_{T-1} = \mathcal{P}_T, \mathcal{R}_T $$

- There are several different types of models.

- Once there is a model, one might just apply any sample-based RL algorithm (Monte Carlo methods, SARSA, Q-learning) to estimate the action-value function.

- A interesting idea is to combine model-free and model-based learnings. The idea is to use both real experiences, from the environment, and simulated experiences, sampled from a model.

- One common way to combine those ideas is to apply Dyna architecture.
    - Formed by two loops, one model-based, called indirect RL and onde model-free called direct RL, but the direct RL loop also updates the model, generating an input for also the model-based loop

-