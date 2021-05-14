# world_model_jssp_pasula

## Objective:

- To create a RL algorithm that uses the model based graph neural network to solve the JSSP problem

## Details

### RL Algorithm:
- The RL algorithm used is World Models ([paper](https://arxiv.org/abs/1803.10122)) initially. If the implementation is deemed difficult, we could always revert to a more a simpler model based RL (Dyna-Q)

### JSSP environment:
- JSSP environments used are the Robot Manufacturing Cell (RMC) and the Injection Moulding Machines (IMM). RMC should be used as a proof of concept and the actual thesis work is to get results in IMM environment. As things stand, the IMM environment is ready (with @sharafath) and the RMC environment is now taken over by @sharafath and will be provided to the student in a few weeks time.

### Graph Neural network
- Graph neural network ([GNN](https://arxiv.org/abs/1806.01261)) is to deployed to the use the graph structure of the environment.
- A message passing algorithm is to be developed (which is core to the thesis) to extract meaningful information between the nodes given in the JSSP.
- The GNN is to be used as form of graph encoding that the RL algorithm uses downstream.
- As a general idea, a graph model based RL can be developed to directly learn the graph structure between the nodes.
    (Current_state, Action) --> Model --> (New_state, reward)

- A GAE ([paper](https://arxiv.org/abs/1611.07308)) style GNN can be implemented to learn the model while the encoder can be used by the RL algorithm for downstream learning.


