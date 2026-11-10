"""Level 1: an agent writes a cheaper program that predicts a network's outputs, and a verifier measures it.

    lang        the entry language (counted operations): builder, interpreter, measure
    networks    model folders: weights, the network written as an entry, what the agent is given
    verifier    scores an entry on a network: coverage, fidelity, costs, reward
    rewards     the reward functions, by name
    session     one agent session: its tools, budgets and log
    loop        the rounds, the lock, the seed check and the report
"""

__version__ = "0.1.0"
