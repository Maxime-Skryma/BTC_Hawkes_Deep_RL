"""Agent courant : apprentissage de valeur sans réseau acteur.

La variante acteur-critique v4 est conservée dans cjr.actor_critic.
"""
from cjr.value_agent import (ALGORITHM_VERSION, Agent, Normalizer, mlp, train,
                             side_terms, greedy_deltas, continuation_values,
                             load_saved, move_saved)
