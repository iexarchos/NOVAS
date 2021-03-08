"""
The main file to get the desired graph.

"""

from graphs.graph_oneshot import *

def get_graph(graph_name, config, dynamics_model, lstm, solver):
    if graph_name == 'oneshot':
        graph, recurrent_path = oneshot(config, dynamics_model, lstm, solver)

    return graph