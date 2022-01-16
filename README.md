# arctic delta analysis
to run code, deposit config_xxx_oceanmasked.png images in ./data/ directory and run a_delta_to_graph to create graphs from the detla image. Then run b_delta_network_analysis to create (or overwrite) the delta_metrics.csv file with information on the following network metrics:
- number of nodes
- number of edges
- number of components
- density (how many of all possible edges do exist)
- diameter (the longest of all shortest path lengths; in km)
- length of channels (in km)
