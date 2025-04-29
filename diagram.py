from graphviz import Digraph

dot = Digraph(comment='System Design')
dot.edge('Client', 'LoadBalancer')
dot.edge('LoadBalancer', 'ServiceA')
dot.edge('LoadBalancer', 'ServiceB')
dot.edge('LoadBalancer', 'ServiceC')

dot.render('diagram_output', format='png', cleanup=True)
print("Saved as diagram_output.png")
