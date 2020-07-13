import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	# YOUR CODE HERE
	connected_component_count = g.connectedComponents().select('component').distinct().count()
	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		# YOUR CODE HERE
		vertices = g.vertices.map(lambda row: row.id).collect()
		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
			# the output
		# YOUR CODE HERE
		components = []
		for vertex in vertices:
			v_list = g.vertices.filter('id !="'+vertex+'"')
			edge_list = g.edges.filter('src!="'+vertex+'"').filter('dst!="'+vertex+'"')
			new_g = GraphFrame(v_list, edge_list)
			new_g_count = new_g.connectedComponents().select('component').distinct().count()
			components.append([vertex, 1] if new_g_count>connected_component_count else [vertex, 0])

		articulations_df = sqlContext.createDataFrame(components, ['id', 'articulation'])
		return articulations_df	
	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
		# YOUR CODE HERE
		nxg = nx.Graph()
		vertices = g.vertices.map(lambda row: row.id).collect()
		edges = g.edges.map(lambda row: (row.src, row.dst)).collect()
		nxg.add_nodes_from(vertices)
		nxg.add_edges_from(edges)
		components = []
		for vertex in vertices:
			copy_nxg = deepcopy(nxg)
			copy_nxg.remove_node(vertex)
			c_count = nx.number_connected_components(copy_nxg)
			components.append([vertex, 1] if c_count>connected_component_count else [vertex, 0])
		
		articulations_df = sqlContext.createDataFrame(components, ['id', 'articulation'])
		return articulations_df	

		

filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
df.filter('articulation = 1').toPandas().to_csv('articulation_out.csv')
print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
df.filter('articulation = 1').toPandas().to_csv('articulation_out2.csv')
