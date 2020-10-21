import numpy as np
import random

class Graph:
    def __init__(self):
        self.M = np.ones((1,1))
        self.vertices = {}
        self.wait_edges = []
        self.deadline_edges = []
        self.names_of_vertex=[] #zero indexed
        self.num_tasks = 0
        self.key_counter = 0
        #TODO: Add iterator for task if name not given
        # self.add_vertex('start')
        # self.add_vertex('end')

    def print_solution(self):
        print(self.vertices)
        print(self.M)

    def add_vertex(self, name):
        vertex = Vertex(self.key_counter)
        self.vertices[self.key_counter] = vertex
        self.names_of_vertex.append(name)
        self.key_counter +=1

    def add_task_vertex_and_edges(self,task):
        vertex_start = Vertex(self.key_counter)
        vertex_end = Vertex(self.key_counter+1)
        self.vertices[self.key_counter] = vertex_start
        self.vertices[self.key_counter+1] = vertex_end
        self.names_of_vertex.append(task.getName() + '_start')
        self.names_of_vertex.append(task.getName() + '_end')
        self.add_edge(self.key_counter,self.key_counter+1, task.getc())
        self.add_edge(self.key_counter+1, self.key_counter, -1* task.getc())
        self.key_counter +=2

    def add_tasks_vertex_and_edges(self,tasks):
        for task in tasks:
            vertex_start = Vertex(self.key_counter)
            vertex_end = Vertex(self.key_counter+1)
            self.vertices[self.key_counter] = vertex_start
            self.vertices[self.key_counter+1] = vertex_end
            self.names_of_vertex.append(task.getName() + '_start')
            self.names_of_vertex.append(task.getName() + '_end')
            self.add_edge(self.key_counter,self.key_counter+1, task.getc())
            self.add_edge(self.key_counter+1, self.key_counter, -1* task.getc())
            self.key_counter +=2

    def get_vertex(self, key):
        """Return vertex object with the corresponding key."""
        return self.vertices[key]

    def is_node_added(self,key):
        return key in self.vertices

    def add_edge(self, src_key, dest_key, weight=1):
        """Add edge from src_key to dest_key with given weight."""
        self.vertices[src_key].add_neighbor(self.vertices[dest_key], weight)

    def add_edge_by_name(self, src_name, dest_name, weight=1):
        src_key = self.names_of_vertex.index(src_name)
        dest_key = self.names_of_vertex.index(dest_name)
        self.add_edge(src_key, dest_key, weight)

    def add_task_edge_by_name(self, src_name, dest_name, weight=1):
        src_key = self.names_of_vertex.index(src_name + '_end')
        dest_key = self.names_of_vertex.index(dest_name + '_start')
        self.add_edge(src_key, dest_key, weight)

    def add_movement_constraint_by_name(self, src_name, weight=1):
        src_key = self.names_of_vertex.index(src_name + '_start')
        dest_key = self.names_of_vertex.index('start')
        self.add_edge(src_key, dest_key, -weight)
        self.add_edge(dest_key, src_key, weight)

    def does_edge_exist(self, src_key, dest_key):
        """Return True if there is an edge from src_key to dest_key."""
        return self.vertices[src_key].does_it_point_to(self.vertices[dest_key])

    def how_many_nodes(self):
        return len(self.vertices)



    def build_M_matrix(self):
        n = len(self.vertices)
        M = np.ones((n,n)) * np.inf
        gamma = dict((v,k) for k,v in self.vertices.items())
        np.fill_diagonal(M,0)

        # for each node
        for vertex_num in self.vertices:
            vertex_object = self.vertices[vertex_num]
            # check where it goes
            what_it_points_to = vertex_object.points_to
            start = vertex_num
            # for each connection
            for each_node in what_it_points_to:
                # add to M
                finish = gamma[each_node]
                M[start][finish] = what_it_points_to[each_node] # the weight
                # for each edge, assign the value by (row,column) as (start,finish)

        self.M = M

    def initialize_all_start_and_end_nodes(self, tasks):
        # each pair of 2 is a task_start and task_finish
        for task in tasks:
            self.add_edge_by_name(task.getName() +'_start','start', 0)
            self.add_edge_by_name('end', task.getName() + '_end', 0)



    def compute_floyd_warshal(self):
        dist = self.M.copy()
        n = self.M.shape[0]
        # print(n)
        for k in range(0,n):
            for i in range(0,n):
                for j in range(0,n):
                    dist[i][j] = min(dist[i][j],
                                     dist[i][k] + dist[k][j]
                                     )
        # print(self.M)
        # print(dist)
        self.M = dist


    def __iter__(self):
        return iter(self.vertices.values())

    def get_random_wait_constraints(self,tasks):
        num_tasks = len(tasks)
        arr_of_wait_deadlines = []
        for main_task in tasks:
            for other_task in tasks:
                if main_task == other_task:
                    continue
                i = random.randint(1, 50)  # could also do it by prob
                if i == 1:
                    weight = random.randint(1, 10)
                    self.add_edge_by_name(main_task.getName() + '_start', other_task.getName() + '_end', -1*weight)
                    arr_of_wait_deadlines.append([other_task, main_task, weight])
        return arr_of_wait_deadlines

    def get_random_deadline_constraints(self, tasks):
        num_tasks = len(tasks)
        dict_of_deadlines = {}
        for main_task in tasks:
            i = random.randint(1, 6) # could also do it by prob
            if i == 1:
                weight = random.randint(15,100)
                self.add_edge_by_name('start', main_task.getName() + '_end', weight)
                dict_of_deadlines[main_task] = weight
        return dict_of_deadlines

    def print_checking(self):
        self.gamma = dict((v, k) for k, v in self.vertices.items())
        print(self.names_of_vertex)
        for i in range(0,len(self.vertices)):
            print(self.names_of_vertex[i], " points to ")
            for element in self.vertices[i].points_to:
                num = self.gamma[element]
                name_of_ele = self.names_of_vertex[num]
                print(name_of_ele, ": ", self.vertices[i].points_to[element])
            print('----------------------')



    def is_feasible(self):
        diag_elements = self.M.diagonal()
        is_feasible = True
        for element in diag_elements:
            if element != 0:
                is_feasible = False
                return is_feasible

        return is_feasible






class Vertex: # stores all vertices
    def __init__(self,key):
        self.key = key
        self.points_to = {}

    def get_key(self):
        """ What number vertex is it"""
        return self.key

    def add_neighbor(self,dest,weight):
        """
        adds a destination and weight,
        dest should be the key of another vertex
        """
        self.points_to[dest] = weight

    def get_neighbors(self):
        """gets all neighbors of a task"""
        return self.points_to.keys()

    def get_weight(self, dest):
        """returns weight on edge"""
        return self.points_to[dest]

    def does_it_point_to(self, dest):
        """returns a bool if it points to something"""
        return dest in self.points_to

