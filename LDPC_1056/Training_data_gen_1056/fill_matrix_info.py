import numpy as np

class Code:
    def __init__(self):
        self.num_edges = 0
    def load_code(H_filename):
    	# parity-check matrix; Tanner graph parameters
    	# H_filename = format('./LDPC_matrix/LDPC_576_432.alist')
    	# G_filename = format('./LDPC_matrix/LDPC_576_432.gmat')
        with open(H_filename,'rt') as f:
            line= str(f.readline()).strip('\n').split(' ')
    		# get n and m (n-k) from first line
            n,m = [int(s) for s in line]
            #for EG_LDPC-273_191 only
            k = n-m
           
    #################################################################################################################
            var_degrees = np.zeros(n).astype(np.int) # degree of each variable node
            chk_degrees = np.zeros(m).astype(np.int) # degree of each check node
    
    		# initialize H
            H = np.zeros([m,n]).astype(np.int)
            line =  str(f.readline()).strip('\n').split(' ')
            max_var_degree, max_chk_degree = [int(s) for s in line]
            line =  str(f.readline()).strip('\n').split(' ')
           # var_degree_dist = [int(s) for s in line[0:-1]] 
            line =  str(f.readline()).strip('\n').split(' ')
           # chk_degree_dist = [int(s) for s in line[0:-1]]
    
            var_edges = [[] for _ in range(0,n)]
            for i in range(0,n):
                line =  str(f.readline()).strip('\n').split(' ')
                var_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
                var_degrees[i] = len(var_edges[i])
                H[var_edges[i], i] = 1
      
            chk_edges = [[] for _ in range(0,m)]
            for i in range(0,m):
                line =  str(f.readline()).strip('\n').split(' ')
                chk_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
                chk_degrees[i] = len(chk_edges[i])
      
    ################################################################################################################
    # numbering each edge in H with a unique number whether horizontally or vertically       
            d = [[] for _ in range(0,n)]
            edge = 0
            for i in range(0,n):
                for j in range(0,var_degrees[i]):
                    d[i].append(edge)
                    edge += 1
      
            u = [[] for _ in range(0,m)]
            edge = 0
            for i in range(0,m):
                for j in range(0,chk_degrees[i]):
                    v = chk_edges[i][j]
                    for e in range(0,var_degrees[v]):
                        if (i == var_edges[v][e]):
                            u[i].append(d[v][e])
    
            num_edges = H.sum()  
         
        code = Code()
        code.H = H
        code.var_degrees = var_degrees
        code.chk_degrees = chk_degrees
        code.num_edges = num_edges
        code.u = u
        code.d = d
        code.check_matrix_column = n
        code.check_matrix_row = m
        code.k = k
        code.var_edges=var_edges
        code.chk_edges = chk_edges
        code.max_chk_degree = max_chk_degree
        code.max_var_degree = max_var_degree
        return code

        

