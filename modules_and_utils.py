##############################
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
#from torch_geometric.utils import scatter_
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
import gurobipy as gp
from gurobipy import GRB
###########

class GATAConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATAConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



def propagate(x, edge_index):
    row, col = edge_index
    out = scatter_add(x[col], row, dim=0)
    return out

def get_mask(x, edge_index, hops):
    for k in range(hops):
        x = propagate(x, edge_index)
    mask = (x>0).float()
    return mask


def total_var(x, edge_index, batch, undirected = True):
    row, col = edge_index
    if undirected:
        tv = (torch.abs(x[row]-x[col])) * 0.5
    else:
        tv = (torch.abs(x[row]-x[col]))

    tv = scatter_add(tv, batch[row], dim=0)
    return  tv

def derandomize_cut(data, probabilities, target, elasticity,  draw=False):
    row, col = data.edge_index
    sets = probabilities.detach()
    deg = degree(row)
    no_loop_index,_ = remove_self_loops(data.edge_index)        
    no_loop_row, no_loop_col = no_loop_index
    total_index = 0

    for graph in range(data.batch.max().item()+1):
         exp_cut = scatter_add(sets*deg, data.batch.cuda(), 0) - scatter_add((sets[row]*sets[col]), data.batch[row].cuda(), 0)
         num_nodes = (data.batch==graph).sum().item()            
         graph_set = sets[data.batch==graph].detach()
         sorted_indices = torch.argsort(graph_set, descending=True)
         mark_edges = batch[row] == graph
         lr_graph, lc_graph = data.edge_index[:,mark_edges]
         lr_graph = lr_graph - total_index
         lc_graph = lc_graph - total_index

         for node in sorted_indices:
             ind_i = total_index + node

             if [ind_i] not in data.locations.tolist():
                 graph_set[node] = 0
                 cond_exp_cut_0 = (graph_set*deg[data.batch==graph]).sum() - (graph_set[lr_graph]*graph_set[lc_graph]).sum()
                 graph_set[node] = 1
                 vol_1 = (graph_set*deg[data.batch==graph]).sum() #compute here cause we're reusing
                 cond_exp_cut_1 = vol_1  -  (graph_set[lr_graph]*graph_set[lc_graph]).sum()

                 if cond_exp_cut_0 > cond_exp_cut_1:
                    if vol_1 <= target[graph]+elasticity*target[graph]:
                        sets[ind_i] = 1

                 else:
                    sets[ind_i] = 0
             else:
                sets[ind_i] = 1
         if draw: 
             dirac = data.locations[graph].item() - total_index
             f1 = plt.figure(graph,figsize=(16,9)) 
             ax1 = f1.add_subplot(121)
             g1,g2 = drawGraphFromData(data.to('cpu'), graph, vals=sets.cpu(), dense=False,seed=dirac, nodecolor=True,edgecolor=False,seedhops=True,hoplabels=True,binarycut=True)
             ax2 = f1.add_subplot(122)
             g1,g2 = drawGraphFromData(data.to('cpu'), graph, vals=probabilities.cpu(), dense=False,seed=dirac, nodecolor=True,edgecolor=False,seedhops=True,hoplabels=True,binarycut=True)

         total_index += num_nodes
         derand_cut = scatter_add(sets*deg, data.batch, 0) - scatter_add((sets[row]*sets[col]), data.batch[row], 0)         
    return sets


def get_diracs(data, N , n_diracs = 1,  sparse = False, flat = False, replace = True, receptive_field = 7, effective_volume_range = 0.1, max_iterations=20, complement = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not sparse:
        graphcount =data.num_nodes #number of graphs in data/batch object
        totalnodecount = data.x.shape[1] #number of total nodes for each graph 
        actualnodecount = 0 #cumulative number of nodes
        diracmatrix= torch.zeros((graphcount,totalnodecount,N),device=device) #matrix with dirac pulses


        for k in range(graphcount):
            graph_nodes = data.mask[k].sum() #number of nodes in the graph
            actualnodecount += graph_nodes #might not need this, we'll see
            probabilities= torch.ones((graph_nodes.item(),1),device=device)/graph_nodes #uniform probs
            node_distribution=OneHotCategorical(probs=probabilities.squeeze())
            node_sample= node_distribution.sample(sample_shape=(N,))
            node_sample= torch.cat((node_sample,torch.zeros((N,totalnodecount-node_sample.shape[1]),device=device)),-1) #concat zeros to fit dataset shape
            diracmatrix[k,:]= torch.transpose(node_sample,dim0=-1,dim1=-2) #add everything to the final matrix
            
        return diracmatrix
    
    else:
            if not is_undirected(data.edge_index):
                data.edge_index = to_undirected(data.edge_index)
                
            original_batch_index = data.batch
            original_edge_index = add_remaining_self_loops(data.edge_index, num_nodes = data.batch.shape[0])[0]
            batch_index = original_batch_index
            
            graphcount = data.num_graphs
            batch_prime = torch.zeros(0,device=device).long()
            
            r,c = original_edge_index
            
            
            global_offset = 0
            all_nodecounts = scatter_add(torch.ones_like(batch_index,device=device), batch_index,0)
            recfield_vols = torch.zeros(graphcount,device=device)
            total_vols = torch.zeros(graphcount,device=device)
            
            for j in range(n_diracs):
                diracmatrix = torch.zeros(0,device=device)
                locationmatrix = torch.zeros(0,device=device).long()
                for k in range(graphcount):
                    #get edges of current graph, remember to subtract offset
                    graph_nodes = all_nodecounts[k]
                    if graph_nodes==0:
                        print("all nodecounts: ", all_nodecounts)
                    graph_edges = (batch_index[r]==k)
                    graph_edge_index = original_edge_index[:,graph_edges]-global_offset           
                    gr, gc = graph_edge_index


                    #get dirac
                    randInt = np.random.choice(range(graph_nodes), N, replace = replace)
                    node_sample = torch.zeros(N*graph_nodes,device=device)
                    offs  = torch.arange(N, device=device)*graph_nodes
                    dirac_locations = (offs + torch.from_numpy(randInt).to(device))
                    node_sample[dirac_locations] = 1
                    #calculate receptive field volume and compare to total volume
                    mask = get_mask(node_sample, graph_edge_index.detach(), receptive_field).float()
                    deg_graph = degree(gr, (graph_nodes.item()))
                    total_volume = deg_graph.sum()
                    recfield_volume = (mask*deg_graph).sum()
                    volume_range = recfield_volume/total_volume
                    total_vols[k] = total_volume
                    recfield_vols[k] = recfield_volume
                    #if receptive field volume is less than x% of total volume, resample
                    for iteration in range(max_iterations):  
                        randInt = np.random.choice(range(graph_nodes), N, replace = replace)
                        node_sample = torch.zeros(N*graph_nodes,device=device)
                        offs  = torch.arange(N, device=device)*graph_nodes
                        dirac_locations = (offs + torch.from_numpy(randInt).to(device))
                        node_sample[dirac_locations] = 1

                        mask = get_mask(node_sample, graph_edge_index, receptive_field).float()
                        recfield_volume = (mask*deg_graph).sum()
                        volume_range = recfield_volume/total_volume

                        if volume_range > effective_volume_range:
                            recfield_vols[k] = recfield_volume
                            total_vols[k] = total_volume
                            break;
                    dirac_locations2 = torch.from_numpy(randInt).to(device) + global_offset
                    global_offset += graph_nodes

                    diracmatrix = torch.cat((diracmatrix, node_sample),0)
                    locationmatrix = torch.cat((locationmatrix, dirac_locations2),0)
            locationmatrix = diracmatrix.nonzero()
            if complement:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols, complement_edge_index = data.complement_edge_index)
            else:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols)

            
#slow version     
def decode_clique_final(data, probabilities, draw=False, weight_factor = 0.0, clique_number_bounds = None ,fig = None, device = 'cpu'):
    row, col = data.edge_index
    sets = probabilities.detach().unsqueeze(-1)
    batch = data.batch
    no_loop_index,_ = remove_self_loops(data.edge_index)        
    no_loop_row, no_loop_col = no_loop_index
    num_graphs = batch.max().item() + 1
    total_index = 0

    for graph in range(num_graphs):
        mark_edges = batch[no_loop_row] == graph
        nlr_graph, nlc_graph = no_loop_index[:,mark_edges]
        nlr_graph = nlr_graph - total_index
        nlc_graph = nlc_graph - total_index
        batch_graph = (batch==graph)
        graph_probs = sets[batch_graph].detach()
        sorted_inds = torch.argsort(graph_probs.squeeze(-1), descending=True)
        pairwise_prodsums = torch.zeros(1, device = device)
        pairwise_prodsums = (torch.conv1d(graph_probs.unsqueeze(-1), graph_probs.unsqueeze(-1))).sum()/2
        self_sums = (graph_probs*graph_probs).sum()       
        num_nodes = batch_graph.float().sum().item()
   
        current_set_cardinality = 0
        
        for node in range(int(num_nodes)):
            ind_i = total_index + sorted_inds[node]
            graph_probs_0 = sets[batch_graph].detach()
            graph_probs_1 = sets[batch_graph].detach()
            
            graph_probs_0[sorted_inds[node]] = 0
            graph_probs_1[sorted_inds[node]] = 1

            pairwise_prodsums_0 = torch.zeros(1, device = device)
            pairwise_prodsums_0 = (torch.conv1d(graph_probs_0.unsqueeze(-1),graph_probs_0.unsqueeze(-1))).sum()/2

            self_sums_0 = (graph_probs_0*graph_probs_0).sum()

            expected_weight_G_0 = (graph_probs_0[nlr_graph]*graph_probs_0[nlc_graph]).sum()/2
            expected_clique_weight_0 = (pairwise_prodsums_0 - self_sums_0)
            clique_dist_0 = weight_factor*0.5*(expected_clique_weight_0 - expected_weight_G_0)-expected_weight_G_0


            pairwise_prodsums_1 = torch.zeros(1, device = device)
            pairwise_prodsums_1 = (torch.conv1d(graph_probs_1.unsqueeze(-1),graph_probs_1.unsqueeze(-1))).sum()/2

            self_sums_1 = (graph_probs_1*graph_probs_1).sum()

            expected_weight_G_1 = (graph_probs_1[nlr_graph]*graph_probs_1[nlc_graph]).sum()/2
            expected_clique_weight_1 = (pairwise_prodsums_1 - self_sums_1)
            clique_dist_1 = weight_factor* 0.5*(expected_clique_weight_1 - expected_weight_G_1)-expected_weight_G_1

            if clique_dist_0 >= clique_dist_1: 
                decided = (graph_probs_1==1).float()
                current_set_cardinality = decided.sum().item()
                current_set_max_edges = (current_set_cardinality*(current_set_cardinality-1))/2
                current_set_edges = (decided[nlr_graph]*decided[nlc_graph]).sum()/2

                if (current_set_edges != current_set_max_edges):
                    sets[ind_i] = 0 #IF NOT A CLIQUE
                else:
                    sets[ind_i] = 1 #IF A CLIQUE
                    

            else:
                   sets[ind_i] = 0

        if draw: 
            dirac = data.locations[graph].item() - total_index
            if fig is None:
                 f1 = plt.figure(graph,figsize=(16,9)) 
            else:
                 f1 = fig
            ax1 = f1.add_subplot(121)
            g1,g2 = drawGraphFromData(data.to('cpu'), graph, vals=sets.squeeze(-1).detach().cpu(), dense=False,seed=dirac, nodecolor=True,edgecolor=False,seedhops=True,hoplabels=True,binarycut=False)
            ax2 = f1.add_subplot(122)
            g1,g2 = drawGraphFromData(data.to('cpu'), graph, vals=probabilities.detach().cpu(), dense=False,seed=dirac, nodecolor=True,edgecolor=False,seedhops=True,hoplabels=True,binarycut=False, clique = True)             
        total_index += num_nodes
        

    expected_weight_G = scatter_add(sets[no_loop_col]*sets[no_loop_row], batch[no_loop_row], 0, dim_size = num_graphs)
    set_cardinality = scatter_add(sets, batch, 0 , dim_size = num_graphs)
    return sets, expected_weight_G.detach(), set_cardinality


#fast version
def decode_clique_final_speed(data, probabilities, draw=False, weight_factor = 0.35, clique_number_bounds = None ,fig = None, device = 'cpu', beam = 1):
       
    row, col = data.edge_index
    sets = probabilities.detach().unsqueeze(-1)
    blank_sets = torch.zeros_like(probabilities)
    batch = data.batch
    
    no_loop_index,_ = remove_self_loops(data.edge_index)        
    no_loop_row, no_loop_col = no_loop_index
    num_graphs = batch.max().item() + 1
    
    max_cardinalities = torch.zeros(num_graphs)

    total_index = 0

    for graph in range(num_graphs):
        mark_edges = batch[no_loop_row] == graph
        nlr_graph, nlc_graph = no_loop_index[:,mark_edges]
        nlr_graph = nlr_graph - total_index
        nlc_graph = nlc_graph - total_index
        batch_graph = (batch==graph)
        graph_probs = sets[batch_graph].detach()
        sorted_inds = torch.argsort(graph_probs.squeeze(-1), descending=True)
        num_nodes = batch_graph.long().sum()        
        current_set_cardinality = 0
        target_neighborhood = torch.tensor([])
        node = 0
        max_width = beam
        if num_nodes>max_width:
            beam_width = max_width 
        else:
            beam_width = num_nodes
            
        max_beam_weight = 0
        max_weight_node = 0
        graph_probs_1 = sets[batch_graph].detach()
        max_cardinality = 0
        
        for node in range(beam_width):
            blank_sets[batch_graph] = 0
            current_set_cardinality = 0
            ind_i = total_index + sorted_inds[node]
            ind_i = total_index + sorted_inds[node]
            blank_sets[ind_i] = 1
            sets[ind_i] = 1 #IF A CLIQUE=
            target_neighborhood = torch.unique(nlc_graph[nlr_graph == sorted_inds[node]])
            decided = blank_sets[batch_graph]
            current_set_max_edges = (current_set_cardinality*(current_set_cardinality-1))/2
            current_set_edges = (decided[nlr_graph]*decided[nlc_graph]).sum()/2
            current_set_cardinality += 1
            neighborhood_probs =  graph_probs[target_neighborhood]
            neigh_inds = torch.argsort(neighborhood_probs.squeeze(-1), descending=True)
            sorted_target_neighborhood = target_neighborhood[neigh_inds]
            for node_2 in sorted_target_neighborhood:
                ind_i  = total_index + node_2
                blank_sets[ind_i] = 1
                sets[ind_i] = 1 #IF A CLIQUE
                current_set_cardinality += 1
                decided = blank_sets[batch_graph]
                current_set_max_edges = (current_set_cardinality*(current_set_cardinality-1))/2
                current_set_edges = (decided[nlr_graph]*decided[nlc_graph]).sum()/2

                if (current_set_edges != current_set_max_edges):
                    sets[ind_i] = 0 #IF NOT A CLIQUE
                    blank_sets[ind_i] = 0  
                    current_set_cardinality =  current_set_cardinality - 1

            if current_set_cardinality > max_cardinality:
                max_cardinality = current_set_cardinality
        max_cardinalities[graph] = max_cardinality
        if draw: 
            dirac = data.locations[graph].item() - total_index
            if fig is None:
                 f1 = plt.figure(graph,figsize=(16,9)) 
            else:
                 f1 = fig
            ax1 = f1.add_subplot(121)
            g1,g2 = drawGraphFromData(data.to('cpu'), graph, vals=sets.squeeze(-1).detach().cpu(), dense=False,seed=dirac, nodecolor=True,edgecolor=False,seedhops=True,hoplabels=True,binarycut=False)
            ax2 = f1.add_subplot(122)
            g1,g2 = drawGraphFromData(data.to('cpu'), graph, vals=probabilities.detach().cpu(), dense=False,seed=dirac, nodecolor=True,edgecolor=False,seedhops=True,hoplabels=True,binarycut=False, clique = True)      
        total_index += num_nodes

    expected_weight_G = scatter_add(blank_sets[no_loop_col]*blank_sets[no_loop_row], batch[no_loop_row], 0, dim_size = num_graphs)
    set_cardinality = scatter_add(blank_sets, batch, 0 , dim_size = num_graphs)
    return blank_sets, expected_weight_G.detach(), max_cardinalities




def solve_gurobi_maxclique(nx_graph, time_limit = None):

    nx_complement = nx.operators.complement(nx_graph)
    x_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag = 0

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_complement.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_complement.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
        count_edges+=1
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]), GRB.MAXIMIZE);


    # Optimize model
    m.optimize();

    set_size = m.objVal;
    x_vals = [var.x for var in m.getVars()] 

    return set_size, x_vals


def solve_gurobi_mis(nx_graph, costs, time_limit = None):

    x_vars = {}
    c_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag = 0

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_graph.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))
        
    for cost in costs:
        c_vars['c_' + str(node)] = m.addVar(name="c_"+str(node))

    count_edges = 0
    for edge in nx_graph.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
        count_edges+=1
    m.setObjective(sum([x_vars['x_'+str(node)]*c_vars['c_' + str(node)] for node in nx_graph.nodes()]), GRB.MAXIMIZE);


    # Optimize model
    m.optimize();

    set_size = m.objVal;
    x_vals = [var.x for var in m.getVars()] 

    return set_size, x_vals