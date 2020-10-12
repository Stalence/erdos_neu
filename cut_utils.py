import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import convert as cnv
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from torch.distributions import categorical
from torch.distributions import Bernoulli
from torch.distributions import relaxed_categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from matplotlib import pyplot as plt
import numpy as np
from torch_geometric.utils import is_undirected, to_undirected, softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
import time
import gurobipy as gp
from gurobipy import GRB

from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import dropout_adj
#from torch_geometric.utils import scatter
from torch_geometric.utils import degree
from torch_geometric.data import Batch 

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean


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
            #original_edge_index, _, node_mask = remove_isolated_nodes(original_edge_index)
            #batch_index = original_batch_index[node_mask]
            batch_index = original_batch_index
            
            graphcount = data.num_graphs #number of graphs in data/batch object
            batch_prime = torch.zeros(0,device=device).long()
            
            r,c = original_edge_index
            
            
            global_offset = 0
            all_nodecounts = scatter_add(torch.ones_like(batch_index,device=device), batch_index,dim=0)
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
            
    #                 print("Gr: ", gr)
    #                 print("Graph edge index: ", graph_edge_index)
    #                 print("gr max: ", gr.max())



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
             
                
            
                #for batch prime
#                 dirac_indices = torch.arange(N, device=device).unsqueeze(-1).expand(-1, graph_nodes).contiguous().view(-1)
#                 dirac_indices = dirac_indices.long()
#                 dirac_indices += k*N
#                 batch_prime = torch.cat((batch_prime, dirac_indices))



            locationmatrix = diracmatrix.nonzero()
            if complement:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols, complement_edge_index = data.complement_edge_index)
            else:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols)

class GATAConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

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



def is_clique(nx_graph, nodes):
    n_clique_edges = len(nodes)*(len(nodes)-1)/2
    self_loops = nx_graph.selfloop_edges()
    
    if (len(list(self_loops))>0):
        nx_graph.remove_edges_from(nx_graph.selfloop_edges())
        
    nx_subgraph = nx.subgraph(nx_graph, nodes)
    n_subgraph_edges = len(list(nx_subgraph.edges()))
    
    
    return ((n_clique_edges-n_subgraph_edges)==0), n_subgraph_edges
    

def solve_ortools_maxclique(nx_graph, solver=None, time_limit = None):  
    
    t0 = time.time()    
    solver = []
    solver = pywraplp.Solver('simple_mip_program',  pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    if time_limit:
        solver.set_time_limit(time_limit)
    
    infinity = solver.infinity()
    x_vars = {}
    nx_complement = nx.operators.complement(nx_graph)
    for node in nx_complement.nodes():
        x_vars['x_'+str(node)] = solver.IntVar(0.0, 1.0, 'x_'+str(node))
    for edge in nx_complement.edges():
        solver.Add(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1)
    solver.Maximize(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]))
    t1 = time.time()  
    status = solver.Solve()
    t2 = time.time()

    set_size = solver.Objective().Value()
    x_vals = [x_vars[key].solution_value() for key in x_vars.keys()]
    
    return  set_size, x_vals, [t2-t1, t1-t0]


import gurobipy as gp
from gurobipy import GRB

def solve_gurobi_maxclique(nx_graph, time_limit = None):

    nx_complement = nx.operators.complement(nx_graph)
    x_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag = 0

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_complement.nodes():
        # Create a new model

        # Create variables
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_complement.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
        count_edges+=1
    # Set objective
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]), GRB.MAXIMIZE);


    # Optimize model
    m.optimize();

# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

# print('Obj: %g' % m.objVal)
    set_size = m.objVal;
    x_vals = [var.x for var in m.getVars()] 

    return set_size, x_vals



import visdom 
from visdom import Visdom 
import numpy as np
import matplotlib.pyplot as plt




def plot_grad_flow( named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.yscale('log')
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8097)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[title_name], name=split_name, update = 'append')
    

    def histog(self,title_name,vals):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.histogram(X=vals,env=self.env,opts=dict(title=title_name,numbins=20))
        else:
            self.viz.histogram(X=vals,env=self.env,win=self.plots[title_name],opts=dict(title=title_name,numbins=20, update = 'replace'))
  
    def gradflow(self, model, title_name):
                    title_name = "Gradflow"
                    layers = []
                    ave_grads = []
                    for n, p in net.named_parameters():
                            if(p.requires_grad) and ("bias" not in n):
                                     layers.append(str(n))
                                     ave_grads.append(p.grad.abs().mean().cpu().numpy())
                    if title_name not in self.plots:
                        self.plots[title_name]= self.viz.line(X= list(range(len(layers))),Y= np.array(ave_grads),  env = self.env,  opts=dict(fillarea=True, title = title_name, xtick = True,
        xtickmin=0, xtickmax=len(layers), xtickvals = list(range(len(layers))), xtickstep=1/len(layers), xticklabels = layers) )    
                    else:
                        self.viz.line(X= list(range(len(layers))), Y=ave_grads, env = self.env ,win=self.plots[title_name], opts=dict(fillarea=True, title = title_name, xtick = True,
        xtickmin=0, xtickmax=len(layers), xtickvals = list(range(len(layers))), xtickstep=1/len(layers), xticklabels = layers, update="append")) 
                        
 

def set_up_facebook_data():               
    dataset = []
    upper_limit = 15000
    lower_limit = 0

    for file in os.listdir("/home/karalias/myrepos/CutMPNN/data/facebook/facebook100/"):
        if file.endswith(".mat"):
            adj_matrix = scipy.io.loadmat('/home/karalias/myrepos/CutMPNN/data/facebook/facebook100/'+str(file))
            edge_index = from_scipy_sparse_matrix(adj_matrix['A'])[0]
            x = torch.ones(adj_matrix['local_info'].shape[0])
            if (adj_matrix['local_info'].shape[0] < lower_limit) or (adj_matrix['local_info'].shape[0] > upper_limit):
                continue
            data_temp = Batch(x = x, edge_index = edge_index.long(), batch = torch.zeros_like(x).long())
            data_proper = get_diracs(data_temp.to('cuda'), 1, sparse = True)
            r,c = data_proper.edge_index
            data = Batch(x = data_temp.x, edge_index = data_temp.edge_index)
            #G = to_networkx(data)

            degrees = degree(r, adj_matrix['local_info'].shape[0])

            print("Graph specs: ")
            print("number of nodes: ", adj_matrix['A'].shape[0])
            print("average degree: ", degrees.mean(0))
            print("total volume: ", data_proper.total_vol)
            print("-------------")


            dataset += [data]

    dataset_name = "facebook_graphs"
    return dataset, dataset_name


def set_up_twitter_data():
    dataset = []

    data_dir = "/home/karalias/myrepos/CutMPNN/data/twitter/twitter/"
    for file in os.listdir(data_dir):
        if file.endswith(".edges"):
              G = nx.read_edgelist(data_dir+file)
              data = cnv.from_networkx(G.to_undirected())
              data.x = torch.ones(data.num_nodes)
              dataset += [data]


    datasetname = "twitter_graphs"



def prepare_dataset(dataset_name = "IMDB-BINARY", random_seed = 200):
#     rseed = 201
#     datasetname = "IMDB-BINARY"
    dataset = get_dataset(datasetname,sparse=1)
    # stored_dataset = open('myrepos/CliqueMPNN/pickles/dataset_shuffle_1'+'.p', 'rb')
    # dataset = pickle.load(stored_dataset)



    dataset_scale = 1.
    total_samples = int(np.floor(len(dataset)*dataset_scale))
    dataset = dataset[:total_samples]

    num_trainpoints = int(np.floor(0.6*len(dataset)))
    num_valpoints = int(np.floor(num_trainpoints/3))
    num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)


    traindata= dataset[0:num_trainpoints]
    valdata = dataset[num_trainpoints:num_trainpoints + num_valpoints]

    testdata = dataset[num_trainpoints + num_valpoints:]

    batch_size = 32

    train_loader = DataLoader(traindata, batch_size, shuffle=True)
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    val_loader =  DataLoader(valdata, batch_size, shuffle=False)

    #set up random seed 
    torch.manual_seed(rseed)
    np.random.seed(2)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def drawGraphFromData(data, index, vals=None, dense=True,seed=None,nodecolor=False,edgecolor=False,seedhops=False,hoplabels=False,binarycut=False, clique=False):   
    if dense:
        myx, myadj, mask = data.x[index], data.adj[index], data.mask[index]    
        if myx.unsqueeze(-1).shape[1]>1:
            myx=myx[:,0]

        #pad x values to fit standardized form
        newx = padToData(myx,Data(x=myx,adj = myadj))

        #convert to nx graph
        G=cnv.to_networkx(getSparseData(myx,myadj,mask))
        G=G.to_undirected()
    else:
        graph_ind = (data.batch==index)
        row,col = data.edge_index
        edge_ind = (data.batch[row]==index)
        offset = (data.batch<index).sum()
        
        if vals is not None: 
              myx = vals[graph_ind]
        else:
              myx = data.x[graph_ind]
        edge_list, batch =  torch.cat((row[edge_ind].unsqueeze(-1), col[edge_ind].unsqueeze(-1)), dim=1), data.batch
        
        edge_list = edge_list - offset

        
        newdata = Data(x=myx, edge_index = edge_list.t())
        G=cnv.to_networkx(newdata)
        G=G.to_undirected()
        if clique: 
            my_cliques = list(find_cliques(G))
            clique_nodes = max(my_cliques,key=len)
            myx[list(clique_nodes)] = 1
            clique_complement = G.nodes() - clique_nodes
            myx[list(clique_complement)] = 0
            
        
            
            
    
    pos= graphviz_layout(G)
    nofnodes= G.number_of_nodes()
        
    if nodecolor:
        #initialize color matrices for the plots
        nodecolors=torch.zeros(nofnodes,3) 
        if dense:
            colv = myx[:nofnodes]
        else:
            colv = myx
            
        #colv=torch.log(colv+1e-6)
        colv=(colv-colv.min())/(colv.max()-colv.min())

        #assign the values to the red channel
        nodecolors[:,0]= colv

        #assign some constant value to other channels, black nodes can be confusing
        nodecolors[:nofnodes,1]= 0.0*torch.ones_like(colv)
        nodecolors[:nofnodes,2]= 0.0*torch.ones_like(colv)
        
        if seedhops == True:
        
            positions={}
            withoutseed={}
            theseed=seed
            shortestpaths=nx.shortest_path_length(G,theseed)
            maxpath= list(shortestpaths.values())[-1] 
            orderpaths = {}
            orderpathswoseed={}
            
            
            for index in range(nofnodes):
                
                if nx.has_path(G,seed,index):
                    orderpaths[index] = str(shortestpaths[index])    
                    if index != theseed:
                        positions[index] = pos[index]
                        orderpathswoseed[index]=shortestpaths[index]
                else:
                    orderpaths[index] = -1 
                    positions[index]=pos[index]
                    orderpathswoseed[index]=-1

       
    
            if binarycut:
                cutnodes = list(myx.nonzero().reshape(-1).numpy())
                cutedg = nx.edge_boundary(G,cutnodes)
                cutedges = []
                for k in cutedg:
                    cutedges += [k]
                cutedges = set(cutedges)
                cutpaths = []
                cutpos = {}
                for i in cutnodes:
                    if nx.has_path(G,seed,i):
                        cutpaths += [shortestpaths[i]]
                        cutpos[i] = pos[i]
                    else:
                        cutpaths += [-1]
                        cutpos[i] = pos[i]
                nx.draw_networkx_nodes(G,cutpos,nodelist=cutnodes,alpha=0.5,node_color=[[1, 0, 0]],node_shape='o',node_size=1000)

            else:
                #print(list(positions.keys()))
                #print("nodecolor ", nodecolors)
                nx.draw_networkx_nodes(G,pos,nodelist=list(pos.keys()),alpha=0.85,node_color=nodecolors,node_shape='o',node_size=1400)

            for key in shortestpaths:
                if key != theseed:
                    scale = 1- shortestpaths[key]/maxpath
                    nodecolors[key,2] = scale*0.7 + 0.3
                    nodecolors[key,1] = scale*0.7 + 0.3
                    withoutseed[key] = key
                else:
                    scale = 1- shortestpaths[key]/maxpath
                    position = {key: pos[key]}
                    
                    nx.draw_networkx_nodes(G,position,alpha=1.0,nodelist=[key],node_color='r',node_size=1200*scale)


            nx.draw_networkx_nodes(G,positions,alpha=0.65,nodelist=list(positions.keys()),node_color=list(orderpathswoseed.values()),vmin=0,vmax=maxpath,cmap=plt.cm.viridis,node_size=350)
            if hoplabels:
                nx.draw_networkx_labels(G,pos,labels=orderpaths,font_color='k',alpha=0.75)
                

    else:
        nodecolors = 'r'
    
    
    if seedhops == False:
        nx.draw_networkx_nodes(G,pos,alpha=0.5,node_color=nodecolors,node_size=200)

    if edgecolor:
        edgecolors= torch.zeros(G.number_of_edges(),3)
        count=0
        edgecolvec= torch.zeros(G.number_of_edges())
        for i in G.edges():
            edgecolvec[count] = data.adj[i]
            count+=1;
        print(edgecolvec)
        edgecolors[:,1]= edgecolvec
        edgecolors[:,0]= 0.2*torch.ones_like(edgecolvec)
        edgecolors[:,2]= 0.2*torch.ones_like(edgecolvec)  
        
        nx.draw_networkx_edges(G,pos,alpha=1, width=edgecolvec.numpy())

    else:
        edgecolor= None
        if binarycut == False:
            nx.draw_networkx_edges(G,pos,alpha=0.5)
        else:
            nx.draw_networkx_edges(G,pos,G.edges()-cutedges,alpha=0.5)
            nx.draw_networkx_edges(G,pos,cutedges,alpha=0.5,width=5,edge_color='r')

    
    return G, pos




def nx_true_clique(loader):
    count = 0
    nx_true_nodes = []
    nx_true_edges = []
    nx_true_times = []
    for data in loader:
        t_0 = time.time()
        print("Current graph: ", count)
        print("number of nodes: ", data.x.shape[0])

        nx_graph = cnv.to_networkx(data).to_undirected()

        maxclique_number= graph_clique_number(nx_graph)
        maxclique_edges_true = (maxclique_number*(maxclique_number-1))/2


        print("Clique number: ", maxclique_number)
        print("-----------")
        t_1 = time.time()-t_0

        nx_true_times += [t_1]
        nx_true_nodes += [maxclique_number]
        nx_true_edges += [maxclique_edges_true]


        count += 1
        

def get_diracs_2(data, N , sparse = False, flat = False, replace = True, receptive_field = 7, effective_volume_range = 0.1, max_iterations=20):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

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
            #original_edge_index, _, node_mask = remove_isolated_nodes(original_edge_index)
            #batch_index = original_batch_index[node_mask]
            batch_index = original_batch_index
            
            graphcount = data.num_graphs #number of graphs in data/batch object
            diracmatrix = torch.zeros(0,device=device)
            batch_prime = torch.zeros(0,device=device).long()
            locationmatrix = torch.zeros(0,device=device).long()
            
            r,c = original_edge_index
            
            
            global_offset = 0
            all_nodecounts = scatter_("add", torch.ones_like(batch_index,device='cpu'), batch_index)
            recfield_vols = torch.zeros(graphcount,device=device)
            total_vols = torch.zeros(graphcount,device=device)
            

            for k in range(graphcount):
                #get edges of current graph, remember to subtract offset
                graph_nodes = all_nodecounts[k]
                if graph_nodes==0:
                    print("all nodecounts: ", all_nodecounts)
                graph_edges = (batch_index[r]==k)
                graph_edge_index = original_edge_index[:,graph_edges]-global_offset           
                gr, gc = graph_edge_index
                
#                 print("Gr: ", gr)
#                 print("Graph edge index: ", graph_edge_index)
#                 print("gr max: ", gr.max())
                
                
                
                #get dirac
                randInt = np.random.choice(range(graph_nodes), N, replace = replace)
                node_sample = torch.zeros(N*graph_nodes,device='cpu')
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
             
                
            
                #for batch prime
#                 dirac_indices = torch.arange(N, device=device).unsqueeze(-1).expand(-1, graph_nodes).contiguous().view(-1)
#                 dirac_indices = dirac_indices.long()
#                 dirac_indices += k*N
#                 batch_prime = torch.cat((batch_prime, dirac_indices))



            locationmatrix = diracmatrix.nonzero()


            return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                         y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols)

 


class cliqueMPNN_hindsight(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden1, hidden2, deltas, elasticity=0.01, num_iterations = 30):
        super(cliqueMPNN_hindsight, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.momentum = 0.1
        #self.nns = Sequential(
                #Linear(2*hidden, hidden*hidden),
                #LeakyReLU(0.1))
        self.conv1 = GINConv(Sequential(
            Linear(1,  self.hidden1),
            ReLU(),
            Linear(self.hidden1, self.hidden1),
            ReLU(),
            BN( self.hidden1, momentum=self.momentum ),
        ),train_eps=False)
        self.num_iterations = num_iterations
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.elasticity = elasticity
        self.heads = 8
        self.concat = True
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(BN( self.hidden1, momentum=self.momentum))

        
#         self.nns = torch.nn.ModuleList()
#         for i in range(num_layers-1):
#             self.nns.append(Sequential(
#                 Linear(2*hidden, hidden*hidden),
#                 LeakyReLU(0.1)))
        
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
            Linear( self.hidden1,  self.hidden1),
            ReLU(),
            Linear( self.hidden1,  self.hidden1),
            ReLU(),
            BN(self.hidden1, momentum=self.momentum),
        ),train_eps=True))
     
        #self.bn1 = BN(hidden)
        self.conv2 = GATAConv( self.hidden1, self.hidden2, concat=self.concat ,heads=self.heads)
        if self.concat:
            self.lin1 = Linear(self.heads*self.hidden2, self.hidden1)
        else:
            self.lin1 = Linear(self.hidden2, self.hidden1)

        self.bn2 = BN(self.hidden1, momentum=self.momentum)
        self.lin2 = Linear(self.hidden1, 1)
                    


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        #self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        self.bn2.reset_parameters()
        self.lin2.reset_parameters()






    def forward(self, data, edge_dropout = None, penalty_coefficient = 0.25):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        num_graphs = batch.max().item() + 1
        row, col = edge_index     
        total_num_edges = edge_index.shape[1]

        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr = (torch.ones(edge_index.shape[1], device=device)).long(), p = edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]
                
        reduced_num_edges = edge_index.shape[1]
        current_edge_percentage = (reduced_num_edges/total_num_edges)
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index

        xinit= x.clone()

                
        mask = get_mask(x,edge_index,1).to(x.dtype).unsqueeze(-1)  
        x = self.conv1(x.unsqueeze(-1), edge_index)
        x = x*mask

            
        for conv, bn in zip(self.convs, self.bns):
            if(x.dim()>1):
                x = x + conv(x, edge_index)
                mask = get_mask(mask,edge_index,1).to(x.dtype)
                x = x*mask
                x = bn(x)


        x = self.conv2(x, edge_index)
        mask = get_mask(mask,edge_index,1).to(x.dtype)
        x = x*mask
        xpostconvs = x.detach()
        #
        x = F.leaky_relu(self.lin1(x)) 
        x = x*mask
        x = self.bn2(x)


        xpostlin1 = x.detach()
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.lin2(x)) 
        x = x*mask
        

        #xprethresh = x.detach()

        N_size = x.shape[0]
        
        #calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)

                
        x = (x-batch_min)/(batch_max+1e-6-batch_min)
        x = x*mask + mask*1e-6
        

        probs=x
        
        x2 = x.detach()              
        deg = degree(row).unsqueeze(-1) 
        totalvol = scatter_add(deg.detach()*torch.ones_like(x, device=device), batch, 0)+1e-6
        totalcard = scatter_add(torch.ones_like(x, device=device), batch, 0)+1e-6
        
                
        #volume within receptive field
        recvol_hard = scatter_add(deg*mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        reccard_hard = scatter_add(mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        
        assert recvol_hard.mean()/totalvol.mean() <=1, "Something went wrong! Receptive field is larger than total volume."

        
        
        x2 =  ((probs - torch.rand_like(x, device = device))>0).float()
        
        vol_1 = scatter_add(probs*deg, batch, 0)+1e-6
        card_1 = scatter_add(probs, batch,0)            
        rec_field = scatter_add(mask, batch, 0)+1e-6
        set_size = scatter_add(x2, batch, 0)
#         tv_hard = total_var(x2, edge_index, batch)
        vol_hard = scatter_add(deg*x2, batch, 0, dim_size = batch.max().item()+1)+1e-6 
#         conduct_hard = tv_hard/vol_hard
            
        rec_field_ratio = set_size/rec_field
        rec_field_volratio = vol_hard/recvol_hard
        total_vol_ratio = vol_hard/totalvol
        
        #volume within receptive field
        recvol_hard = scatter_add(deg*mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        reccard_hard = scatter_add(mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        
        
        #calculating the terms for the expected distance between clique and graph
        pairwise_prodsums = torch.zeros(num_graphs, device = device)
        for graph in range(num_graphs):
            batch_graph = (batch==graph)
            pairwise_prodsums[graph] = (torch.conv1d(probs[batch_graph].unsqueeze(-1), probs[batch_graph].unsqueeze(-1))).sum()/2
        
        self_sums = scatter_add((probs*probs), batch, 0, dim_size = num_graphs)/2
        
        #expected_weight_G = scatter_add(probs[no_loop_row]*probs[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2
        
        expected_weight_G = scatter_add(probs[no_loop_row]*probs[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2
        expected_clique_weight = (pairwise_prodsums.unsqueeze(-1) - self_sums)
        expected_distance = (expected_clique_weight - expected_weight_G)
        
        
        lambda_factors = (torch.rand((30,1), device=device))*penalty_coefficient-0.10
        
        
        #hindsight = torch.ones_like(lambda_factors)*expected_distance.unsqueeze(-1)*0.5 - lambda_factors*expected_weight_G.unsqueeze(-1)
        

       # print(expected_clique_weight.shape)
        #print(self_sums.shape)

       # expected_loss =  #torch.median(hindsight, 1)[0]
        #print(expected_loss.shape)
        expected_loss =expected_distance*0.5 - (penalty_coefficient)*expected_weight_G
        

        
        set_weight = (scatter_add(x2[no_loop_row]*x2[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2)+1e-6
        clique_edges_hard = (set_size*(set_size-1)/2) +1e-6
        clique_dist_hard = set_weight/clique_edges_hard
    
    
        clique_check = ((clique_edges_hard != clique_edges_hard))
        
        
        setedge_check  = ((set_weight != set_weight))
        
        
        
        
        assert ((clique_dist_hard>=1.1).sum())<=1e-6, "Invalid set vol/clique vol ratio."
        
        loss = expected_loss

        retdict = {}
        
        retdict["output"] = [probs.squeeze(-1),"hist"]   #output
        retdict["clique_check"] = [clique_edges_hard, "hist"]
        retdict["set_weight_check"] = [set_weight, "hist"]
        #retdict["|Expected_vol - Target|"]= [targetcheck.squeeze(-1), "hist"] #absolute distance from targetvol
        retdict["Expected_volume"] = [vol_1.mean(),"sequence"] #volume
        retdict["Expected_cardinality"] = [card_1.mean(),"sequence"]
        retdict["Set sizes"] = [set_size.squeeze(-1),"hist"]
        retdict["volume_hard"] = [vol_hard.mean(),"aux"] #volume2
        retdict["cardinality_hard"] = [set_size[0],"sequence"] #volumeq
        retdict["Expected weight(G)"]= [expected_weight_G.mean(), "sequence"]
        retdict["Expected maximum weight"] = [expected_clique_weight.mean(),"sequence"]
        retdict["Expected distance"]= [expected_distance.mean(), "sequence"]
        #retdict["cut1"] = [tv.mean(),"sequence"] #cut1
        #retdict["cut_hard"] = [tv_hard.mean(),"sequence"] #cut1
        #retdict["Average cardinality ratio of receptive field "] = [rec_field_ratio.mean(),"sequence"] 
        #retdict["Recfield volume/Total volume"] = [recvol_hard.mean()/totalvol.mean(), "sequence"]
        #retdict["Average ratio of receptive field volume"]= [rec_field_volratio.mean(),'sequence']
        retdict["Currvol/Cliquevol"] = [clique_dist_hard.mean(),'sequence']
        retdict["Currvol/Cliquevol all graphs in batch"] = [clique_dist_hard.squeeze(-1),'hist']
        retdict["Average ratio of total volume"]= [total_vol_ratio.mean(),'sequence']
        retdict["Current edge percentage"] = [torch.tensor(current_edge_percentage),'sequence']
        #retdict["Clique Weight hard"] = [clique_edges_hard[0], 'sequence']
        #retdict["Set weight"] = [set_weight[0], 'sequence']
        #retdict["mask"] = [mask, "aux"] #mask
        #retdict["xinit"] = [xinit,"hist"] #layer input diracs
        #retdict["xpostlin1"] = [xpostlin1.mean(1),"hist"] #after first linear layer
        #retdict["xprethresh"] = [xprethresh.mean(1),"hist"] #pre thresholding activations 195 x 1
        #retdict["xsoftbin"] = [xsoftbin.mean(1),"hist"] #soft binarized output
        #retdict["lossvol"] = [lossvol.mean(),"sequence"] #volume constraint
        #retdict["losscard"] = [losscard.mean(),"sequence"] #cardinality constraint
        retdict["loss"] = [loss.mean().squeeze(),"sequence"] #final loss

        return retdict
    
    def __repr__(self):
        return self.__class__.__name__

        
        





def propagate(x, edge_index):
    row, col = edge_index
    out = scatter_add( x[col], row, dim=0)
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





###############MODEL    
    
class cliqueMPNN_hindsight_earlyGAT(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden1, hidden2, deltas, elasticity=0.01, num_iterations = 30):
        super(cliqueMPNN_hindsight_earlyGAT, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.momentum = 0.1
        #self.nns = Sequential(
                #Linear(2*hidden, hidden*hidden),
                #LeakyReLU(0.1))
        self.num_iterations = num_iterations
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.elasticity = elasticity
        self.heads = 4
        self.concat = True
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(BN(self.heads*self.hidden1, momentum=self.momentum))

        
#         self.nns = torch.nn.ModuleList()
#         for i in range(num_layers-1):
#             self.nns.append(Sequential(
#                 Linear(2*hidden, hidden*hidden),
#                 LeakyReLU(0.1)))
        
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
                #self.convs.append(GATAConv(self.heads*self.hidden1, self.hidden1, concat=self.concat ,heads=self.heads))

#                 self.convs.append(GINConv(Sequential(
#             Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
#             ReLU(),
#             Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
#             ReLU(),
#             BN(self.heads*self.hidden1, momentum=self.momentum),
#         ),train_eps=True))
                self.convs.append(GCNConv2(self.heads*self.hidden1, self.heads*self.hidden1))
        self.bn1 = BN(self.heads*self.hidden1)
        self.conv1 = GATAConv(1, self.hidden1, concat=self.concat ,heads=self.heads)
#         self.conv1 = GINConv(Sequential(Linear( 1,  self.heads*self.hidden1),
#             ReLU(),
#             Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
#             ReLU(),
#             BN(self.heads*self.hidden1, momentum=self.momentum),
#         ),train_eps=True)

        if self.concat:
            self.lin1 = Linear(self.heads*self.hidden1, self.hidden1)
        else:
            self.lin1 = Linear(self.hidden1, self.hidden1)

        #self.bn2 = BN(self.hidden1, momentum=self.momentum)
        self.lin2 = Linear(self.hidden1, 1)
        self.gnorm = GraphSizeNorm()

                    


    def reset_parameters(self):
        self.conv1.reset_parameters()
        #self.conv2.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        #self.bn2.reset_parameters()
        self.lin2.reset_parameters()






    def forward(self, data, edge_dropout = None, penalty_coefficient = 0.25):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        num_graphs = batch.max().item() + 1
        row, col = edge_index     
        total_num_edges = edge_index.shape[1]

        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr = (torch.ones(edge_index.shape[1], device=device)).long(), p = edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]
                
        reduced_num_edges = edge_index.shape[1]
        current_edge_percentage = (reduced_num_edges/total_num_edges)
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index

        xinit= x.clone()

        
        #
#         x = x.squeeze(-1)

        mask = get_mask(x,edge_index,1).to(x.dtype).unsqueeze(-1) 
        
        #print(x.shape)

        x = F.leaky_relu(self.conv1(x.unsqueeze(-1), edge_index)) + x.unsqueeze(-1)
        #x = self.conv1(x, edge_index)
        #print(x.shape)
        x = self.gnorm(x)
        x = self.bn1(x)
        
        x = x*mask
        #print(x.shape)
            
        for conv, bn in zip(self.convs, self.bns):
            if(x.dim()>1):
                x =  x+F.leaky_relu(conv(x, edge_index))
                mask = get_mask(mask,edge_index,1).to(x.dtype)
                x = x*mask
                x = self.gnorm(x)
                x = bn(x)


#         x = self.conv2(x, edge_index)
#         mask = get_mask(mask,edge_index,1).to(x.dtype)
#         x = x*mask
        xpostconvs = x.detach()
        #
        x = F.leaky_relu(self.lin1(x)) 
        x = x*mask
        #x = self.gnorm(x)
        #x = self.bn2(x)


        xpostlin1 = x.detach()
        #x = F.dropout(x, p=0.0, training=self.training)
        x = F.leaky_relu(self.lin2(x)) 
       # x = self.gnorm(x)

        x = x*mask
        

        #xprethresh = x.detach()

        N_size = x.shape[0]
        
        #calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)

                
        x = (x-batch_min)/(batch_max+1e-6-batch_min)
        x = x*mask + mask*1e-6
        

        probs=x
        
        x2 = x.detach()              
        deg = degree(row).unsqueeze(-1) 
        totalvol = scatter_add(deg.detach()*torch.ones_like(x, device=device), batch, 0)+1e-6
        totalcard = scatter_add(torch.ones_like(x, device=device), batch, 0)+1e-6
        
                
        #volume within receptive field
        recvol_hard = scatter_add(deg*mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        reccard_hard = scatter_add(mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        
        assert recvol_hard.mean()/totalvol.mean() <=1, "Something went wrong! Receptive field is larger than total volume."

        
        
        x2 =  ((probs - torch.rand_like(x, device = device))>0).float()
        
        vol_1 = scatter_add(probs*deg, batch, 0)+1e-6
        card_1 = scatter_add(probs, batch,0)            
        rec_field = scatter_add(mask, batch, 0)+1e-6
        set_size = scatter_add(x2, batch, 0)
#         tv_hard = total_var(x2, edge_index, batch)
        vol_hard = scatter_add(deg*x2, batch, 0, dim_size = batch.max().item()+1)+1e-6 
#         conduct_hard = tv_hard/vol_hard
            
        rec_field_ratio = set_size/rec_field
        rec_field_volratio = vol_hard/recvol_hard
        total_vol_ratio = vol_hard/totalvol
        
        #volume within receptive field
        recvol_hard = scatter_add(deg*mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        reccard_hard = scatter_add(mask.float(), batch, 0, dim_size = num_graphs)+1e-6 
        
        
        #calculating the terms for the expected distance between clique and graph
        pairwise_prodsums = torch.zeros(num_graphs, device = device)
        for graph in range(num_graphs):
            batch_graph = (batch==graph)
            pairwise_prodsums[graph] = (torch.conv1d(probs[batch_graph].unsqueeze(-1), probs[batch_graph].unsqueeze(-1))).sum()/2
        
        self_sums = scatter_add((probs*probs), batch, 0, dim_size = num_graphs)/2
        
        #expected_weight_G = scatter_add(probs[no_loop_row]*probs[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2
        
        expected_weight_G = scatter_add(probs[no_loop_row]*probs[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2
        expected_clique_weight = (pairwise_prodsums.unsqueeze(-1) - self_sums)
        expected_distance = (expected_clique_weight - expected_weight_G)
        
        
        lambda_factors = (torch.rand((30,1), device=device))*penalty_coefficient-0.10
        
        
        #hindsight = torch.ones_like(lambda_factors)*expected_distance.unsqueeze(-1)*0.5 - lambda_factors*expected_weight_G.unsqueeze(-1)
        

       # print(expected_clique_weight.shape)
        #print(self_sums.shape)

       # expected_loss =  #torch.median(hindsight, 1)[0]
        #print(expected_loss.shape)
        expected_loss =(penalty_coefficient)*expected_distance- 0.5*expected_weight_G
        

        
        set_weight = (scatter_add(x2[no_loop_row]*x2[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2)+1e-6
        clique_edges_hard = (set_size*(set_size-1)/2) +1e-6
        clique_dist_hard = set_weight/clique_edges_hard
    
    
#         for iter_graph in range(num_graphs):
#             if clique_dist_hard[iter_graph]<0.5:
#                 print("problematic graph exp distance: ", expected_distance[iter_graph])
#                 print("problematic graph exp cardinality : ", card_1[iter_graph])

#                 print("problematic graph num nodes: ", totalcard[iter_graph])
                
        clique_check = ((clique_edges_hard != clique_edges_hard))
        
        
        setedge_check  = ((set_weight != set_weight))
        
        
        
        
        assert ((clique_dist_hard>=1.1).sum())<=1e-6, "Invalid set vol/clique vol ratio."
        
        loss = expected_loss
        #loss = expected_loss*(1/(clique_dist_hard+0.1))


        retdict = {}
        
        retdict["output"] = [probs.squeeze(-1),"hist"]   #output
        retdict["clique_check"] = [clique_edges_hard, "hist"]
        retdict["set_weight_check"] = [set_weight, "hist"]
        #retdict["|Expected_vol - Target|"]= [targetcheck.squeeze(-1), "hist"] #absolute distance from targetvol
        retdict["Expected_volume"] = [vol_1.mean(),"sequence"] #volume
        retdict["Expected_cardinality"] = [card_1.mean(),"sequence"]
        retdict["Expected_cardinality_hist"] = [card_1,"hist"]
        retdict["losses histogram"] = [loss.squeeze(-1),"hist"]
        retdict["Set sizes"] = [set_size.squeeze(-1),"hist"]
        retdict["volume_hard"] = [vol_hard.mean(),"aux"] #volume2
        retdict["cardinality_hard"] = [set_size[0],"sequence"] #volumeq
        retdict["Expected weight(G)"]= [expected_weight_G.mean(), "sequence"]
        retdict["Expected maximum weight"] = [expected_clique_weight.mean(),"sequence"]
        retdict["Expected distance"]= [expected_distance.mean(), "sequence"]
        #retdict["cut1"] = [tv.mean(),"sequence"] #cut1
        #retdict["cut_hard"] = [tv_hard.mean(),"sequence"] #cut1
        #retdict["Average cardinality ratio of receptive field "] = [rec_field_ratio.mean(),"sequence"] 
        #retdict["Recfield volume/Total volume"] = [recvol_hard.mean()/totalvol.mean(), "sequence"]
        #retdict["Average ratio of receptive field volume"]= [rec_field_volratio.mean(),'sequence']
        retdict["Currvol/Cliquevol"] = [clique_dist_hard.mean(),'sequence']
        retdict["Currvol/Cliquevol all graphs in batch"] = [clique_dist_hard.squeeze(-1),'hist']
        retdict["Average ratio of total volume"]= [total_vol_ratio.mean(),'sequence']
        retdict["Current edge percentage"] = [torch.tensor(current_edge_percentage),'sequence']
        #retdict["Clique Weight hard"] = [clique_edges_hard[0], 'sequence']
        #retdict["Set weight"] = [set_weight[0], 'sequence']
        #retdict["mask"] = [mask, "aux"] #mask
        #retdict["xinit"] = [xinit,"hist"] #layer input diracs
        #retdict["xpostlin1"] = [xpostlin1.mean(1),"hist"] #after first linear layer
        #retdict["xprethresh"] = [xprethresh.mean(1),"hist"] #pre thresholding activations 195 x 1
        #retdict["xsoftbin"] = [xsoftbin.mean(1),"hist"] #soft binarized output
        #retdict["lossvol"] = [lossvol.mean(),"sequence"] #volume constraint
        #retdict["losscard"] = [losscard.mean(),"sequence"] #cardinality constraint
        retdict["loss"] = [loss.mean().squeeze(),"sequence"] #final loss

        return retdict
    
    def __repr__(self):
        return self.__class__.__name__
    
    
    
    
class GCNConv2(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv2, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

#         print("before: ")
#         print(edge_weight.shape)
#         print(edge_index.shape)
        
        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        #edge_index = add_self_loops(edge_index, num_nodes)
#         loop_weight = torch.full((num_nodes, ),
#                                  1 if not improved else 2,
#                                  dtype=edge_weight.dtype,
#                                  device=edge_weight.device)
        #edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
#         print("after: ")
#         print(edge_weight.shape)
#         print(edge_index.shape)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    
    
class GATAConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

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


    
#####################################################VISDOM
def plot_grad_flow( named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.yscale('log')
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8097)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[title_name], name=split_name, update = 'append')
    

    def histog(self,title_name,vals):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.histogram(X=vals,env=self.env,opts=dict(title=title_name,numbins=20))
        else:
            self.viz.histogram(X=vals,env=self.env,win=self.plots[title_name],opts=dict(title=title_name,numbins=20, update = 'replace'))
  
    def gradflow(self, model, title_name):
                    title_name = "Gradflow"
                    layers = []
                    ave_grads = []
                    for n, p in net.named_parameters():
                            if(p.requires_grad) and ("bias" not in n):
                                     layers.append(str(n))
                                     ave_grads.append(p.grad.abs().mean().cpu().numpy())
                    if title_name not in self.plots:
                        self.plots[title_name]= self.viz.line(X= list(range(len(layers))),Y= np.array(ave_grads),  env = self.env,  opts=dict(fillarea=True, title = title_name, xtick = True,
        xtickmin=0, xtickmax=len(layers), xtickvals = list(range(len(layers))), xtickstep=1/len(layers), xticklabels = layers) )    
                    else:
                        self.viz.line(X= list(range(len(layers))), Y=ave_grads, env = self.env ,win=self.plots[title_name], opts=dict(fillarea=True, title = title_name, xtick = True,
        xtickmin=0, xtickmax=len(layers), xtickvals = list(range(len(layers))), xtickstep=1/len(layers), xticklabels = layers, update="append")) 


############################################DERANDOMIZATION
def derandomize_clique_final_speed(data, probabilities, draw=False, weight_factor = 0.35, clique_number_bounds = None ,fig = None, device = 'cpu', beam = 1):
       
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
        #torch.cuda.memory_summary()
        #print("NEW GRAPH-----------------------------------")
            #select subset of nodes and edges corresponding to graph
        mark_edges = batch[no_loop_row] == graph
        nlr_graph, nlc_graph = no_loop_index[:,mark_edges]
        nlr_graph = nlr_graph - total_index
        nlc_graph = nlc_graph - total_index
        batch_graph = (batch==graph)
        graph_probs = sets[batch_graph].detach()
        sorted_inds = torch.argsort(graph_probs.squeeze(-1), descending=True)

        
         #print(batch_graph)
#         pairwise_prodsums = torch.zeros(1, device = device)
#         pairwise_prodsums = (torch.conv1d(graph_probs.unsqueeze(-1), graph_probs.unsqueeze(-1))).sum()/2
#         self_sums = (graph_probs*graph_probs).sum()
        num_nodes = batch_graph.long().sum()
        #print(num_nodes)
        
        current_set_cardinality = 0
        target_neighborhood = torch.tensor([])
        #final_set = []
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
            #final_set += [ind_i]
            target_neighborhood = torch.unique(nlc_graph[nlr_graph == sorted_inds[node]])
            decided = blank_sets[batch_graph]
            #current_set_cardinality = decided.sum()
            #print("current card: ", current_set_cardinality)
            current_set_max_edges = (current_set_cardinality*(current_set_cardinality-1))/2
            current_set_edges = (decided[nlr_graph]*decided[nlc_graph]).sum()/2
            current_set_cardinality += 1
            neighborhood_probs =  graph_probs[target_neighborhood]
            #print(neighborhood_probs.shape)
            neigh_inds = torch.argsort(neighborhood_probs.squeeze(-1), descending=True)
            #if not neigh_inds:
            sorted_target_neighborhood = target_neighborhood[neigh_inds]


            for node_2 in sorted_target_neighborhood:
            #    print("here")
                ind_i  = total_index + node_2

                #if clique_dist_0 >= clique_dist_1: 
                blank_sets[ind_i] = 1
                sets[ind_i] = 1 #IF A CLIQUE
                current_set_cardinality += 1
                decided = blank_sets[batch_graph]
                current_set_max_edges = (current_set_cardinality*(current_set_cardinality-1))/2
                current_set_edges = (decided[nlr_graph]*decided[nlc_graph]).sum()/2

                if (current_set_edges != current_set_max_edges):
    #                 print("current edges: ", current_set_edges)
    #                 print("current max edges: ", current_set_max_edges)

                    sets[ind_i] = 0 #IF NOT A CLIQUE
                    blank_sets[ind_i] = 0  
                    current_set_cardinality =  current_set_cardinality - 1

            if current_set_cardinality > max_cardinality:
                max_cardinality = current_set_cardinality
            blank_sets[ind_i] = 0

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

            clique_size = len(list(max_clique(g1)))
#              print("NX Clique size is: ", clique_size)
#              print("True clique number: ", graph_clique_number(g1))
        
        
        total_index += num_nodes
        

    expected_weight_G = scatter_add(blank_sets[no_loop_col]*blank_sets[no_loop_row], batch[no_loop_row], 0, dim_size = num_graphs)
    set_cardinality = scatter_add(blank_sets, batch, 0 , dim_size = num_graphs)
    return blank_sets, expected_weight_G.detach(), max_cardinalities


# import gurobipy as gp
# from gurobipy import GRB

def solve_gurobi_mincut(nx_graph, seed=4, time_limit = None,target =5 ,tolerance=0.1):

    #nx_complement = nx.operators.complement(nx_graph)
    x_vars = {}
    y_vars = {}
    z_vars = {}
    
    m = gp.Model("mip1")
    #m.params.OutputFlag = 0
    print(len(list(my_graph.nodes())))

    if time_limit:
        m.params.TimeLimit = time_limit

    for node in nx_graph.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))
    for edge in nx_graph.edges():
        y_vars['y_'+str(edge[0])+str(edge[1])] = m.addVar(vtype=GRB.INTEGER, name ='y_'+str(edge[0])+str(edge[1]) )
        z_vars['z'+str(edge[0])+str(edge[1])] =  m.addVar(lb=0,ub=1, name ='z_'+str(edge[0])+str(edge[1]) )

    count_edges = 0
#     for edge in nx_complement.edges():
# #         m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] <= 1,'c_'+str(count_edges))
# #         count_edges+=1
    
    m.addConstr(sum([degree[1]*x_vars['x_'+str(degree[0])] for degree in nx_graph.degree()])<=target+tolerance*target,'c_1')
    m.addConstr(sum([degree[1]*x_vars['x_'+str(degree[0])] for degree in nx_graph.degree()])>=target-target*tolerance,'c_2')
    m.addConstr(x_vars['x_'+str(seed)]==1.0)
#     # Set objective
    count = 0
    for edge in nx_graph.edges():
    #     z = (y_vars['y_'+str(edge[0])+str(edge[1])]==[x_vars['x_'+str(edge[0])] - x_vars['x_'+str(edge[1])])
        m.addConstr(y_vars['y_'+str(edge[0])+str(edge[1])] == x_vars['x_'+str(edge[0])] - x_vars['x_'+str(edge[1])])
        #m.addConstr(z_vars['z'+str(edge[0])+str(edge[1])] == abs_(y_vars['y_'+str(edge[0])+str(edge[1])]))
        m.addGenConstrAbs(z_vars['z'+str(edge[0])+str(edge[1])], y_vars['y_'+str(edge[0])+str(edge[1])], "absconstr")
        
    m.setObjective(sum([z_vars['z'+str(edge[0])+str(edge[1])] for edge in nx_graph.edges()]), GRB.MINIMIZE);


    # Optimize model
    m.optimize();
    #m.update()
# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

# print('Obj: %g' % m.objVal)
    set_size = m.objVal;
    print()
    x_vals = [m.getVars()[i].x for i in range(nx_graph.number_of_nodes())] 
    print(x_vals)

    return set_size, x_vals

import gurobipy as gp
from gurobipy import GRB

def gurobi_mincut(data, seed=0, target = 10, tolerance = 0.5, budget = None):
    # Create a new model
    m = gp.Model("maxcut")
    r,c = data.edge_index
    degrees = degree(r.cpu())
    
    if budget:
        m.params.TimeLimit = budget

    x_vars = {}
    for i in range(data.num_nodes):
        x_vars["x_" + str(i)] = m.addVar(vtype=GRB.BINARY, name="x_" + str(i))

    obj = gp.QuadExpr()
    for source, target in zip(*data.edge_index.tolist()):
        qi_qj = (x_vars['x_' + str(source)] - x_vars['x_' + str(target)])
        obj += qi_qj * qi_qj / 2
        
    m.addConstr(x_vars['x_'+str(seed)]==1.0)
    m.addConstr(sum([degrees[node].item()*x_vars['x_'+str(node)] for node in range(len(degrees))])<=target+tolerance*target,'c_1')
    m.addConstr(sum([degrees[node].item()*x_vars['x_'+str(node)] for node in range(len(degrees))])>=target-target*tolerance,'c_2')
        
        
        
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    
    x_vals = [m.getVars()[i].x for i in range(data.x.shape[0])] 
    
    return m.objVal, x_vals

