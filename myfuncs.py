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
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch 

def getNdiracs(data, N , sparse = False, flat = False, replace = True):
    
    if not sparse:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
            original_batch_index = data.batch
            original_edge_index = data.edge_index
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            graphcount =data.num_graphs #number of graphs in data/batch object
            diracmatrix = torch.zeros(0,device=device)
            batch_prime = torch.zeros(0,device=device).long()
            locationmatrix = torch.zeros(0,device=device).long()
            
            global_offset = 0
            for k in range(graphcount):
                graph_nodes = (data.batch == k).sum()
                #probabilities = torch.ones((graph_nodes.item(),1),device=device)/graph_nodes #uniform probs
                #node_distribution = OneHotCategorical(probs=probabilities.squeeze())
                #node_sample = node_distribution.sample(sample_shape=(N,))
                
                
#                 if flat:
#                     diracmatrix = torch.cat((diracmatrix, node_sample.view(-1)),0)
#                 else:
#                     diracmatrix = torch.cat((diracmatrix, node_sample.t(),0))
                
                #for diracmatrix
                randInt = np.random.choice(range(graph_nodes), N, replace = replace)
                node_sample = torch.zeros(N*graph_nodes,device=device)
                offs  = torch.arange(N, device=device)*graph_nodes
                dirac_locations = (offs + torch.from_numpy(randInt).to(device))
                node_sample[dirac_locations] = 1
                
                dirac_locations2 = torch.from_numpy(randInt).to(device) + global_offset
                global_offset += graph_nodes
                
                diracmatrix = torch.cat((diracmatrix, node_sample),0)
                locationmatrix = torch.cat((locationmatrix, dirac_locations2),0)
            
                
            
                #for batch prime
                dirac_indices = torch.arange(N, device=device).unsqueeze(-1).expand(-1, graph_nodes).contiguous().view(-1)
                dirac_indices = dirac_indices.long()
                dirac_indices += k*N
                batch_prime = torch.cat((batch_prime, dirac_indices))



            
#             locationmatrix = diracmatrix.nonzero()
            edge_index_prime = torch.arange(N).unsqueeze(-1).expand(-1,data.edge_index.shape[1]).contiguous().view(-1)*data.batch.shape[0]
            offset = torch.arange(N).unsqueeze(-1).expand(-1,data.edge_index.size()[1]).contiguous().view(-1)*data.batch.shape[0]
            offset_2 = torch.cat(2*[offset.unsqueeze(0)],dim = 0)
            edge_index_prime = torch.cat(N*[data.edge_index], dim = 1) + offset_2
            normalization_indices = data.batch.unsqueeze(-1).expand(-1,N).contiguous().view(-1).to(device)

            return Batch(batch = batch_prime, x = diracmatrix, edge_index = edge_index_prime,
                         y = data.y, locations = locationmatrix, norm_index = normalization_indices, batch_old = original_batch_index, edge_index_old = original_edge_index)

        


def trainModel(loader,optimizer,device):

    total_loss=0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out1, out2 = net(data)
        loss = F.nll_loss(out1, one_hot(data.y).long())
        loss.backward()
        total_loss += loss.item() * data.num_nodes
        optimizer.step()
    return total_loss

def getSparseData(x, adj, mask):

    myedge_index = sp.dense_to_sparse(adj)[0]
    myx          = x[mask]
    mydata       = Data(x=myx,edge_index = myedge_index) 
    return mydata

def padToData(x,data,adj= None):

    if x.shape[0]==data.x.shape[0]:
        return x
    totalsize= data.x.shape[0]
    actualsize= data.num_nodes
    newx= torch.zeros((totalsize))
    newx[:actualsize]=x
    if adj!= None:
        newadj= torch.zeros((totalsize,totalsize))
        newadj[:actualsize,:actualsize]=adj
        return newx,newadj
    else:
        return newx
    
def drawGraphFromBatch(mybatch, index):    
   
    G=cnv.to_networkx(getSparseData(mybatch.x[index],mybatch.adj[index],mybatch.mask[index]))
    G=G.to_undirected()
    pos= graphviz_layout(G)
    nx.draw(G,pos,alpha=0.75)
    return G, pos

def drawGraphFromData(myx,myadj,mask,seed=None,nodecolor=False,edgecolor=False,seedhops=False,hoplabels=False,binarycut=False):   
    
    if myx.unsqueeze(-1).shape[1]>1:
        myx=myx[:,0]
    
    #pad x values to fit standardized form
    newx = padToData(myx,Data(x=myx,adj = myadj))
    
    #convert to nx graph
    G=cnv.to_networkx(getSparseData(myx,myadj,mask))
    G=G.to_undirected()
    pos= graphviz_layout(G)
    nofnodes= G.number_of_nodes()
    
    if nodecolor:
        #initialize color matrices for the plots
        nodecolors=torch.zeros(nofnodes,3) 
        colv=myx[:nofnodes]
        colv=torch.log(colv+1e-6)
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
            
            for key in shortestpaths:
                if key != theseed:
                    scale = 1- shortestpaths[key]/maxpath
                    nodecolors[key,2] = scale*0.7 + 0.3
                    nodecolors[key,1] = scale*0.7 + 0.3
                    withoutseed[key] = key
                    #print(position)
                else:
                    scale = 1- shortestpaths[key]/maxpath
                    position = {key: pos[key]}
                    
                    nx.draw_networkx_nodes(G,position,alpha=1.0,nodelist=[key],node_color='r',node_size=1200*scale)
                    
            nx.draw_networkx_nodes(G,positions,alpha=0.65,nodelist=list(positions.keys()),node_color=list(orderpathswoseed.values()),vmin=0,vmax=maxpath,cmap=plt.cm.hsv,node_size=450)
            if hoplabels:
                nx.draw_networkx_labels(G,pos,labels=orderpaths,font_color='k',alpha=0.75)
                
    else:
        nodecolors = 'r'
    
    
    if seedhops == False:
        nx.draw_networkx_nodes(G,pos,alpha=0.75,node_color=nodecolors,node_size=200)

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

class GraphConvolution(Module):
    #kipf's model

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
def sweep(myp,G,data):

    supp = torch.nonzero(myp).squeeze().tolist()
    degs = data.adj[supp,:].sum(-1)
    sortedsupp = torch.argsort(myp[supp]/degs,descending=True).squeeze().tolist()
    support = [supp[i] for i in sortedsupp]
    sweepset = []
    bestconductance = 1000
    bestvolume = 0
    bestset = []
    for i in support:
        sweepset += [i]
        volume = nx.volume(G,sweepset)
        conductance = nx.conductance(G,sweepset)
        if(conductance < bestconductance):
            bestconductance = conductance
            bestvolume = volume
            bestset = sweepset
    return bestset, bestconductance, bestvolume

def pushv(p,r,index,adj,a):

    rprime=r
    pprime=p
    pprime[index] = p[index] + a*r[index]
    rprime[index] = (1-a)*r[index]/2
    for i in torch.nonzero(adj[index,:]):
        rprime[i.item()]= r[i.item()] + (1-a)*r[index]/(2*adj[index,:].sum(-1))
    return pprime, rprime

def approxPRank(index,alpha,epsilon,data):

    adj= data.adj[0:data.num_nodes,0:data.num_nodes]
    p = torch.zeros((data.num_nodes)) 
    r = torch.zeros((data.num_nodes))
    r[index]=1 
    maxval,maxplace=  (r/adj.sum(dim=-1)).max(-1)[0], (r/adj.sum(dim=-2)).max(-1)[1]
    while maxval>= epsilon:
        p,r = pushv(p,r,maxplace,adj,alpha)
        maxval,maxplace=  (r/adj.sum(dim=-1)).max(-1)[0], (r/adj.sum(dim=-2)).max(-1)[1]
    return p,r        

def pagerank_nibble(index,phi,vol,data,G):
 
    logvol = torch.ceil(torch.log2(data.adj.sum()/2.))
    vol = torch.tensor(vol).float()
    print(torch.log2(vol))
    vol = 1. + torch.log2(vol)
    print(vol,logvol)
    vol = torch.min(vol,logvol)
    alpha =  (phi*phi)/(225*torch.log((vol))*torch.sqrt((data.adj.sum()/2.)))
    eps=(1/(torch.pow(torch.tensor(2.),vol)))*(1/(48*logvol))
    myp, r= approxPRank(index,alpha,eps,data)
    bset,bcond,bvol = sweep(myp,G,data)
    return bset,bcond,bvol,myp

def lovaszSimonovits(myp,G,data):
    supp=torch.nonzero(myp).squeeze().tolist()
    degs=data.adj[supp,:].sum(-1)
    sortedsupp=torch.argsort(myp[supp]/degs,descending=True).squeeze().tolist()
    support = [supp[i] for i in sortedsupp]
    vols = []
    sweepset=[]
    probmass=[]
    for i in support:
        sweepset+= [i]
        vols +=[nx.volume(G,sweepset)]
        probmass+=[myp[sweepset].sum()]
    plt.plot(vols,probmass)    
    return probmass,vols    

def clusterBench(method,graphno,dataset):
    
    numnodes= dataset[graphno].num_nodes
    seedset= torch.randint(numnodes,(3,1)).squeeze()
    efo=plt.figure(2,figsize=(8,8))
    G,pos=drawGraphFromData(dataset[graphno].x[:,0],dataset[graphno].adj,dataset[graphno].mask,nodecolor=False)
    f=plt.figure(1,figsize=(20,20))
    for i in range(9):
        no = 330+i+1
        g= f.add_subplot(no)
        node = seedset[torch.tensor(i%3).long()].item()
        if(method==1):
            phi= np.round(np.power(0.1,np.floor(i/3)+1),5)
            bs,bc,bv,myp = pagerank_nibble(node,phi,60+i,dataset[graphno],G)
            g=g.text(0.5,-0.1, str(i+1)+") " + "Node: " + str(node) + " Phi= " + str(phi), size=12, ha="center", 
             transform=g.transAxes)
        valuevec= torch.zeros(195)
        valuevec[bs]=1
        p,g=drawGraphFromData(valuevec,dataset[graphno].adj,dataset[graphno].mask,nodecolor=True)
        
def AdjToLocal(adj,mask):
    truesize = mask.sum().item()
    newadj=adj[0,:truesize,:truesize]
    mygraph = nx.from_numpy_matrix(newadj.numpy())
    nx.write_edgelist(mygraph,'temp.edgelist',data=[])
    g = GraphLocal('temp.edgelist','edgelist',' ') 
    return g
               
def AdjToLocal2(adj):
    mygraph = nx.from_numpy_matrix(data.adj[0].numpy())
    nx.write_edgelist(mygraph,'temp.edgelist',data=[])
    g = GraphLocal('temp.edgelist','edgelist',' ') 
    return g

def TestLocalClust(methodclass,smethod,test_loader,iterations=10000,delta=1e-04,param1=0.15,param2=1e-6,param3=100,param4=0.5,draw=False):
    mean_conductance= 0 
    counter = 0
    for data in test_loader:
        AdjToLocal(data.adj)
        counter += 1
        seed=data.x.nonzero()[0,0].item()
        if methodclass == "spectral":
            epsilon=1e-2 
            cutset, conductance = spectral_clustering(g,[seed],method=smethod,alpha=param1,rho=param2,vol=param3,phi=param4)
        if methodclass == "flow":
            cutset, conductance = flow_clustering(g,[seed],method,U=param1,h=param2,w=param3)
        mean_conductance+=conductance         
    mean_conductance = mean_conductance/counter
    return mean_conductance
             
def barabasi_albert_graph(num_nodes, num_edges):
    assert num_edges > 0 and num_edges < num_nodes
    row, col = torch.arange(num_edges), torch.randperm(num_edges)
    for i in range(num_edges, num_nodes):
        row = torch.cat([row, torch.full((num_edges, ), i, dtype=torch.long)])
        choice = np.random.choice(torch.cat([row, col]).numpy(), num_edges)
        col = torch.cat([col, torch.from_numpy(choice)])
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes)
    return edge_index    

def remove_Connected(dataset):
    custom_dataset = []
    for data in dataset:
        g = cnv.to_networkx(data).to_undirected()
        no_of_components = nx.connected_components(g)
        maxset = []
        maxsize = 0
        count = 0
        for comp in no_of_components:
            count += 1
            comp_size = len(comp)
            if(comp_size > maxsize):
                maxsize = comp_size
                maxset = comp 
        maxset = list(maxset)
        adj = nx.adjacency_matrix(g).todense()
        adj = adj[maxset,:]  
        adj = adj[:,maxset]
        g2 = nx.from_numpy_matrix(adj).to_undirected()
        custom_dataset +=  [cnv.from_networkx(g2)]     
    return custom_dataset
      
def create_BAdataset(noderange, edge_param, start = 50, graphs_per_nodecount = 10, dense = False):   
    dataset = []
    final_size = start + noderange
    for i in range(noderange):
        for k in range(graphs_per_nodecount):
            eind = myfuncs.barabasi_albert_graph(start+i,edge_param)
            data = Data(x = torch.ones(start+i), edge_index = eind )
            dataset += [data]
    if dense:
        for k in range(len(dataset)):
            dataset[k] = Data(adj = to_dense_adj.to_dense_adj(dataset[k].edge_index).squeeze(0), x = dataset[k].x)         
    return dataset

def data_to_3D(x, edge_index, dimsize):    
    return x.contiguous().view(dimsize,-1,3), edge_index.contiguous().view(2,dimsize,-1)

def data_to_2D(x, edge_index, featdim ):
    return x.view(-1,x.shape[featdim]), edge_index.view(2,-1)
