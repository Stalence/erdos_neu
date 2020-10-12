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
from torch_geometric.utils import scatter_
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from my_utils.cut_utils import get_diracs
###########

class cut_MPNN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden1, hidden2, deltas, elasticity=0.01, num_iterations = 30):
        super(cut_MPNN, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.conv1 = GINConv(Sequential(
            Linear(1,  self.hidden1),
            ReLU(),
            Linear(self.hidden1, self.hidden1),
            ReLU(),
            BN( self.hidden1),
        ),train_eps=False)
        self.num_iterations = num_iterations
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.elasticity = elasticity
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(BN( self.hidden1))
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
            Linear( self.hidden1,  self.hidden1),
            ReLU(),
            Linear( self.hidden1,  self.hidden1),
            ReLU(),
            BN(self.hidden1),
        ),train_eps=False))
     
        self.conv2 = GATAConv( self.hidden1, self.hidden2 ,heads=8)
        self.lin1 = Linear(8*self.hidden2, self.hidden1)
        self.bn2 = BN(self.hidden1)
        self.lin2 = Linear(self.hidden1, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters() 
        for conv in self.convs:
            conv.reset_parameters()    
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.bn2.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data, tvol = None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch 
        xinit= x.clone()
        row, col = edge_index
        mask = get_mask(x,edge_index,1).to(x.dtype).unsqueeze(-1)

        x = self.conv1(x, edge_index)
        xpostconv1 = x.detach() 
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
        

        xprethresh = x.detach()
        N_size = x.shape[0]    
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)
        
        #min-max normalize       
        x = (x-batch_min)/(batch_max+1e-6-batch_min)
        x = x*mask + mask*1e-6
        

        #add dirac in the set
        x = x + xinit.unsqueeze(-1)
        
        #calculate
        x2 = x.detach()              
        r, c = edge_index
        tv = total_var(x, edge_index, batch)
        deg = degree(r).unsqueeze(-1) 
        conduct_1 = (tv)
        totalvol = scatter_add(deg.detach()*torch.ones_like(x, device=device), batch, 0)+1e-6
        totalcard = scatter_add(torch.ones_like(x, device=device), batch, 0)+1e-6
        
                
        #receptive field
        recvol_hard = scatter_add(deg*mask.float(), batch, 0, dim_size = batch.max().item()+1)+1e-6 
        reccard_hard = scatter_add(mask.float(), batch, 0, dim_size = batch.max().item()+1)+1e-6 
        
        assert recvol_hard.mean()/totalvol.mean() <=1, "Something went wrong! Receptive field is larger than total volume."
        target = torch.zeros_like(totalvol)
        
        #generate target vol
        if tvol is None:
            feasible_vols = data.recfield_vol/data.total_vol-0.0
            target = torch.rand_like(feasible_vols, device=device)*feasible_vols*0.85 + 0.1
            target = target.squeeze(-1)*totalvol.squeeze(-1)
        else:
            target = tvol*totalvol.squeeze(-1)
        a = torch.ones((batch.max().item()+1,1), device = device)
        xfilt = x
                
        
        ###############################################################################
        #iterative rescaling
        counter_no2 = 0
        for iteration in range(self.num_iterations):
            counter_no2 += 1
            keep = (((a[batch]*xfilt)<1).to(x.dtype))

            
            x_k, d_k, d_nk = xfilt*keep*mask, deg*keep*mask, deg*(1-keep)*mask
            
            
            diff = target.unsqueeze(-1) - scatter_add(d_nk, batch, 0)
            dot = scatter_add(x_k*d_k, batch, 0)
            a = diff/(dot+1e-5)
            volcur = (scatter_add(torch.clamp(a[batch]*xfilt,max = 1., min = 0.)*deg,batch,0))

            volcheck = (torch.abs(target - volcur.squeeze(-1))>0.1)
            checki = torch.abs(target.squeeze(-1)-volcur.squeeze(-1))>0.01

            targetcheck = torch.abs(volcur.squeeze(-1) - target)
            
            check = (targetcheck<= self.elasticity*target).to(x.dtype)

            if (tvol is not None):
                pass
            if(check.sum()>=batch.max().item()+1):
                break;
        
        probs = torch.clamp(a[batch]*x*mask, max = 1., min = 0.)
        ###############################################################################

            
            
        #collect useful numbers    
        x2 =  ((probs - torch.rand_like(x, device = device))>0).float()         
        vol_1 = scatter_add(probs*deg, batch, 0)+1e-6
        card_1 = scatter_add(probs, batch,0) 
        rec_field = scatter_add(mask, batch, 0)+1e-6
        cut_size = scatter_add(x2, batch, 0)
        tv_hard = total_var(x2, edge_index, batch)
        vol_hard = scatter_add(deg*x2, batch, 0, dim_size = batch.max().item()+1)+1e-6 
        conduct_hard = tv_hard/vol_hard         
        rec_field_ratio = cut_size/rec_field
        rec_field_volratio = vol_hard/recvol_hard
        total_vol_ratio = vol_hard/totalvol
        
        #calculate loss
        expected_cut = scatter_add(probs*deg, batch, 0) - scatter_add((probs[row]*probs[col]), batch[row], 0)   
        loss = expected_cut   


        #return dict 
        retdict = {}
        retdict["output"] = [probs.squeeze(-1),"hist"]   #output
        #retdict["|Expected_vol - Target|"]= [targetcheck, "sequence"] #absolute distance from targetvol
        retdict["Expected_volume"] = [vol_1.mean(),"sequence"] #volume
        retdict["Expected_cardinality"] = [card_1.mean(),"sequence"]
        retdict["volume_hard"] = [vol_hard.mean(),"sequence"] #volume2
        #retdict["cut1"] = [tv.mean(),"sequence"] #cut1
        retdict["cut_hard"] = [tv_hard.mean(),"sequence"] #cut1
        retdict["Average cardinality ratio of receptive field "] = [rec_field_ratio.mean(),"sequence"] 
        retdict["Recfield volume/Total volume"] = [recvol_hard.mean()/totalvol.mean(), "sequence"]
        retdict["Average ratio of receptive field volume"]= [rec_field_volratio.mean(),'sequence']
        retdict["Average ratio of total volume"]= [total_vol_ratio.mean(),'sequence']
        retdict["mask"] = [mask, "aux"] #mask
        retdict["xinit"] = [xinit,"hist"] #layer input diracs
        retdict["xpostlin1"] = [xpostlin1.mean(1),"hist"] #after first linear layer
        retdict["xprethresh"] = [xprethresh.mean(1),"hist"] #pre thresholding activations 195 x 1
        retdict["lossvol"] = [lossvol.mean(),"sequence"] #volume constraint
        retdict["losscard"] = [losscard.mean(),"sequence"] #cardinality constraint
        retdict["loss"] = [loss.mean().squeeze(),"sequence"] #final loss

        return retdict
    
    def __repr__(self):
        return self.__class__.__name__

    
    
    
    
class clique_MPNN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden1, hidden2, deltas, elasticity=0.01, num_iterations = 30):
        super(cliqueMPNN_hindsight_earlyGAT, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.momentum = 0.1
        self.num_iterations = num_iterations
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.elasticity = elasticity
        self.heads = 8
        self.concat = True
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(BN(self.heads*self.hidden1, momentum=self.momentum))
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            BN(self.heads*self.hidden1, momentum=self.momentum),
        ),train_eps=True))
        self.bn1 = BN(self.heads*self.hidden1)       
        self.conv1 = GINConv(Sequential(Linear(self.hidden2,  self.heads*self.hidden1),
            ReLU(),
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            BN(self.heads*self.hidden1, momentum=self.momentum),
        ),train_eps=True)

        if self.concat:
            self.lin1 = Linear(self.heads*self.hidden1, self.hidden1)
        else:
            self.lin1 = Linear(self.hidden1, self.hidden1)
        self.lin2 = Linear(self.hidden1, 1)
        self.gnorm = GraphSizeNorm()

                    


    def reset_parameters(self):
        self.conv1.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters() 
        for bn in self.bns:
            bn.reset_parameters()
        self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()






    def forward(self, data, edge_dropout = None, penalty_coefficient = 0.25):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        num_graphs = batch.max().item() + 1
        row, col = edge_index     
        total_num_edges = edge_index.shape[1]
        N_size = x.shape[0]

        
        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr = (torch.ones(edge_index.shape[1], device=device)).long(), p = edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]
                
        reduced_num_edges = edge_index.shape[1]
        current_edge_percentage = (reduced_num_edges/total_num_edges)
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index

        xinit= x.clone()
        x = x.unsqueeze(-1)
        mask = get_mask(x,edge_index,1).to(x.dtype)
        x = F.leaky_relu(self.conv1(x, edge_index))# +x
        x = x*mask
        x = self.gnorm(x)
        x = self.bn1(x)
        
            
        for conv, bn in zip(self.convs, self.bns):
            if(x.dim()>1):
                x =  x+F.leaky_relu(conv(x, edge_index))
                mask = get_mask(mask,edge_index,1).to(x.dtype)
                x = x*mask
                x = self.gnorm(x)
                x = bn(x)

        xpostconvs = x.detach()
        #
        x = F.leaky_relu(self.lin1(x)) 
        x = x*mask


        xpostlin1 = x.detach()
        x = F.leaky_relu(self.lin2(x)) 
        x = x*mask


        #calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)

        #min-max normalize
        x = (x-batch_min)/(batch_max+1e-6-batch_min)
        probs=x
           
        x2 = x.detach()              
        deg = degree(row).unsqueeze(-1) 
        totalvol = scatter_add(deg.detach()*torch.ones_like(x, device=device), batch, 0)+1e-6
        totalcard = scatter_add(torch.ones_like(x, device=device), batch, 0)+1e-6               
        x2 =  ((x2 - torch.rand_like(x, device = device))>0).float()    
        vol_1 = scatter_add(probs*deg, batch, 0)+1e-6
        card_1 = scatter_add(probs, batch,0)            
        set_size = scatter_add(x2, batch, 0)
        vol_hard = scatter_add(deg*x2, batch, 0, dim_size = batch.max().item()+1)+1e-6 
        total_vol_ratio = vol_hard/totalvol
        
        
        #calculating the terms for the expected distance between clique and graph
        pairwise_prodsums = torch.zeros(num_graphs, device = device)
        for graph in range(num_graphs):
            batch_graph = (batch==graph)
            pairwise_prodsums[graph] = (torch.conv1d(probs[batch_graph].unsqueeze(-1), probs[batch_graph].unsqueeze(-1))).sum()/2
        
        
        ###calculate loss terms
        self_sums = scatter_add((probs*probs), batch, 0, dim_size = num_graphs)
        expected_weight_G = scatter_add(probs[no_loop_row]*probs[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2.
        expected_clique_weight = (pairwise_prodsums.unsqueeze(-1) - self_sums)/1.
        expected_distance = (expected_clique_weight - expected_weight_G)        
        
        
        ###useful numbers 
        max_set_weight = (scatter_add(torch.ones_like(x)[no_loop_row], batch[no_loop_row], 0, dim_size = num_graphs)/2).squeeze(-1)                
        set_weight = (scatter_add(x2[no_loop_row]*x2[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs)/2)+1e-6
        clique_edges_hard = (set_size*(set_size-1)/2) +1e-6
        clique_dist_hard = set_weight/clique_edges_hard
        clique_check = ((clique_edges_hard != clique_edges_hard))
        setedge_check  = ((set_weight != set_weight))      
        
        assert ((clique_dist_hard>=1.1).sum())<=1e-6, "Invalid set vol/clique vol ratio."

        ###calculate loss
        expected_loss = (penalty_coefficient)*expected_distance*0.5 - 0.5*expected_weight_G  
        

        loss = expected_loss


        retdict = {}
        
        retdict["output"] = [probs.squeeze(-1),"hist"]   #output
        retdict["Expected_cardinality"] = [card_1.mean(),"sequence"]
        retdict["Expected_cardinality_hist"] = [card_1,"hist"]
        retdict["losses histogram"] = [loss.squeeze(-1),"hist"]
        retdict["Set sizes"] = [set_size.squeeze(-1),"hist"]
        retdict["volume_hard"] = [vol_hard.mean(),"aux"] #volume2
        retdict["cardinality_hard"] = [set_size[0],"sequence"] #volumeq
        retdict["Expected weight(G)"]= [expected_weight_G.mean(), "sequence"]
        retdict["Expected maximum weight"] = [expected_clique_weight.mean(),"sequence"]
        retdict["Expected distance"]= [expected_distance.mean(), "sequence"]
        retdict["Currvol/Cliquevol"] = [clique_dist_hard.mean(),'sequence']
        retdict["Currvol/Cliquevol all graphs in batch"] = [clique_dist_hard.squeeze(-1),'hist']
        retdict["Average ratio of total volume"]= [total_vol_ratio.mean(),'sequence']
        retdict["cardinalities"] = [cardinalities.squeeze(-1),"hist"]
        retdict["Current edge percentage"] = [torch.tensor(current_edge_percentage),'sequence']
        retdict["loss"] = [loss.mean().squeeze(),"sequence"] #final loss

        return retdict
    
    def __repr__(self):
        return self.__class__.__name__
