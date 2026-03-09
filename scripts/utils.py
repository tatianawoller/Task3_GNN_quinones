# Recurring function in the jupyter notebook
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit.Chem import AllChem
import torch.nn as nn
import os
from sklearn.metrics import mean_absolute_error, r2_score,explained_variance_score,max_error

def identify_quinone_derivatives(smiles, patterns: dict):
    ''' identifies whether a smile corresponds to a quinone based on dictionary of patterns'''  
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    results = {}
    for derivative, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        results[derivative] = mol.HasSubstructMatch(pattern)

    return results

def mol_to_nx(molecule_smiles):
    ''' converts a smiles to a graph'''
    molecule = Chem.MolFromSmiles(molecule_smiles)
    molecule = Chem.AddHs(molecule)
    G = nx.Graph()

    # Add nodes (atoms)
    for atom in molecule.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())

    # Add edges (bonds)
    for bond in molecule.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return G

def plot_molecule_rdkit(mol_smiles):
    ''' plot a molecule using rdkit and matplotlib'''
    molecule=Chem.MolFromSmiles(mol_smiles)
    molecule = Chem.AddHs(molecule)
    if molecule is None:
        print("Invalid SMILES string")
    else:
        img = Draw.MolToImage(molecule, size=(300, 300))
    return img


def plot_molecule_networkx(mol_smiles):
    ''' plot a molecule using networkx and matplotlib'''
    G = mol_to_nx(mol_smiles)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(G)  # Position the nodes for visualization
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.show()

def process_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    num_atoms = molecule.GetNumAtoms()
    adj_matrix = rdmolops.GetAdjacencyMatrix(molecule)
    atom_types = []
    atom_symbols = []
    atomic_num_symbol_map = {}
    
    for atom in molecule.GetAtoms():
        atom_num = atom.GetAtomicNum()
        atom_symbol = atom.GetSymbol()
        atom_types.append(atom_num)  
        atom_symbols.append(atom_symbol)  
        
        if atom_num not in atomic_num_symbol_map:
            atomic_num_symbol_map[atom_num] = atom_symbol

    edge_list = []
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        edge_list.append((atom1, atom2))
        edge_list.append((atom2, atom1))

    return atom_types, adj_matrix, edge_list, atomic_num_symbol_map

def dataframe_input_gnn(input_df):
    stoichio_list = []
    smiles_list = []
    atom_types_list = []
    adj_matrix_list = []
    edge_list_list = []
    atomic_num_symbol_map_list = []
    for idx, row in input_df.iterrows():
        smiles = row['smiles']
        stoichiometry = row['stoichiometry']
        atom_types, adj_matrix, edge_list, atomic_num_symbol_map = process_molecule(smiles)
        stoichio_list.append(stoichiometry)
        smiles_list.append(smiles)
        atom_types_list.append(atom_types)
        adj_matrix_list.append(adj_matrix)
        edge_list_list.append(edge_list)
        atomic_num_symbol_map_list.append(atomic_num_symbol_map)
    df_quinone_processed = pd.DataFrame({
        'stochiometry': stoichio_list,
        'smiles': smiles_list,
        'atom_types': atom_types_list,
        'adj_matrix': adj_matrix_list,
        'edge_list': edge_list_list,
        'atomic_num_symbol_map': atomic_num_symbol_map_list})
    df_quinone_processed.set_index('stochiometry', inplace=True)
    return df_quinone_processed


def df_row_to_data(row, property):
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None:
        raise ValueError(f"Invalid SMILES: {row['smiles']}")

    # Build all graph tensors from the same molecule to guarantee consistent indexing.
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    pos = torch.tensor(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=torch.float,
    )

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([row[property]], dtype=torch.float)

    num_nodes = z.size(0)
    if pos.size(0) != num_nodes:
        raise ValueError(f"pos and z mismatch: pos={pos.size(0)} z={num_nodes}")
    if edge_index.numel() > 0 and int(edge_index.max()) >= num_nodes:
        raise ValueError(
            f"edge_index out of range: max={int(edge_index.max())}, num_nodes={num_nodes}"
        )

    return Data(z=z, pos=pos, edge_index=edge_index, y=y)


class DenseRadiusGraph(nn.Module):
    def __init__(self, cutoff=10.0, max_num_neighbors=32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos, batch):
        if batch is None:
            batch = pos.new_zeros(pos.size(0), dtype=torch.long)

        all_src = []
        all_dst = []

        for b in batch.unique(sorted=True):
            node_idx = (batch == b).nonzero(as_tuple=False).view(-1)
            pos_b = pos[node_idx]
            n = pos_b.size(0)
            if n <= 1:
                continue

            dist = torch.cdist(pos_b, pos_b, p=2)
            mask = (dist <= self.cutoff) & (~torch.eye(n, device=pos.device, dtype=torch.bool))

            if self.max_num_neighbors is not None:
                for i in range(n):
                    nbrs = mask[i].nonzero(as_tuple=False).view(-1)
                    if nbrs.numel() > self.max_num_neighbors:
                        d = dist[i, nbrs]
                        keep = nbrs[torch.topk(d, self.max_num_neighbors, largest=False).indices]
                        new_row = torch.zeros_like(mask[i])
                        new_row[keep] = True
                        mask[i] = new_row

            src_local, dst_local = mask.nonzero(as_tuple=True)
            all_src.append(node_idx[src_local])
            all_dst.append(node_idx[dst_local])

        if all_src:
            row = torch.cat(all_src)
            col = torch.cat(all_dst)
            edge_index = torch.stack([row, col], dim=0)
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            edge_weight = torch.empty((0,), dtype=pos.dtype, device=pos.device)

        return edge_index, edge_weight



def prediction_model(device,data_list, model):
    predictions=[]
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            pred = model(data.z, data.pos).view(-1)[0]
            predictions.append(pred.item())
    return predictions

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch)
        loss = criterion(pred.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_fresh(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch)
            loss = criterion(pred.view(-1), batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def plot_regression(target_val, predicted_val,title:str,plot_name:str,xlabel:str, ylabel:str):
    ''' plot correlation and some metrics for regression task'''
    plt.figure(figsize=(8, 8))
    plt.scatter(target_val, predicted_val, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    min_val = min(min(target_val), min(predicted_val))
    max_val = max(max(target_val), max(predicted_val))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y=x")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../validation/{plot_name}.pdf", format="pdf")  

def compute_metrics(target_val:list,predicted_val:list):
    ''' compute metrics for regression task'''
    mae=mean_absolute_error(target_val,predicted_val)
    rsq=r2_score(target_val,predicted_val)
    maxerror=explained_variance_score(target_val,predicted_val)
    variance_score=max_error(target_val,predicted_val)
    return mae, rsq, maxerror, variance_score

