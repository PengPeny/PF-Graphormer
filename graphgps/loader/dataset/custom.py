import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):

        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # self.raw_dir
        # self.processed_dir
        #其中processed_paths来自于Dataset类,返回数据

    @property
    def raw_file_names(self):
        return ['case118_0.csv', 'case118_y.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']  

    def process(self):
        
        nodes = pd.read_csv(os.path.join(self.raw_dir,'case118_0.csv'))
        x = torch.tensor(nodes.iloc[:].values, dtype=torch.float)
      
        edges = pd.read_csv(os.path.join(self.raw_dir,'case118_y.csv'))  
        assert not edges.isnull().values.any(), "边数据存在空值"
        edge_index = torch.tensor(edges.iloc[:, [0,1]].values.T, dtype=torch.long)
        edge_attr = torch.tensor(edges.iloc[:, [2,3]].values, dtype=torch.float)
        assert edge_index.min() >= 0, f"无效节点编号：{edge_index.min().item()}"
        

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset = CustomGraphDataset(root='datasets/custom')
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[0])

    import shutil
    shutil.rmtree('./datasets/custom/processed')
