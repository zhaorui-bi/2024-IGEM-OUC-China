
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Load the ESOL dataset
# data = MoleculeNet(root='.', name='ESOL')

# data.
# print(data)
# # Investigating the dataset
# print("Dataset type: ", type(data))
# print("Dataset features: ", data.num_features)
# print("Dataset target: ", data.num_classes)
# print("Dataset length: ", data.len)
# loaded_1 = torch.load('./esol/processed/data.pt')
# loaded_2 = torch.load('./esol/processed/pre_filter.pt')
# loaded_3 = torch.load('./esol/processed/pre_transform.pt')
# print(loaded_1)
# print(loaded_2)
# print(loaded_3)
# # Case Study
# print("Dataset sample: ", data[0])
# print("Sample  nodes: ", data[0].num_nodes)
# print("Sample  edges: ", data[0].num_edges)
# print(data[0].x)
# print(data[0].edge_index.T)
# print(data[0].y)
#
# print("Dataset sample smiles:", data[0]["smiles"])
# molecule = Chem.MolFromSmiles(data[0]["smiles"])
# Draw.MolToFile(molecule, "drug.png")

import csv
import re

# import OPIG
# import SMILES2graph as sg
from torch_geometric.datasets import MoleculeNet
data = MoleculeNet(root='.', name='ESOL')
print(len(data))
print(data[1].y)
print(data[1])
# print(data.num_classes)
# Read CSV Data_File
# csv_reader=csv.reader(open("./esol/raw/delaney-processed.csv"))

# def process(self):
#     with open(self.raw_paths[0], 'r') as f:  # 读取原始数据文件
#         dataset = f.read().split('\n')[1:-1]  # 按行分割，并去掉第一行
#         dataset = [x for x in dataset if len(x) > 0]  # 去掉空行
#
#     for line in dataset:  # 遍历每行
#         line = re.sub(r'\".*\"', '', line)  # 去掉".*"字符串
#         line = line.split(',')  # 逗号分隔
#
#         smiles = line[self.names[self.name][3]]  # 获取到smiles字符串
#         ys = line[self.names[self.name][4]]  # 获取到y值
#         ys = ys if isinstance(ys, list) else [ys]  # 将y值统一成数组形式
#
#         ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]  # 将y转成float类型
#         y = torch.tensor(ys, dtype=torch.float).view(1, -1)  # 将y转成torch.float类型
#
#         # 重点：获取x、edge_index、edge_attr数据，需要查看from_smiles函数
#         data = OPIG.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles, y)
#
#     return data