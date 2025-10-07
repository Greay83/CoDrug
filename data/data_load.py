from torch.utils.data import Dataset
import pandas as pd
import json
import torch
from tqdm import tqdm
class text2SMILES_Datasets(Dataset):
    def __init__(self, root):
        self.root = root
        CID2text_dir = root + "/CID2text.json"
        CID2SMILES_dir = root + "/CID2SMILES.csv"
        print(CID2SMILES_dir)
        self.file_load(CID2text_dir, CID2SMILES_dir)

    def file_load(self, CID2text, CID2SMILES):
        df = pd.read_csv(CID2SMILES)
        CID_list, SMILES_list = df["CID"].tolist(), df["SMILES"].tolist()
        config_dir = "../molformer/hparams.yaml"
        tok_dir = "../molformer/bert_vocab.txt"
        ckpt_dir = "../molformer/N-Step-Checkpoint_3_30000.ckpt"
        encode_vocab_dir = "../molformer/pubchem_canon_zinc_final_vocab_sorted.pth"
        SMILES_model = MF_load(config_dir, tok_dir, ckpt_dir, encode_vocab_dir)
        embedding = MF_emb(SMILES_model, df.SMILES, batch_size=256)
        CID2SMILES = {}
        index = 0
        for CID in tqdm(CID_list):
            CID = str(CID)
            CID2SMILES[CID] = embedding[index]
            index += 1
        print("len of CID2SMILES: {}".format(len(CID2SMILES.keys())))
        torch.save(CID2SMILES, "../dataset/CID2SMILES_dict.pt")

        with open(CID2text, "r") as f:
            self.CID2text_data = json.load(f)
        print("len of CID2text: {}".format(len(self.CID2text_data.keys())))

        self.text_list = []
        missing_count = 0
        for CID, value_list in self.CID2text_data.items():
            if CID not in CID2SMILES:
                print("CID {} missing".format(CID))
                missing_count += 1
                continue
            for value in value_list:
                self.text_list.append([CID, value])
        print("missing", missing_count)
        print("len of text_list: {}".format(len(self.text_list)))


    def __getitem__(self, index):
        loaded_tensor_dict = torch.load("../dataset/CID2SMILES_dict.pt")
        CID, text = self.text_list[index]
        SMILES = loaded_tensor_dict[CID]
        return text, SMILES

    def __len__(self):
        return len(self.text_list)