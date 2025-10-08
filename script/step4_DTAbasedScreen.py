import datetime
import os.path
import pickle
import random
import time
import esm
import argparse
import warnings
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from molbart.tokeniser import MolEncTokeniser, DEFAULT_PAD_TOKEN
from molbart.models.pre_train import BARTModel
from molbart.decoder import DecodeSampler
from transformers import AutoTokenizer, AutoModel

from utils.SciBert_load import prepare_text_tokens
from utils.utils import loss_fn, latent_transform, smiles_list_tokenizer, get_lr, vec2smi, seed_mol
import torch.nn.functional as F

def text_screen(model_list,text,SMILES_list):
    # 取出每个model
    SMILES_model, seq_model, s2t_model, t2s_model, text_model, pred_model = model_list

    # 数据集构建
    dataset = SMILES_dataset(SMILES_list)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)

    # 对text进行编码嵌入
    text_input = [text for i in range(len(SMILES_list))]
    text_tokens = prepare_text_tokens(description=text_input, tokenizer=text_tokenizer,
                                      max_seq_len=512)

    ids, masks = text_tokens[0].clone().detach().cuda(), text_tokens[1].clone().detach().cuda()
    with torch.no_grad():
        description_output = text_model(input_ids=ids, attention_mask=masks)
    description_repr = description_output["pooler_output"]

    description_fusion_vector = t2s_model(description_repr)

    # 新加的↓
    #description_fusion_vector = torch.mean(description_fusion_vector, dim=0)

    if args.norm:
        description_fusion_vector = F.normalize(description_fusion_vector, dim=-1)

    score_list = []
    for batch in dataloader:
        idx = batch[0].cuda()
        mask = batch[1].cuda()

        # SMIELS嵌入
        with torch.no_grad():
            encode_input = {"encoder_input": idx.T, "encoder_pad_mask": mask.T}
            molecule_embedding = SMILES_model.encode(encode_input)
        molecule_embedding = latent_transform(molecule_embedding, mask, "cls")
        if args.norm:
            molecule_embedding = F.normalize(molecule_embedding, dim=-1)

        # dta预测
        with torch.no_grad():
            result = pred_model(molecule_embedding,description_fusion_vector)
        for item in result:
            score_list.append(item.item())

    return score_list
class SMILES_dataset(Dataset):
    def __init__(self, smiles_list):
        super(SMILES_dataset, self).__init__()
        self.smiles_list = smiles_list
        # SMLIES_tokenizer加载
        SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"../llm_ckpt/chemformer-ckpt/bart_vocab.txt", REGEX,
                                                           DEFAULT_CHEM_TOKEN_START)

        # smiles转化为idx和mas
        self.SMILES_ids, self.SMILES_masks = smiles_list_tokenizer(smiles_list, SMILES_tokenizer, 512)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index):
        return (self.SMILES_ids[:,index],self.SMILES_masks[:,index])

    def collate_fn(self,batch):
        ids_list = [item[0] for item in batch]
        masks_list = [item[1] for item in batch]
        return (torch.stack(ids_list),torch.stack(masks_list))

class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(channel, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, smi_latent, text_latent):
        x = torch.cat((smi_latent, text_latent), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()

if __name__ == "__main__":
    #########################################参数设置##############################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--norm", type=bool, default=True)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--dataset_root", type=str, default='../dataset')
    parser.add_argument('--loss_type', type=str, default='EBM_NCE', choices=['EBM_NCE', 'InfoNCE', "cosine_sim_loss"])

    parser.add_argument("--description_type", type=str, default="pubmedbert-full", choices=["pubmedbert-full"])
    parser.add_argument("--SMILES_type", type=str, default="chemformer", choices=["chemformer"])
    parser.add_argument("--transform_type", type=str, default="linear", choices=["linear", "MLP", "transform"])
    parser.add_argument("--result_type", type=str, default="pooler", choices=["pooler", "cls"])

    parser.add_argument("--llm_freeze", type=bool, default=True, choices=[True, False])

    parser.add_argument("--latent_dim", type=int, default=768)
    args = parser.parse_args()
    print("arguments\t", args)
    warnings.filterwarnings("ignore")  # 老是弹出什么CSR警告，一点卵用都没有，直接屏蔽掉

    # 时间
    time_now = str(datetime.datetime.now())
    time_now = time_now.replace(" ", "_")
    time_now = time_now.replace(":", "-")

    # 设置cuda，设置随机种子
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    # 预训练SMILES模型加载
    REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    DEFAULT_CHEM_TOKEN_START = 272

    SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"../llm_ckpt/chemformer-ckpt/bart_vocab.txt", REGEX,
                                                       DEFAULT_CHEM_TOKEN_START)
    sampler = DecodeSampler(SMILES_tokenizer, 512)
    SMILES_model = BARTModel.load_from_checkpoint(checkpoint_path="../llm_ckpt/chemformer-ckpt/step=1000000.ckpt",
                                                  pad_token_idx=SMILES_tokenizer.vocab[SMILES_tokenizer.pad_token],
                                                  decode_sampler=sampler).cuda(device)
    molecule_dim = 512

    # 预训练蛋白模型加载
    seq_model, seq_tokenizer = esm.pretrained.load_model_and_alphabet_local(
        "../llm_ckpt/esm/esm1b/esm1b_t33_650M_UR50S.pt")
    seq_model = seq_model.cuda(device)
    seq_dim = 1280

    # 预训练大语言模型加载
    if args.description_type == "pubmedbert-abs":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract")
        text_model = AutoModel.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract").cuda(device)
        text_dim = 768
    elif args.description_type == "pubmedbert-full":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        text_model = AutoModel.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext").cuda(device)
        text_dim = 768

    else:
        raise ValueError("no such description_model")

    # 线性层
    s2t_model = torch.nn.Linear(seq_dim, args.latent_dim).to(device)
    # molecule_to_description
    t2s_model = torch.nn.Linear(text_dim, args.latent_dim).to(device)

    # 预测器
    pred_model = MLP(molecule_dim + args.latent_dim).to(device)
    pred_model.load_state_dict(torch.load("../model_save/pred.pth"))

    model_list = [SMILES_model, seq_model, s2t_model, t2s_model, text_model, pred_model]

    # 参数冻结
    for model in model_list:
        for param in model.parameters():
            param.requires_grad = False

    text = "Glucagon-like peptide-1 (GLP-1) is an incretin hormone primarily secreted by the intestines in response to nutrient intake, playing a key role in glucose metabolism by enhancing insulin secretion, inhibiting glucagon release, and slowing gastric emptying. GLP-1 binds to its receptor (GLP-1R) on pancreatic beta cells and other tissues, activating signaling pathways like cAMP/PKA, which improves blood glucose control and promotes satiety. Due to its beneficial effects on glucose regulation and weight reduction, GLP-1 is a valuable target in treating type 2 diabetes and obesity. GLP-1 receptor agonists, such as liraglutide and semaglutide, have shown significant efficacy in lowering blood sugar and aiding weight loss. Current research aims to refine GLP-1 therapies to enhance efficacy, duration, and reduce side effects, expanding its applications in metabolic diseases."
    #SMILES_list = ["CCC(C)(C(C(=O)O)O)O","C1CCC(=O)NCCCCCC(=O)NCC1","C1=CC(=C(C=C1Cl)Cl)Cl"]
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\guruiTarget\drugbank_filt.csv")
    SMILES_list = df['canonical_smiles'].tolist()
    sublists = [SMILES_list[i:i + args.batch_size] for i in range(0, len(SMILES_list), args.batch_size)]

    df_list = []
    for sub in tqdm(sublists):
        scores = text_screen(model_list,text,sub)
        df = pd.DataFrame({"scores":scores,"smiles":sub}, index=sub)
        df_list.append(df)

    # 将所有list中的df进行聚合
    combined_df = pd.concat(df_list, ignore_index=True)

    # 取前n个，值越小越好
    top = 1000
    sorted_df = combined_df.sort_values(by='scores', ascending=True).head(top)
    print(sorted_df)
    sorted_df.to_csv('D:\pycharm files\GRmol_local\screen_result.csv')


