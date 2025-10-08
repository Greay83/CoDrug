import argparse
from functools import partial

import numpy as np
import torch
from molbart.decoder import DecodeSampler
from molbart.models.pre_train import BARTModel
from molbart.tokeniser import MolEncTokeniser
from rdkit import Chem
from torch import optim, nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from utils.utils import latent_transform, smiles_list_tokenizer,REGEX,DEFAULT_CHEM_TOKEN_START,get_lr,vec2smi


def load_language_molecule_and_edit_models(device):
    # text_model
    text_model = BertModel.from_pretrained('../llm_ckpt/scibert/allenai/scibert_scivocab_uncased').cuda(device)
    text_dim = 768

    # text_tokenizer
    text_tokenizer = BertTokenizer.from_pretrained('../llm_ckpt/scibert/allenai/scibert_scivocab_uncased/vocab.txt')


    # molecule_model
    REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    DEFAULT_CHEM_TOKEN_START = 272
    SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"..\llm_ckpt\chemformer-ckpt\bart_vocab.txt", REGEX,
                                                       DEFAULT_CHEM_TOKEN_START)
    sampler = DecodeSampler(SMILES_tokenizer, 512)
    molecule_model = BARTModel.load_from_checkpoint(checkpoint_path="../llm_ckpt/chemformer-ckpt/step=1000000.ckpt",
                                                  pad_token_idx=SMILES_tokenizer.vocab[SMILES_tokenizer.pad_token],
                                                  decode_sampler=sampler).cuda(device)

    # molecule_dim
    molecule_dim = 512

    # text2latent
    text2latent = torch.nn.Linear(text_dim, 256).to(device)
    text2latent.load_state_dict(torch.load("../model_save/step1_2024-04-25_11-04-53.491871/d2m.pt"))

    # mol2latent
    mol2latent = torch.nn.Linear(molecule_dim, 256).to(device)
    mol2latent.load_state_dict(torch.load("../model_save/step1_2024-04-25_11-04-53.491871/m2d.pt"))


    return text_model, text_tokenizer, text_dim, molecule_model, molecule_dim, \
        text2latent, mol2latent

def edit(SMILES,text):
    # 先把text存入list？
    text_list = [text]

    # 把text进行token,然后embedding为特征向量，输出pooler_output
    description_tokenizer = BertTokenizer.from_pretrained('../llm_ckpt/scibert/allenai/scibert_scivocab_uncased/vocab.txt')

    from utils.SciBert_load import prepare_text_tokens
    text_tokens = prepare_text_tokens(description=text_list, tokenizer=description_tokenizer,
                                      max_seq_len=512)
    text_output = text_model(input_ids=text_tokens[0].clone().detach().cuda(), attention_mask=text_tokens[1].clone().detach().cuda())
    text_repr = text_output["pooler_output"]

    # 再让text通过t2l，得到latent表征
    text_repr = text2latent(text_repr)

    # 生成一个空列表：第一和第二SMILES_list
    first_and_second_SMILES_list = []

    SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"..\llm_ckpt\chemformer-ckpt\bart_vocab.txt", REGEX,
                                                       DEFAULT_CHEM_TOKEN_START)
    # 使用MegaMolBART对smiles_list进行embedding(不通过线性层)
    latent_code_init, pad_mask_init = smiles_list_tokenizer([SMILES], SMILES_tokenizer, 512)
    with torch.no_grad():
        encode_input = {"encoder_input": latent_code_init, "encoder_pad_mask": pad_mask_init}
        latent_code_init = molecule_model.encode(encode_input)
    # 使用MegaMolBART解码重生成mol#################
    regenerated_mols = vec2smi(molecule_model,latent_code_init,pad_mask_init)
    ##############这里可以再写一个循环，下面也是：先找出能被rdkit读取的分子，其次（可选）找出最相似的分子？######################
    #########################################

    # 把SMILES又添加进这个列表里：所以第一个SMILES是原始SMILES，第二个是重建的SMILES
    first_and_second_SMILES_list.append(SMILES)
    first_and_second_SMILES_list.append(regenerated_mols)

    # 设置不同的λ步长
    #l2_lambda_list = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    l2_lambda_list = [1e1, 5e0, 1e0, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4]

    # 这里生成的俩空列表我不知道啥意思
    result_SMILES_list = []

    if args.use_noise_for_init:
        print("Use random noise for init")
        # 生成一个随机噪声，维度尺寸是跟lantent一样
        random_noise = torch.randn(latent_code_init.size()).to(device)
        # 使用五个不同步长进行遍历
    result = []
    for l2_lambda in l2_lambda_list:
        print("l2 lambda: {}".format(l2_lambda))
        # 重新生成一个跟刚刚类似的list
        current_SMILES_list = [first_and_second_SMILES_list[0]] + [first_and_second_SMILES_list[1]]

        # 如果使用噪声，那就把latent加上噪声；否则不加
        if args.use_noise_for_init:
            print("Use random noise for init")
            latent = latent_code_init.detach().clone() + random_noise
        else:
            print("No random noise for init")
            latent = latent_code_init.detach().clone()

        # 复制pad_mask？
        pad_mask = pad_mask_init.detach().clone()
        latent.requires_grad = True

        # 优化器设置
        optimizer = optim.Adam([latent], lr=args.lr)



    # 按照epochs进行迭代
        # verbose进度条设置
        if args.verbose:
            L = tqdm(range(args.epochs))
        else:
            L = range(args.epochs)
        for i in L:
            # epoch的倒数，也就是进度？
            t = i / args.epochs

            # 用上述得到的t去获取学习率
            lr = get_lr(t, l2_lambda)

            # 把学习率放到优化器的参数里？
            optimizer.param_groups[0]["lr"] = lr

            # “第一线性层”输入latent和mask，得到分子的表征
            molecule_repr = latent_transform(latent, pad_mask, "pooler")  # [B, d]
            molecule_repr = mol2latent(molecule_repr)
            # 这里没看懂是什么loss
            molecule_repr = F.normalize(molecule_repr, dim=-1)
            text_repr = F.normalize(text_repr, dim=-1)
            #l1_similarity_loss = -torch.mm(molecule_repr, text_repr.transpose(0, 1))[0]
            l1_similarity_loss = nn.functional.cosine_similarity(molecule_repr, text_repr)
            # 二级loss = λ * ((latent-latent)**2),这减完不就是噪音了嘛————这里应该是去噪loss
            l2_de_noise_loss = l2_lambda * ((latent_code_init - latent) ** 2).mean()

            # 两个loss相加？
            loss = l1_similarity_loss + l2_de_noise_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        result.append(latent)
    generated_mols = []
    for item in result:
        generated = vec2smi(molecule_model, item, pad_mask_init)
        torch.cuda.empty_cache()
        generated_mols.append(generated)
    print(generated_mols)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)

    ########## for editing ##########
    parser.add_argument("--input_description", type=str, default='This molecule should have more hydrogen-bond acceptors.')
    parser.add_argument("--input_SMILES", type=str, default="CCOc1ccc(Cl)cc1C1=CC(=O)[C@@H]2Cc3cc(OC)ccc3[C@@H](C1)C2")
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--use_noise_for_init", dest="use_noise_for_init", action="store_true")
    parser.add_argument("--no_noise_for_init", dest="use_noise_for_init", action="store_false")
    parser.set_defaults(use_noise_for_init=True)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)

    ########## for MoleculeSTM ##########
    parser.add_argument("--MoleculeSTM_model_dir", type=str, default=r"E:\古老师模型\MoleculeSTM\模型自带数据\checkpoint_model\demo\demo_checkpoints_SMILES")
    parser.add_argument("--MoleculeSTM_molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph"])

    ########## for MegaMolBART ##########
    parser.add_argument("--MegaMolBART_generation_model_dir", type=str, default="../data/pretrained_MegaMolBART/checkpoints")
    parser.add_argument("--vocab_path", type=str, default=r"D:\pycharm files\MoleculeSTM-main-new\MoleculeSTM-main\scripts\MoleculeSTM\bart_vocab.txt")

    ########## for MoleculeSTM and generation projection ##########
    parser.add_argument("--language_edit_model_dir", type=str, default="edit_temp/EBM_NCE")

    ########## for editing ##########
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # 各个向量的维度
    parser.add_argument("--latent_dim", type=int, default=256)

    args = parser.parse_args()
    print("arguments\t", args)
    #########################################警告屏蔽#############################################

    # 设置cuda，设置随机种子
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    # 模型定义
    text_model, text_tokenizer, text_dim, molecule_model, molecule_dim, text2latent, mol2latent = load_language_molecule_and_edit_models(device)

    # 将三个model加载到device

    # 所有模型开启eval模式

#############################start###############################
    SMILES = args.input_SMILES
    description = args.input_description
    print("===== for description {} =====".format(description))
    print("===== for SMILES {} =====".format(SMILES))

    # 在这里输入SMILES和description
    edit(SMILES, description)

    print("\n\n\n")
