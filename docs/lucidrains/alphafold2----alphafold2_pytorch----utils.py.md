# `.\lucidrains\alphafold2\alphafold2_pytorch\utils.py`

```
# 导入必要的库
import os
import re
import numpy as np
import torch
import contextlib
from functools import wraps
from einops import rearrange, repeat
# import torch_sparse # only needed for sparse nth_deg adj calculation

# 导入生物信息学相关库
from Bio import SeqIO
import itertools
import string

# 导入sidechainnet相关库
from sidechainnet.utils.sequence import ProteinVocabulary, ONE_TO_THREE_LETTER_MAP
from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, BB_BUILD_INFO, SC_BUILD_INFO
from sidechainnet.structure.StructureBuilder import _get_residue_build_iter

# 导入自定义库
import mp_nerf

# 构建蛋白质词汇表
VOCAB = ProteinVocabulary()

# 常量
import alphafold2_pytorch.constants as constants

# 辅助函数
def exists(val):
    return val is not None

# 常量：与alphafold2.py中相同
DISTANCE_THRESHOLDS = torch.linspace(2, 20, steps = constants.DISTOGRAM_BUCKETS)

# 距离分箱函数
def get_bucketed_distance_matrix(coords, mask, num_buckets = constants.DISTOGRAM_BUCKETS, ignore_index = -100):
    distances = torch.cdist(coords, coords, p=2)
    boundaries = torch.linspace(2, 20, steps = num_buckets, device = coords.device)
    discretized_distances = torch.bucketize(distances, boundaries[:-1])
    discretized_distances.masked_fill_(~(mask[..., None] & mask[..., None, :]), ignore_index)
    return discretized_distances

# 装饰器
def set_backend_kwarg(fn):
    @wraps(fn)
    def inner(*args, backend = 'auto', **kwargs):
        if backend == 'auto':
            backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
        kwargs.update(backend = backend)
        return fn(*args, **kwargs)
    return inner

def expand_dims_to(t, length = 3):
    if length == 0:
        return t
    return t.reshape(*((1,) * length), *t.shape) # will work with both torch and numpy

def expand_arg_dims(dim_len = 3):
    """ pack here for reuse. 
        turns input into (B x D x N)
    """
    def outer(fn):
        @wraps(fn)
        def inner(x, y, **kwargs):
            assert len(x.shape) == len(y.shape), "Shapes of A and B must match."
            remaining_len = dim_len - len(x.shape)
            x = expand_dims_to(x, length = remaining_len)
            y = expand_dims_to(y, length = remaining_len)
            return fn(x, y, **kwargs)
        return inner
    return outer

def invoke_torch_or_numpy(torch_fn, numpy_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            backend = kwargs.pop('backend')
            passed_args = fn(*args, **kwargs)
            passed_args = list(passed_args)
            if isinstance(passed_args[-1], dict):
                passed_kwargs = passed_args.pop()
            else:
                passed_kwargs = {}
            backend_fn = torch_fn if backend == 'torch' else numpy_fn
            return backend_fn(*passed_args, **passed_kwargs)
        return inner
    return outer

@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# 预处理数据
def get_atom_ids_dict():
    """ 获取将每个原子映射到令牌的字典。 """
    ids = set(["", "N", "CA", "C", "O"])

    for k,v in SC_BUILD_INFO.items():
        for name in v["atom-names"]:
            ids.add(name)
            
    return {k: i for i,k in enumerate(sorted(ids))}

def make_cloud_mask(aa):
    """ 重要点为1，填充点为0。 """
    mask = np.zeros(constants.NUM_COORDS_PER_RES)
    # 如果是填充令牌，则提前停止
    if aa == "_":
        return mask
    # 获取aa中的原子数
    n_atoms = 4+len( SC_BUILD_INFO[ ONE_TO_THREE_LETTER_MAP[aa] ]["atom-names"] )
    mask[:n_atoms] = 1
    return mask

def make_atom_id_embedds(aa, atom_ids):
    """ 返回aa中每个原子的令牌。 """
    mask = np.zeros(constants.NUM_COORDS_PER_RES)
    # 如果当前氨基酸是填充标记"_", 则直接返回掩码
    if aa == "_":
        return mask
    # 获取氨基酸的原子ID
    atom_list = ["N", "CA", "C", "O"] + SC_BUILD_INFO[ ONE_TO_THREE_LETTER_MAP[aa] ]["atom-names"]
    # 遍历原子列表，获取每个原子对应的ID，并存储到掩码中
    for i,atom in enumerate(atom_list):
        mask[i] = ATOM_IDS[atom]
    # 返回更新后的掩码
    return mask
# 获取原子ID字典
ATOM_IDS = get_atom_ids_dict()
# 创建自定义信息字典，包括云掩码和原子ID嵌入
CUSTOM_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                   "atom_id_embedd": make_atom_id_embedds(k, atom_ids=ATOM_IDS),
                  } for k in "ARNDCQEGHILKMFPSTWYV_"}

# 常用工具

# 从RCSB PDB下载PDB条目
def download_pdb(name, route):
    """ Downloads a PDB entry from the RCSB PDB. 
        Inputs:
        * name: str. the PDB entry id. 4 characters, capitalized.
        * route: str. route of the destin file. usually ".pdb" extension
        Output: route of destin file
    """
    os.system(f"curl https://files.rcsb.org/download/{name}.pdb > {route}")
    return route

# 清理PDB结构，只保留重要部分
def clean_pdb(name, route=None, chain_num=None):
    """ Cleans the structure to only leave the important part.
        Inputs: 
        * name: str. route of the input .pdb file
        * route: str. route of the output. will overwrite input if not provided
        * chain_num: int. index of chain to select (1-indexed as pdb files)
        Output: route of destin file.
    """
    import mdtraj
    destin = route if route is not None else name
    # 读取输入
    raw_prot = mdtraj.load_pdb(name)
    # 遍历蛋白质并选择指定的链
    idxs = []
    for chain in raw_prot.topology.chains:
        # 如果传递了参数，只选择该链
        if chain_num is not None:
            if chain_num != chain.index:
                continue
        # 选择链的索引
        chain_idxs = raw_prot.topology.select(f"chainid == {str(chain.index)}")
        idxs.extend( chain_idxs.tolist() )
    # 排序：拓扑和xyz选择是有序的
    idxs = sorted(idxs)
    # 从选择的索引子集获取新的轨迹并保存
    prot = mdtraj.Trajectory(xyz=raw_prot.xyz[:, idxs], 
                             topology=raw_prot.topology.subset(idxs))
    prot.save(destin)
    return destin

# 将自定义表示转换为.pdb文件
def custom2pdb(coords, proteinnet_id, route):
    """ Takes a custom representation and turns into a .pdb file. 
        Inputs:
        * coords: array/tensor of shape (3 x N) or (N x 3). in Angstroms.
                  same order as in the proteinnnet is assumed (same as raw pdb file)
        * proteinnet_id: str. proteinnet id format (<class>#<pdb_id>_<chain_number>_<chain_id>)
                         see: https://github.com/aqlaboratory/proteinnet/
        * route: str. destin route.
        Output: tuple of routes: (original, generated) for the structures. 
    """
    import mdtraj
    # 转换为numpy
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    # 确保(1, N, 3)
    if coords.shape[1] == 3:
        coords = coords.T
    coords = np.newaxis(coords, axis=0)
    # 获取pdb id和链号
    pdb_name, chain_num = proteinnet_id.split("#")[-1].split("_")[:-1]
    pdb_destin = "/".join(route.split("/")[:-1])+"/"+pdb_name+".pdb"
    # 下载pdb文件并选择适当的链
    download_pdb(pdb_name, pdb_destin)
    clean_pdb(pdb_destin, chain_num=chain_num)
    # 加载轨迹模板并替换坐标 - 假设顺序相同
    scaffold = mdtraj.load_pdb(pdb_destin)
    scaffold.xyz = coords
    scaffold.save(route)
    return pdb_destin, route

# 将坐标转换为PDB文件
def coords2pdb(seq, coords, cloud_mask, prefix="", name="af2_struct.pdb"):
    """ Turns coordinates into PDB files ready to be visualized. 
        Inputs:
        * seq: (L,) tensor of ints (sidechainnet aa-key pairs)
        * coords: (3, N) coords of atoms
        * cloud_mask: (L, C) boolean mask of occupied spaces in scn format
        * prefix: str. directory to save files.
        * name: str. name of destin file (ex: pred1.pdb)
    """
    scaffold = torch.zeros( cloud_mask.shape, 3 )
    scaffold[cloud_mask] = coords.cpu().float()
    # 构建结构并保存
    pred = scn.StructureBuilder( seq, crd=scaffold ) 
    # 将预测结果保存为PDB文件，文件名由前缀和名称组成
    pred.to_pdb(prefix+name)
# 定义函数，用于移除序列中的插入部分，以便在MSA中加载对齐的序列
def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

# 从MSA文件中读取前nseq个序列，自动移除插入部分
def read_msa(filename: str, nseq: int):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

# 将氨基酸id转换为用于计算ESM和MSA变换器嵌入的氨基酸字符串输入
def ids_to_embed_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(ids_to_embed_input(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(lambda c: isinstance(c, str), out)):
        return (None, ''.join(out))

    return out

# 将氨基酸id转换为用于计算ESM和MSA变换器嵌入的氨基酸字符串输入
def ids_to_prottran_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for ids in x:
        chars = ' '.join([id2aa[i] for i in ids])
        chars = re.sub(r"[UZOB]", "X", chars)
        out.append(chars)

    return out

# 获取ProtTrans嵌入
def get_prottran_embedd(seq, model, tokenizer, device = None):
    from transformers import pipeline

    fe = pipeline('feature-extraction', model = model, tokenizer = tokenizer, device = (-1 if not exists(device) else device.index))

    max_seq_len = seq.shape[1]
    embedd_inputs = ids_to_prottran_input(seq.cpu().tolist())

    embedding = fe(embedd_inputs)
    embedding = torch.tensor(embedding, device = device)

    return embedding[:, 1:(max_seq_len + 1)]

# 获取MSA嵌入
def get_msa_embedd(msa, embedd_model, batch_converter, device = None):
    """ Returns the MSA_tr embeddings for a protein.
        Inputs: 
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: MSA_tr model (see train_end2end.py for an example)
        * batch_converter: MSA_tr batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA
            * embedd_dim: number of embedding dimensions. 768 for MSA_Transformer
    """
    # use MSA transformer
    REPR_LAYER_NUM = 12
    device = seq.device
    max_seq_len = msa.shape[-1]
    embedd_inputs = ids_to_embed_input(msa.cpu().tolist())

    msa_batch_labels, msa_batch_strs, msa_batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model(msa_batch_tokens.to(device), repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :]
    return token_reps

# 获取ESM嵌入
def get_esm_embedd(seq, embedd_model, batch_converter, msa_data=None):
    """ Returns the ESM embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: ESM model (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for ESM-1b
            * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
    """
    # use ESM transformer
    # 获取输入序列的设备信息
    device = seq.device
    # 定义表示层编号
    REPR_LAYER_NUM = 33
    # 获取序列的最大长度
    max_seq_len = seq.shape[-1]
    # 将序列转换为嵌入输入
    embedd_inputs = ids_to_embed_input(seq.cpu().tolist())

    # 使用批量转换器将嵌入输入转换为批量标签、字符串和令牌
    batch_labels, batch_strs, batch_tokens = batch_converter(embedd_inputs)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用嵌入模型获取结果
        results = embedd_model(batch_tokens.to(device), repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    # 从结果中提取令牌表示，排除起始令牌
    token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :].unsqueeze(dim=1)
    # 返回令牌表示
    return token_reps
# 返回给定蛋白质的ProtT5-XL-U50嵌入
def get_t5_embedd(seq, tokenizer, encoder, msa_data=None, device=None):
    """ Returns the ProtT5-XL-U50 embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * tokenizer:  tokenizer model: T5Tokenizer
        * encoder: encoder model: T5EncoderModel
                 ex: from transformers import T5EncoderModel, T5Tokenizer
                     model_name = "Rostlab/prot_t5_xl_uniref50"
                     tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
                     model = T5EncoderModel.from_pretrained(model_name)
                     # prepare model 
                     model = model.to(device)
                     model = model.eval()
                     if torch.cuda.is_available():
                         model = model.half()
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for T5 models
            * embedd_dim: number of embedding dimensions. 1024 for T5 models
    """
    # 获取参数并准备
    device = seq.device if device is None else device
    embedd_inputs = ids_to_prottran_input(seq.cpu().tolist())
    
    # 嵌入 - https://huggingface.co/Rostlab/prot_t5_xl_uniref50
    inputs_embedding = []
    shift_left, shift_right = 0, -1
    ids = tokenizer.batch_encode_plus(embedd_inputs, add_special_tokens=True,
                                                     padding=True, 
                                                     return_tensors="pt")
    with torch.no_grad():
        embedding = encoder(input_ids=torch.tensor(ids['input_ids']).to(device), 
                            attention_mask=torch.tensor(ids["attention_mask"]).to(device))
    # 返回 (batch, seq_len, embedd_dim)
    token_reps = embedding.last_hidden_state[:, shift_left:shift_right].to(device)
    token_reps = expand_dims_to(token_reps, 4-len(token_reps.shape))
    return token_reps.float()


# 获取所有蛋白质的ID
def get_all_protein_ids(dataloader, verbose=False):
    """ Given a sidechainnet dataloader for a CASP version, 
        Returns all the ids belonging to proteins.
        Inputs: 
        * dataloader: a sidechainnet dataloader for a CASP version
        Outputs: a set containing the ids for all protein entries. 
    """
    # 在此处存储ID
    ids = set([])
    # 遍历所有批次
    for i,batch in tqdm(enumerate(dataloaders['train'])):
        # 用于同时跳出两个循环
        try:
            for i in range(batch.int_seqs.shape[0]):
                # 检查所有片段是否为：4_LETTER_PDB + NUM + CHAIN
                max_len_10 = len(batch.pids[i]) < 10 
                fragments  = [len(x) <= 4 for x in batch.pids[i].split("_")] 
                fragments_under_4 = sum(fragments) == len(fragments) # AND CONDITION
                # 记录ID
                if max_len_10 and fragments_under_4:
                    ids.add(batch.pids[i])
                else: 
                    if verbose:
                        print("skip:", batch.pids[i], "under 4", fragments)
        except StopIteration:
            break
    # 返回ID集合
    return ids
    

# 获取SCN序列的布尔掩码原子位置（不是所有氨基酸都具有相同的原子）
def scn_cloud_mask(scn_seq, boolean=True, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * scn_seq: (batch, length) sequence as provided by Sidechainnet package
        * boolean: whether to return as array of idxs or boolean values
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (batch, length, NUM_COORDS_PER_RES) boolean mask 
    """

    scn_seq = expand_dims_to(scn_seq, 2 - len(scn_seq.shape))
    # 用于坐标掩码的早期检查
    # 如果给定坐标不为空
    if coords is not None: 
        # 重新排列坐标，将坐标的维度重新排列为'... l c d'，其中c为每个残基的坐标数
        batch_mask = ( rearrange(coords, '... (l c) d -> ... l c d', c=constants.NUM_COORDS_PER_RES) == 0 ).sum(dim=-1) < coords.shape[-1]
        # 如果需要返回布尔值
        if boolean:
            # 返回布尔类型的批量掩码
            return batch_mask.bool()
        else: 
            # 返回非零元素的索引
            return batch_mask.nonzero()

    # 在 CPU 上执行循环
    device = scn_seq.device
    # 初始化空列表用于存储批量掩码
    batch_mask = []
    # 将 scn_seq 转移到 CPU 并转换为列表
    scn_seq = scn_seq.cpu().tolist()
    # 遍历 scn_seq 中的序列
    for i, seq in enumerate(scn_seq):
        # 获取每个蛋白质的掩码（每个氨基酸的点）
        batch_mask.append( torch.tensor([CUSTOM_INFO[VOCAB._int2char[aa]]['cloud_mask'] \
                                         for aa in seq]).bool().to(device) )
    # 在最后一个维度上连接
    batch_mask = torch.stack(batch_mask, dim=0)
    # 返回掩码（布尔值或索引）
    if boolean:
        # 返回布尔类型的批量掩码
        return batch_mask.bool()
    else: 
        # 返回非零元素的索引
        return batch_mask.nonzero()
def scn_backbone_mask(scn_seq, boolean=True, n_aa=3):
    """ Gets the boolean mask for N and CA positions. 
        Inputs: 
        * scn_seq: sequence(s) as provided by Sidechainnet package (int tensor/s)
        * n_aa: number of atoms in a backbone. (may include cbeta as 4th pos)
        * bool: whether to return as array of idxs or boolean values
        Outputs: (N_mask, CA_mask, C_mask)
    """
    # 创建一个与输入形状相同的全零张量
    wrapper = torch.zeros(*scn_seq.shape, n_aa).to(scn_seq.device)
    # 将N设为每个氨基酸的第一个原子，CA设为第二个原子
    wrapper[..., 0] = 1
    wrapper[..., 1] = 2
    wrapper[..., 2] = 3
    # 重新排列张量的维度
    wrapper = rearrange(wrapper, '... l c -> ... (l c)')
    # 创建N、CA、C的布尔掩码
    N_mask  = wrapper == 1
    CA_mask = wrapper == 2
    C_mask  = wrapper == 3 
    if boolean:
        return N_mask, CA_mask, C_mask
    return torch.nonzero(N_mask), torch.nonzero(CA_mask), torch.nonzero(C_mask)

def scn_atom_embedd(scn_seq):
    """ Returns the token for each atom in the aa. 
        Inputs: 
        * scn_seq: sequence(s) as provided by Sidechainnet package (int tensor/s)
    """
    device = scn_seq.device
    batch_tokens = []
    # 在CPU上进行循环
    scn_seq = scn_seq.cpu().tolist()
    for i,seq in enumerate(scn_seq):
        # 为每个氨基酸中的原子返回令牌
        batch_tokens.append( torch.tensor([CUSTOM_INFO[VOCAB.int2char(aa)]["atom_id_embedd"] \
                                           for aa in seq]) )
    batch_tokens = torch.stack(batch_tokens, dim=0).long().to(device)
    return batch_tokens

def mat_input_to_masked(x, x_mask=None, edges_mat=None, edges=None, 
                          edge_mask=None, edge_attr_mat=None, 
                          edge_attr=None): 
    """ Turns the padded input and edges + mask into the
        non-padded inputs and edges.
        At least one of (edges_mat, edges) must be provided. 
        The same format for edges and edge_attr must be provided 
        (either adj matrix form or flattened form).
        Inputs: 
        * x: ((batch), N, D) a tensor of N nodes and D dims for each one
        * x_mask: ((batch), N,) boolean mask for x
        * edges: (2, E) optional. indices of the corresponding adjancecy matrix. 
        * edges_mat: ((batch), N, N) optional. adjacency matrix for x
        * edge_mask: optional. boolean mask of the same shape of either "edge_mat" or "edges".
        * edge_attr: (E, D_edge) optional. edge attributes of D_edge dims.
        * edge_attr_mat: ((batch), N, N) optional. adjacency matrix with features 
        Outputs: 
        * x: (N_, D) the masked node features
        * edge_index: (2, E_) the masked x-indices for the edges
        * edge_attr: (E_, D_edge) the masked edge attributes 
        * batch: (N_,) the corresponding index in the batch for each node 
    """
    # 折叠批处理维度
    if len(x.shape) == 3:
        batch_dim = x.shape[1] 
        # 为x和其掩码折叠
        x = rearrange(x, 'b n d ... -> (b n) d ...')
        if x_mask is not None:
            x_mask = rearrange(x_mask, 'b n ... -> (b n) ...')
        else: 
            x_mask = torch.ones_like(x[..., 0]).bool()

        # 如果需要，为边索引和属性折叠
        if edges_mat is not None and edges is None:
            edges = torch.nonzero(edges_mat, as_tuple=False).t()
            edges = edges[1:] + edges[:1]*batch_dim
        # 获取每个节点的批处理标识符
        batch = (torch.arange(x.shape[0], device=x.device) // batch_dim)[x_mask]
    else:
        # 将边转换为索引格式
        if edges_mat is not None and edges is None:
            edges = torch.nonzero(edges_mat, as_tuple=False).t()
        # 获取每个节点的批处理标识符
        batch = torch.zeros(x.shape[0], device=x.device).to(x.device)

    # 如果提供了边属性矩阵，则调整边属性
    if edge_attr_mat is not None and edge_attr is None: 
            edge_attr = edge_attr[edges_mat.bool()]
    # 如果未提供边掩码，则生成边掩码
    if edge_mask is None:
        edge_mask = torch.ones_like(edges[-1]).bool()
    # 开始应用掩码，筛选出符合条件的元素
    x = x[x_mask]
    # 处理边的索引：获取方阵并移除所有非编码原子
    # 计算边的最大值，用于创建方阵
    max_num = edges.max().item()+1
    # 创建一个全零的方阵，大小为最大值，转移到与 x 相同的设备上
    wrapper = torch.zeros(max_num, max_num).to(x.device)
    # 根据边的索引，将对应位置置为 1
    wrapper[edges[0][edge_mask], edges[1][edge_mask]] = 1
    # 根据 x 的掩码，筛选出对应的行和列，得到新的方阵
    wrapper = wrapper[x_mask, :][:, x_mask]
    # 找到非零元素的索引，作为新的边索引
    edge_index = torch.nonzero(wrapper, as_tuple=False).t()
    # 处理边的属性
    # 如果边属性不为空，则根据边的掩码筛选出对应的属性，否则为 None
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    
    # 返回处理后的 x、边索引、边属性和批次信息
    return x, edge_index, edge_attr, batch
def nth_deg_adjacency(adj_mat, n=1, sparse=False):
    """ Calculates the n-th degree adjacency matrix.
        计算第 n 次邻接矩阵。
        Performs mm of adj_mat and adds the newly added.
        执行 adj_mat 的矩阵乘法并添加新添加的部分。
        Default is dense. Mods for sparse version are done when needed.
        默认为密集矩阵。在需要时进行稀疏版本的修改。
        Inputs: 
        * adj_mat: (N, N) adjacency tensor
        * n: int. degree of the output adjacency
        * sparse: bool. whether to use torch-sparse module
        输入：
        * adj_mat: (N, N) 邻接张量
        * n: int。输出邻接的度
        * sparse: bool。是否使用 torch-sparse 模块
        Outputs: 
        * edge_idxs: ij positions of the adjacency matrix
        * edge_attrs: degree of connectivity (1 for neighs, 2 for neighs^2, ... )
        输出：
        * edge_idxs: 邻接矩阵的 ij 位置
        * edge_attrs: 连通度的度数（1 表示邻居，2 表示邻居的平方，...）
    """
    adj_mat = adj_mat.float()
    attr_mat = torch.zeros_like(adj_mat)
    new_adj_mat = adj_mat.clone()
        
    for i in range(n):
        if i == 0:
            attr_mat += adj_mat
            continue

        if i == 1 and sparse: 
            idxs = adj_mat.nonzero().t()
            vals = adj_mat[idxs[0], idxs[1]]
            new_idxs = idxs.clone()
            new_vals = vals.clone() 
            m, k, n = 3 * [adj_mat.shape[0]] # (m, n) * (n, k) , but adj_mats are squared: m=n=k            

        if sparse:
            new_idxs, new_vals = torch_sparse.spspmm(new_idxs, new_vals, idxs, vals, m=m, k=k, n=n)
            new_vals = new_vals.bool().float()
            # fill by indexes bc it's faster in sparse mode - will need an intersection function
            previous = attr_mat[new_idxs[0], new_idxs[1]].bool().float()
            attr_mat[new_idxs[0], new_idxs[1]] = (1 - previous)*(i+1)
        else:
            new_adj_mat = (new_adj_mat @ adj_mat).bool().float() 
            attr_mat.masked_fill( (new_adj_mat - attr_mat.bool().float()).bool(), i+1 )

    return new_adj_mat, attr_mat

def prot_covalent_bond(seqs, adj_degree=1, cloud_mask=None, mat=True, sparse=False):
    """ Returns the idxs of covalent bonds for a protein.
        返回蛋白质的共价键的索引。
        Inputs 
        * seq: (b, n) torch long.
        * adj_degree: int. adjacency degree
        * cloud_mask: mask selecting the present atoms.
        * mat: whether to return as indexes of only atoms (PyG version)
               or matrices of masked atoms (for batched training). 
               for indexes, only 1 seq is supported.
        * sparse: bool. whether to use torch_sparse for adj_mat calc
        输入
        * seq: (b, n) torch long.
        * adj_degree: int. 邻接度
        * cloud_mask: 选择当前原子的掩码。
        * mat: 是否返回仅原子的索引（PyG 版本）或掩码原子的矩阵（用于批量训练）。
               对于索引，仅支持 1 个 seq。
        * sparse: bool。是否使用 torch_sparse 计算 adj_mat
        Outputs: edge_idxs, edge_types (degree of adjacency). 
        输出：edge_idxs, edge_types（邻接度）。
    """
    device = seqs.device
    # set up container adj_mat (will get trimmed - less than 14)
    next_aa = NUM_COORDS_PER_RES
    adj_mat = torch.zeros(seqs.shape[0], *[seqs.shape[1]*NUM_COORDS_PER_RES]*2)
    # not needed to device since it's only for indices
    seq_list = seqs.cpu().tolist()
    for s,seq in enumerate(seq_list): 
        next_idx = 0
        for i,idx in enumerate(seq):
            aa_bonds = constants.AA_DATA[VOCAB._int2char[idx]]['bonds']
            # if no edges -> padding token -> finish bond creation for this seq
            if len(aa_bonds) == 0: 
                break
            # correct next position. for indexes functionality
            next_aa = max(aa_bonds, key=lambda x: max(x))[-1]
            # offset by pos in chain ( intra-aa bonds + with next aa )
            bonds = next_idx + torch.tensor( aa_bonds + [[2, next_aa]] ).t()
            next_idx += next_aa
            # delete link with next if final AA in seq
            if i == seqs.shape[1] - 1:
                bonds = bonds[:, :-1]
            # modify adj mat
            adj_mat[s, bonds[0], bonds[1]] = 1
        # convert to undirected
        adj_mat[s] = adj_mat[s] + adj_mat[s].t()
        # do N_th degree adjacency
        adj_mat, attr_mat = nth_deg_adjacency(adj_mat, n=adj_degree, sparse=sparse)

    if mat: 
        # return the full matrix/tensor
        return attr_mat.bool().to(seqs.device), attr_mat.to(device)
    else:
        edge_idxs = attr_mat[0].nonzero().t().long()
        edge_types = attr_mat[0, edge_idxs[0], edge_idxs[1]]
        return edge_idxs.to(seqs.device), edge_types.to(seqs.device)
def sidechain_container(seqs, backbones, atom_mask, cloud_mask=None, padding_tok=20):
    """ Gets a backbone of the protein, returns the whole coordinates
        with sidechains (same format as sidechainnet). Keeps differentiability.
        Inputs: 
        * seqs: (batch, L) either tensor or list
        * backbones: (batch, L*n_aa, 3): assume batch=1 (could be extended (?not tested)).
                     Coords for (N-term, C-alpha, C-term, (c_beta)) of every aa.
        * atom_mask: (14,). int or bool tensor specifying which atoms are passed.
        * cloud_mask: (batch, l, c). optional. cloud mask from scn_cloud_mask`.
                      sets point outside of mask to 0. if passed, else c_alpha
        * padding: int. padding token. same as in sidechainnet: 20
        Outputs: whole coordinates of shape (batch, L, 14, 3)
    """
    # 将 atom_mask 转换为布尔类型，并移动到 CPU 上进行分离
    atom_mask = atom_mask.bool().cpu().detach()
    # 计算累积的原子掩码
    cum_atom_mask = atom_mask.cumsum(dim=-1).tolist()

    # 获取 backbones 的设备信息和形状
    device = backbones.device
    batch, length = backbones.shape[0], backbones.shape[1] // cum_atom_mask[-1]
    predicted  = rearrange(backbones, 'b (l back) d -> b l back d', l=length)

    # 如果整个链已经被预测，则直接返回预测结果
    if cum_atom_mask[-1] == 14:
        return predicted

    # 从 (N, CA, C, CB) 构建支架 - 在 CPU 上进行
    new_coords = torch.zeros(batch, length, constants.NUM_COORDS_PER_RES, 3)
    predicted  = predicted.cpu() if predicted.is_cuda else predicted

    # 如果原子已经传递，则填充原子
    for i,atom in enumerate(atom_mask.tolist()):
        if atom:
            new_coords[:, :, i] = predicted[:, :, cum_atom_mask[i]-1]

    # 如果原子未传递，则生成侧链
    for s,seq in enumerate(seqs): 
        # 格式化序列
        if isinstance(seq, torch.Tensor):
            padding = (seq == padding_tok).sum().item()
            seq_str = ''.join([VOCAB._int2char[aa] for aa in seq.cpu().numpy()[:-padding or None]])
        elif isinstance(seq, str):
            padding = 0
            seq_str = seq
        # 获取支架
        scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq_str, angles=None, device="cpu")
        coords, _ = mp_nerf.proteins.sidechain_fold(wrapper = new_coords[s, :-padding or None].detach(),
                                                    **scaffolds, c_beta = cum_atom_mask[4]==5)
        # 添加分离的 scn
        for i,atom in enumerate(atom_mask.tolist()):
            if not atom:
                new_coords[:, :-padding or None, i] = coords[:, i]

    new_coords = new_coords.to(device)
    if cloud_mask is not None:
        new_coords[torch.logical_not(cloud_mask)] = 0.

    # 用前一个点位置（或 N 如果位置是 AA 的第 13 个）替换任何 NaN
    nan_mask = list(torch.nonzero(new_coords!=new_coords, as_tuple=True))
    new_coords[nan_mask[0], nan_mask[1], nan_mask[2]] = new_coords[nan_mask[0], 
                                                                   nan_mask[1],
                                                                   (nan_mask[-2]+1) % new_coords.shape[-1] 
    return new_coords.to(device)


# 距离工具（距离直方图到距离矩阵 + 掩码）

def center_distogram_torch(distogram, bins=DISTANCE_THRESHOLDS, min_t=1., center="mean", wide="std"):
    """ Returns the central estimate of a distogram. Median for now.
        Inputs:
        * distogram: (batch, N, N, B) where B is the number of buckets.
        * bins: (B,) containing the cutoffs for the different buckets
        * min_t: float. lower bound for distances.
        Outputs:
        * central: (batch, N, N)
        * dispersion: (batch, N, N)
        * weights: (batch, N, N)
    """
    shape, device = distogram.shape, distogram.device
    # 将阈值转换为权重，并找到每个桶的平均值
    n_bins = ( bins - 0.5 * (bins[2] - bins[1]) ).to(device)
    n_bins[0]  = 1.5
    n_bins[-1] = 1.33*bins[-1] # 忽略最后一个阈值以上的值
    # 计算中心性和离散度的度量 -
    # 计算直方图的幅度
    magnitudes = distogram.sum(dim=-1)
    # 如果选择中心为"中位数"
    if center == "median":
        # 计算累积分布
        cum_dist = torch.cumsum(distogram, dim=-1)
        # 计算中位数
        medium = 0.5 * cum_dist[..., -1:]
        # 找到中心位置
        central = torch.searchsorted(cum_dist, medium).squeeze()
        # 限制中心位置在合理范围内
        central = n_bins[torch.min(central, max_bin_allowed)]
    # 如果选择中心为"均值"
    elif center == "mean":
        # 计算加权平均值
        central = (distogram * n_bins).sum(dim=-1) / magnitudes
    # 创建最后一个类别的掩码 - (IGNORE_INDEX)
    mask = (central <= bins[-2].item()).float()
    # 将对角线上的距离设为0，避免就地操作错误
    diag_idxs = np.arange(shape[-2])
    central = expand_dims_to(central, 3 - len(central.shape))
    central[:, diag_idxs, diag_idxs] *= 0.
    # 提供权重
    if wide == "var":
        # 计算方差
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes
    elif wide == "std":
        # 计算标准差
        dispersion = ((distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes).sqrt()
    else:
        # 如果未指定宽度，则权重为0
        dispersion = torch.zeros_like(central, device=device)
    # 重新缩放到0-1。较低的标准差/方差 --> 权重=1。将潜在的NaN值设为0
    weights = mask / (1 + dispersion)
    weights[weights != weights] *= 0.
    weights[:, diag_idxs, diag_idxs] *= 0.
    return central, weights
# 将距离矩阵转换为三维坐标
def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ 获取距离矩阵，输出三维坐标。参见下面的包装器。
        假设（目前）距离图是（N x N）且对称的
        输出：
        * best_3d_coords: （batch x 3 x N）
        * historic_stresses: （batch x steps）
    """
    device, dtype = pre_dist_mat.device, pre_dist_mat.type()
    # 确保批处理的MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length=(3 - len(pre_dist_mat.shape)))
    # 开始
    batch, N, _ = pre_dist_mat.shape
    diag_idxs = np.arange(N)
    his = [torch.tensor([np.inf]*batch, device=device)]

    # 通过特征分解进行初始化：https://www.lptmc.jussieu.fr/user/lesne/bioinformatics.pdf
    # 参考：https://www.biorxiv.org/content/10.1101/2020.11.27.401232v1.full.pdf
    D = pre_dist_mat**2
    M = 0.5 * (D[:, :1, :] + D[:, :, :1] - D) 
    # 使用循环SVD，因为它更快：（在CPU上快2-3倍，在GPU上快1-2倍）
    # https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336
    svds = [torch.svd_lowrank(mi) for mi in M]
    u = torch.stack([svd[0] for svd in svds], dim=0)
    s = torch.stack([svd[1] for svd in svds], dim=0)
    v = torch.stack([svd[2] for svd in svds], dim=0)
    best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[..., :3]

    # 仅使用特征分解 - 更快但不支持权重
    if weights is None and eigen==True:
        return torch.transpose(best_3d_coords, -1, -2), torch.zeros_like(torch.stack(his, dim=0))
    elif eigen==True:
        if verbose:
            print("如果激活权重，则无法使用特征分解标志。回退到迭代方式")

    # 继续迭代方式
    if weights is None:
        weights = torch.ones_like(pre_dist_mat)

    # 迭代更新：
    for i in range(iters):
        # 计算坐标和应力的距离矩阵
        best_3d_coords = best_3d_coords.contiguous()
        dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()

        stress = (weights * (dist_mat - pre_dist_mat)**2).sum(dim=(-1,-2)) * 0.5
        # 扰动 - 使用Guttman变换更新X - 类似于sklearn
        dist_mat[dist_mat <= 0] += 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

        # 更新
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))

        if verbose >= 2:
            print('迭代次数：%d，��力 %s' % (i, stress))
        # 如果相对改进超过容差，则更新指标
        if (his[-1] - stress / dis).mean() <= tol:
            if verbose:
                print('在迭代 %d 中以应力 %s 结束' % (i, stress / dis))
            break

        best_3d_coords = coords
        his.append(stress / dis)

    return torch.transpose(best_3d_coords, -1, -2), torch.stack(his, dim=0)

def mds_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ 获取距离矩阵。输出三维坐标。参见下面的包装器。
        假设（目前）距离图是（N x N）且对称的
        输出：
        * best_3d_coords: （3 x N）
        * historic_stress 
    """
    if weights is None:
        weights = np.ones_like(pre_dist_mat)

    # 确保批处理的MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length=(3 - len(pre_dist_mat.shape)))
    # 开始
    batch, N, _ = pre_dist_mat.shape
    his = [np.inf]
    # 初始化随机坐标
    best_stress = np.inf * np.ones(batch)
    best_3d_coords = 2*np.random.rand(batch, 3, N) - 1
    # 迭代更新：
    # 遍历指定次数的迭代
    for i in range(iters):
        # 计算坐标和压力的距离矩阵
        dist_mat = np.linalg.norm(best_3d_coords[:, :, :, None] - best_3d_coords[:, :, None, :], axis=-3)
        stress   = (( weights * (dist_mat - pre_dist_mat) )**2).sum(axis=(-1, -2)) * 0.5
        # 扰动 - 使用 Guttman 变换更新 X - 类似于 sklearn
        dist_mat[dist_mat == 0] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio 
        B[:, np.arange(N), np.arange(N)] += ratio.sum(axis=-1)
        # 更新 - 双重转置。待办事项：考虑修复
        coords = (1. / N * np.matmul(best_3d_coords, B))
        dis = np.linalg.norm(coords, axis=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # 如果相对改进超过容差，则更新指标
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        best_stress = stress / dis
        his.append(best_stress)

    return best_3d_coords, np.array(his)
# 定义一个函数，用于计算四个坐标点之间的二面角（dihedral angle）并返回结果，使用 torch 库
def get_dihedral_torch(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Can't use torch.dot bc it does not broadcast
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    # 计算四个坐标点之间的向量
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    # 使用 torch 库中的 atan2 函数计算二面角
    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) ) 


# 定义一个函数，用于计算四个坐标点之间的二面角（dihedral angle）并返回结果，使用 numpy 库
def get_dihedral_numpy(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
        * c1: (batch, 3) or (3,)
    """
    # 计算四个坐标点之间的向量
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    # 使用 numpy 库中的 arctan2 函数计算二面角
    return np.arctan2( ( (np.linalg.norm(u2, axis=-1, keepdims=True) * u1) * np.cross(u2,u3, axis=-1)).sum(axis=-1),  
                       ( np.cross(u1,u2, axis=-1) * np.cross(u2, u3, axis=-1) ).sum(axis=-1) ) 


# 定义一个函数，用于计算蛋白质的 phi 角度，选择具有最多负 phi 角度的镜像
def calc_phis_torch(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (batch, N) boolean mask for N-term positions
        * CA_mask: (batch, N) boolean mask for C-alpha positions
        * C_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
        Note: use [0] since all prots in batch have same backbone
    """ 
    # 分离梯度以进行角度计算 - 选择镜像
    pred_coords_ = torch.transpose(pred_coords.detach(), -1 , -2).cpu()
    # 确保维度正确
    N_mask = expand_dims_to( N_mask, 2-len(N_mask.shape) )
    CA_mask = expand_dims_to( CA_mask, 2-len(CA_mask.shape) )
    if C_mask is not None: 
        C_mask = expand_dims_to( C_mask, 2-len(C_mask.shape) )
    else:
        C_mask = torch.logical_not(torch.logical_or(N_mask,CA_mask))
    # 选择点
    n_terms  = pred_coords_[:, N_mask[0].squeeze()]
    c_alphas = pred_coords_[:, CA_mask[0].squeeze()]
    c_terms  = pred_coords_[:, C_mask[0].squeeze()]
    # 计算每个批次中每个蛋白质的 phi 角度
    phis = [get_dihedral_torch(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # 返回小于 0 的比例
    if prop: 
        return torch.stack([(x<0).float().mean() for x in phis], dim=0 ) 
    return phis


def calc_phis_numpy(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Filters mirrors selecting the 1 with most N of negative phis.
        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (N, ) boolean mask for N-term positions
        * CA_mask: (N, ) boolean mask for C-alpha positions
        * C_mask: (N, ) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
    """ 
    # detach gradients for angle calculation - mirror selection
    # 转置预测坐标，将维度顺序变为 (batch, N, 3)
    pred_coords_ = np.transpose(pred_coords, (0, 2, 1))
    # 获取 N 位置的坐标
    n_terms  = pred_coords_[:, N_mask.squeeze()]
    # 获取 C-alpha 位置的坐标
    c_alphas = pred_coords_[:, CA_mask.squeeze()]
    # 如果未传入 C_mask，则自动选择 C-term
    if C_mask is not None: 
        c_terms = pred_coords_[:, C_mask]
    else:
        # 根据 N_mask 和 CA_mask 自动计算 C-term
        c_terms  = pred_coords_[:, (np.ones_like(N_mask)-N_mask-CA_mask).squeeze().astype(bool) ]
    # 计算每个批次中蛋白质的 phi 角度
    phis = [get_dihedral_numpy(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # 返回小于 0 的比例
    if prop: 
        return np.array( [(x<0).mean() for x in phis] ) 
    return phis
# alignment by centering + rotation to compute optimal RMSD
# adapted from : https://github.com/charnley/rmsd/

def kabsch_torch(X, Y, cpu=True):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    device = X.device
    # center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu: 
        C = C.cpu()
    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C)
    
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_

def kabsch_numpy(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    # center X and Y to the origin
    X_ = X - X.mean(axis=-1, keepdims=True)
    Y_ = Y - Y.mean(axis=-1, keepdims=True)
    # calculate convariance matrix (for each prot in the batch)
    C = np.dot(X_, Y_.transpose())
    # Optimal rotation matrix via SVD
    V, S, W = np.linalg.svd(C)
    # determinant sign for direction correction
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = np.dot(V, W)
    # calculate rotations
    X_ = np.dot(X_.T, U).T
    # return centered and aligned
    return X_, Y_

# metrics - more formulas here: http://predictioncenter.org/casp12/doc/help.html

def distmat_loss_torch(X=None, Y=None, X_mat=None, Y_mat=None, p=2, q=2,
                       custom=None, distmat_mask=None, clamp=None):
    """ Calculates a loss on the distance matrix - no need to align structs.
        Inputs: 
        * X: (N, d) tensor. the predicted structure. One of (X, X_mat) is needed.
        * X_mat: (N, N) tensor. the predicted distance matrix. Optional ()
        * Y: (N, d) tensor. the true structure. One of (Y, Y_mat) is needed.
        * Y_mat: (N, N) tensor. the predicted distance matrix. Optional ()
        * p: int. power for the distance calculation (2 for euclidean)
        * q: float. power for the scaling of the loss (2 for MSE, 1 for MAE, etc)
        * custom: func or None. custom loss over distance matrices. 
                  ex: lambda x,y: 1 - 1/ (1 + ((x-y))**2) (1 is very bad. 0 is good)
        * distmat_mask: (N, N) mask (boolean or weights for each ij pos). optional.
        * clamp: tuple of (min,max) values for clipping distance matrices. ex: (0,150)
    """
    assert (X is not None or X_mat is not None) and \
           (Y is not None or Y_mat is not None), "The true and predicted coords or dist mats must be provided"
    # calculate distance matrices
    if X_mat is None: 
        X = X.squeeze()
        if clamp is not None:
            X = torch.clamp(X, *clamp)
        X_mat = torch.cdist(X, X, p=p)
    if Y_mat is None: 
        Y = Y.squeeze()
        if clamp is not None:
            Y = torch.clamp(Y, *clamp)
        Y_mat = torch.cdist(Y, Y, p=p)
    if distmat_mask is None:
        distmat_mask = torch.ones_like(Y_mat).bool()

    # do custom expression if passed
    if custom is not None:
        return custom(X_mat.squeeze(), Y_mat.squeeze()).mean()
    # **2 ensures always positive. Later scale back to desired power
    else:
        loss = ( X_mat - Y_mat )**2 
        if q != 2:
            loss = loss**(q/2)
        return loss[distmat_mask].mean()

def rmsd_torch(X, Y):
    # 假设 x 和 y 都是 (B x D x N) 的张量，计算它们的平方差，然后对最后两个维度求平均值再开方，返回结果
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )
def rmsd_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    # 计算均方根偏差(RMSD)的numpy实现
    return np.sqrt( np.mean((X - Y)**2, axis=(-1, -2)) )

def gdt_torch(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    # 计算全局距离差(GDT)的torch实现
    device = X.device
    if weights is None:
        weights = torch.ones(1,len(cutoffs))
    else:
        weights = torch.tensor([weights]).to(device)
    # 初始化GDT为零，并填充值
    GDT = torch.zeros(X.shape[0], len(cutoffs), device=device)
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # 遍历阈值
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).float().mean(dim=-1)
    # 加权平均
    return (GDT*weights).mean(-1)

def gdt_numpy(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    # 计算全局距离差(GDT)的numpy实现
    if weights is None:
        weights = np.ones( (1,len(cutoffs)) )
    else:
        weights = np.array([weights])
    # 初始化GDT为零，并填充值
    GDT = np.zeros( (X.shape[0], len(cutoffs)) )
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # 遍历阈值
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).mean(axis=-1)
    # 加权平均
    return (GDT*weights).mean(-1)

def tmscore_torch(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    # 计算TM得分的torch实现
    L = max(15, X.shape[-1])
    d0 = 1.24 * (L - 15)**(1/3) - 1.8
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # 公式计算
    return (1 / (1 + (dist/d0)**2)).mean(dim=-1)

def tmscore_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    # 计算TM得分的numpy实现
    L = max(15, X.shape[-1])
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # 公式计算
    return (1 / (1 + (dist/d0)**2)).mean(axis=-1)

def mdscaling_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, 
                    eigen=False, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # MDS的torch实现，处理蛋白质的特殊情况（镜像等）
    preds, stresses = mds_torch(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, eigen=eigen, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    phi_ratios = calc_phis_torch(preds, N_mask, CA_mask, C_mask, prop=True)
    to_correct = torch.nonzero( (phi_ratios < 0.5)).view(-1)
    # 修正镜像
    preds[to_correct, -1] = (-1)*preds[to_correct, -1]
    if verbose == 2:
        print("Corrected mirror idxs:", to_correct)
            
    return preds, stresses

def mdscaling_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # MDS的numpy实现，处理蛋白质的特殊情况（镜像等）
    preds, stresses = mds_numpy(pre_dist_mat, weights=weights,iters=iters, 
                                              tol=tol, verbose=verbose)
    if not fix_mirror:
        return preds, stresses

    phi_ratios = calc_phis_numpy(preds, N_mask, CA_mask, C_mask, prop=True)
    for i,pred in enumerate(preds):
        if phi_ratios < 0.5:
            preds[i, -1] = (-1)*preds[i, -1]
            if verbose == 2:
                print("Corrected mirror in struct no.", i)

    return preds, stresses
def lddt_ca_torch(true_coords, pred_coords, cloud_mask, r_0=15.):
    """ Computes the lddt score for each C_alpha.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * true_coords: (b, l, c, d) in sidechainnet format.
        * pred_coords: (b, l, c, d) in sidechainnet format.
        * cloud_mask : (b, l, c) adapted for scn format.
        * r_0: float. maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt for c_alpha scores (ranging between 0 and 1)
        See wrapper below.
    """
    device, dtype = true_coords.device, true_coords.type()
    thresholds = torch.tensor([0.5, 1, 2, 4], device=device).type(dtype)
    # adapt masks
    cloud_mask = cloud_mask.bool().cpu()
    c_alpha_mask  = torch.zeros(cloud_mask.shape[1:], device=device).bool() # doesn't have batch dim
    c_alpha_mask[..., 1] = True
    # container for c_alpha scores (between 0,1)
    wrapper = torch.zeros(true_coords.shape[:2], device=device).type(dtype)

    for bi, seq in enumerate(true_coords):
        # select atoms for study
        c_alphas = cloud_mask[bi]*c_alpha_mask # only pick c_alpha positions
        selected_pred = pred_coords[bi, c_alphas, :] 
        selected_target = true_coords[bi, c_alphas, :]
        # get number under distance
        dist_mat_pred = torch.cdist(selected_pred, selected_pred, p=2)
        dist_mat_target = torch.cdist(selected_target, selected_target, p=2) 
        under_r0_target = dist_mat_target < r_0
        compare_dists = torch.abs(dist_mat_pred - dist_mat_target)[under_r0_target]
        # measure diff below threshold
        score = torch.zeros_like(under_r0_target).float()
        max_score = torch.zeros_like(under_r0_target).float()
        max_score[under_r0_target] = 4.
        # measure under how many thresholds
        score[under_r0_target] = thresholds.shape[0] - \
                                 torch.bucketize( compare_dists, boundaries=thresholds ).float()
        # dont include diagonal
        l_mask = c_alphas.float().sum(dim=-1).bool()
        wrapper[bi, l_mask] = ( score.sum(dim=-1) - thresholds.shape[0] ) / \
                              ( max_score.sum(dim=-1) - thresholds.shape[0] )

    return wrapper


################
### WRAPPERS ###
################

@set_backend_kwarg
@invoke_torch_or_numpy(mdscaling_torch, mdscaling_numpy)
def MDScaling(pre_dist_mat, **kwargs):
    """ Gets distance matrix (-ces). Outputs 3d.  
        Assumes (for now) distrogram is (N x N) and symmetric.
        For support of ditograms: see `center_distogram_torch()`
        Inputs:
        * pre_dist_mat: (1, N, N) distance matrix.
        * weights: optional. (N x N) pairwise relative weights .
        * iters: number of iterations to run the algorithm on
        * tol: relative tolerance at which to stop the algorithm if no better
               improvement is achieved
        * backend: one of ["numpy", "torch", "auto"] for backend choice
        * fix_mirror: int. number of iterations to run the 3d generation and
                      pick the best mirror (highest number of negative phis)
        * N_mask: indexing array/tensor for indices of backbone N.
                  Only used if fix_mirror > 0.
        * CA_mask: indexing array/tensor for indices of backbone C_alpha.
                   Only used if fix_mirror > 0.
        * verbose: whether to print logs
        Outputs:
        * best_3d_coords: (3 x N)
        * historic_stress: (timesteps, )
    """
    pre_dist_mat = expand_dims_to(pre_dist_mat, 3 - len(pre_dist_mat.shape))
    return pre_dist_mat, kwargs

@expand_arg_dims(dim_len = 2)
@set_backend_kwarg
@invoke_torch_or_numpy(kabsch_torch, kabsch_numpy)
def Kabsch(A, B):
    """
    返回通过将 A 对齐到 B 而产生的 Kabsch 旋转矩阵。
    从 https://github.com/charnley/rmsd/ 改编而来。
    * 输入：
        * A,B 是 (3 x N) 的矩阵
        * backend: 选择 ["numpy", "torch", "auto"] 之一作为后端
    * 输出：形状为 (3 x N) 的张量/数组
    """
    # 运行计算 - 选择第 0 个，因为额外的维度已经被创建
    return A, B
# 为 RMSD 函数添加装饰器，用于扩展参数维度
# 为 RMSD 函数添加装饰器，设置后端参数
# 调用 torch 或 numpy 中的 rmsd_torch 或 rmsd_numpy 函数
def RMSD(A, B):
    """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs: 
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    return A, B

# 为 GDT 函数添加装饰器，用于扩展参数维度
# 为 GDT 函数添加装饰器，设置后端参数
# 调用 torch 或 numpy 中的 gdt_torch 或 gdt_numpy 函数
def GDT(A, B, *, mode="TS", cutoffs=[1,2,4,8], weights=None):
    """ Returns GDT score as defined here (highre is better):
        Supports both TS and HA
        http://predictioncenter.org/casp12/doc/help.html
        * Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * cutoffs: defines thresholds for gdt
            * weights: list containing the weights
            * mode: one of ["numpy", "torch", "auto"] for backend
        * Outputs: tensor/array of size (B,)
    """
    # 根据不同的模式设置不同的截断值和权重
    cutoffs = [0.5,1,2,4] if mode in ["HA", "ha"] else [1,2,4,8]
    # 计算 GDT
    return A, B, cutoffs, {'weights': weights}

# 为 TMscore 函数添加装饰器，用于扩展参数维度
# 为 TMscore 函数添加装饰器，设置后端参数
# 调用 torch 或 numpy 中的 tmscore_torch 或 tmscore_numpy 函数
def TMscore(A, B):
    """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding. 
        = 0.2. https://en.wikipedia.org/wiki/Template_modeling_score
        Warning! It's not exactly the code in:
        https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp
        but will suffice for now. 
        Inputs: 
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
    return A, B
```