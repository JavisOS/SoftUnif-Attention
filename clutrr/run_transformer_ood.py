import os
import wandb
import csv
import re
import math
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import logging

# 配置日志：同时输出到文件和控制台
# 1. 创建 logger
logger = logging.getLogger("CLUTRR_CoT")
logger.setLevel(logging.INFO)

# 2. 创建文件处理器 (写入 training.log)
file_handler = logging.FileHandler("training.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# 3. 创建控制台处理器 (只输出简略信息，防止刷屏) -> 可选，如果你只想看进度条就不加这个
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING) # 控制台只打印警告以上，或者你可以不加这行

# 4. 添加处理器
logger.addHandler(file_handler)
# logger.addHandler(console_handler) # 如果你不想在终端看到CoT，就不要加这行

# ==========================================
# 1. 配置与常量
# ==========================================

relation_id_map = {
  'daughter': 0, 'sister': 1, 'son': 2, 'aunt': 3, 'father': 4, 'husband': 5,
  'granddaughter': 6, 'brother': 7, 'nephew': 8, 'mother': 9, 'uncle': 10,
  'grandfather': 11, 'wife': 12, 'grandmother': 13, 'niece': 14, 'grandson': 15,
  'son-in-law': 16, 'father-in-law': 17, 'daughter-in-law': 18, 'mother-in-law': 19,
  'nothing': 20,
}

# ==========================================
# 2. 数据集定义
# ==========================================

class CLUTRRDataset(Dataset):
  def __init__(self, root, dataset, split, data_percentage):
    self.dataset_dir = os.path.join(root, f"{dataset}/")
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    # 读取 CSV，跳过表头
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]
    self.data_num = math.floor(len(self.data) * data_percentage / 100)
    self.data = self.data[:self.data_num]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    # 复用原代码清洗逻辑 (注意：这里索引也要相应+1，因为有Index列)
    # A=0(Index), B=1(id), C=2(story), D=3(query), E=4(text_query), F=5(target), G=6(text_target)
    # 之前是 row[2] story, 现在应该是 row[2] 吗？
    # 让我们核对一下：
    # A(0): Index
    # B(1): id
    # C(2): story
    # D(3): query
    # F(5): target
    
    # 看起来 content 还是在 2 (C列)，query 在 3 (D列)，answer 在 5 (F列)。这部分是对的。
    context = [s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""]
    query_sub_obj = eval(self.data[i][3])
    query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())
    answer = self.data[i][5]

    # 【修正】解析中间推理路径 (Ground Truth Graph)
    # L列 (第12列) -> Index 11
    # M列 (第13列) -> Index 12
    # Q列 (第17列) -> Index 16
    ground_truth_relations = []
    try:
        story_edges = eval(self.data[i][11])  # L列
        edge_types = eval(self.data[i][12])   # M列
        node_mapping = eval(self.data[i][16]) # Q列
        
        # 反转映射 ID -> Name
        id2name = {int(v): k.lower() for k, v in node_mapping.items()}
        
        for (u, v), r_type in zip(story_edges, edge_types):
            if r_type in relation_id_map and u in id2name and v in id2name:
                # 存储为 (Subject, Object, Relation_ID)
                ground_truth_relations.append((id2name[u], id2name[v], relation_id_map[r_type]))
    except Exception as e:
        # print(f"Error parsing graph: {e}") 
        pass

    return ((context, query), answer, ground_truth_relations)

  @staticmethod
  def collate_fn(batch):
    queries = [query for ((_, query), _, _) in batch]
    contexts = [fact for ((context, _), _, _) in batch for fact in context]
    context_lens = [len(context) for ((context, _), _, _) in batch]
    context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
    answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer, _) in batch])
    ground_truth_relations = [gt for (_, _, gt) in batch]
    
    return ((contexts, queries, context_splits), answers, ground_truth_relations)

def clutrr_loader(root, dataset, batch_size, training_data_percentage):
    train_dataset = CLUTRRDataset(root, dataset, "train", training_data_percentage)
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True, num_workers=0, pin_memory=False)
    
    test_dataset = CLUTRRDataset(root, dataset, "test", 100)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader

# ==========================================
# 3. 模型定义: Universal Transformer + ALiBi
# ==========================================

class UniversalReasoningLayer(nn.Module):
    """通用推理层：将被递归调用的 Transformer Encoder Layer"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_bias=None):
        # 将 attn_bias 展平以匹配 PyTorch 接口: (Batch * Num_Heads, L, S)
        if attn_bias is not None:
            B, H, L, S = attn_bias.shape
            attn_bias = attn_bias.reshape(B * H, L, S)

        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class CLUTRRTransformerOOD(nn.Module):
    def __init__(self, device="cpu", no_fine_tune_roberta=False, steps_train=2, steps_test=8):
        super().__init__()
        self.device = device
        self.steps_train = steps_train
        self.steps_test = steps_test
        
        # 1. 感知层 (RoBERTa)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.embed_dim = self.roberta.config.hidden_size
        
        if no_fine_tune_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # 2. 推理层 (Universal Transformer)
        self.n_heads = 8
        self.reasoner = UniversalReasoningLayer(self.embed_dim, num_heads=self.n_heads)
        
        # 3. 输出层 (Relation Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, len(relation_id_map))
        )
        
        # ALiBi 斜率
        self.alibi_slopes = torch.tensor([1.0 / (2 ** (i + 1)) for i in range(self.n_heads)], device=device)

    def _compute_alibi_bias(self, positions):
        """计算基于相对距离的 Attention Bias"""
        B, N = positions.shape
        # Distance Matrix: (B, N, N)
        dist = torch.abs(positions.unsqueeze(2) - positions.unsqueeze(1))
        # Bias: (B, H, N, N) -> slope * distance
        bias = dist.unsqueeze(1) * self.alibi_slopes.view(1, self.n_heads, 1, 1) * -0.1 
        return bias

    def _get_entity_embeddings(self, contexts, context_splits):
        """
        完整复刻原 run_with_constraints.py 中的 _preprocess_contexts 逻辑。
        通过 [Name] 标记精确定位人名对应的 Token，并进行 Pooling。
        """
        
        # 1. 预处理所有 Story
        all_clean_sents = []
        all_maps = []
        story_ranges = [] # 记录 (start_idx, end_idx) in all_clean_sents
        
        cursor = 0
        for (start, end) in context_splits:
            story_clean_sents = []
            story_maps = []
            
            skip_next = False
            skip_until = 0
            for j, sentence in zip(range(start, end), contexts[start:end]):
                if skip_next:
                    if j >= skip_until: skip_next = False
                    continue

                # 合并句子逻辑
                names = re.findall(r"\[(\w+)\]", sentence)
                union_sentence = f"{sentence}"
                
                for k in range(j + 1, end):
                    next_s = contexts[k]
                    next_names = re.findall(r"\[(\w+)\]", next_s)
                    if len(names) == 1 or len(next_names) == 1:
                        if len(next_names) > 0:
                            union_sentence += f". {next_s}"
                            names += next_names
                        skip_next = True
                        skip_until = k if len(next_names) != 1 else k - 1
                    else:
                        break
                names = set(names)
                
                # Token 对齐
                clean_s = union_sentence.replace("[", "").replace("]", "")
                splitted = [u.strip() for t in union_sentence.split("[") for u in t.split("]") if u.strip() != ""]
                is_name_ids = {s: [idx for idx, sp in enumerate(splitted) if sp == s] for s in names}
                
                map_i = {}
                
                # 预计算每段的长度
                curr_len = 1 # CLS
                for sp in splitted:
                    # 模拟 RoBERTa 分词
                    ids = self.tokenizer.encode(sp, add_special_tokens=False, add_prefix_space=True)
                    l = len(ids)
                    
                    # 检查是否是名字
                    for name, phrase_indices in is_name_ids.items():
                        if sp == name: # 简单匹配
                             if name not in map_i: map_i[name] = []
                             map_i[name].extend(range(curr_len, curr_len + l))
                    
                    curr_len += l
                
                story_clean_sents.append(clean_s)
                story_maps.append(map_i)
            
            all_clean_sents.extend(story_clean_sents)
            all_maps.extend(story_maps)
            story_ranges.append((cursor, cursor + len(story_clean_sents)))
            cursor += len(story_clean_sents)

        # 2. Batch Roberta
        if len(all_clean_sents) == 0:
             return [], [], []

        tokenized = self.tokenizer(all_clean_sents, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.roberta(**tokenized)
        embeddings = outputs.last_hidden_state # (Total_Sents, Seq, Dim)

        # 3. 聚合
        batch_embs = []
        batch_pos = []
        batch_names = []

        for (start, end) in story_ranges:
            name_to_emb_list = {}
            name_to_pos_list = {}
            
            # 遍历该 Story 的所有 Merged Sentences
            for i in range(start, end):
                local_map = all_maps[i] # {name: [token_indices]}
                sent_emb = embeddings[i] # (Seq, Dim)
                
                for name, indices in local_map.items():
                    # 过滤越界索引
                    valid_indices = [idx for idx in indices if idx < sent_emb.size(0)]
                    if not valid_indices: continue
                    
                    # 提取 Embedding
                    emb = sent_emb[valid_indices].mean(dim=0)
                    
                    # 计算绝对位置: 句子索引 * 1000 + Token 索引
                    abs_pos = (i - start) * 1000 + valid_indices[0]
                    
                    if name not in name_to_emb_list:
                        name_to_emb_list[name] = []
                        name_to_pos_list[name] = []
                    
                    name_to_emb_list[name].append(emb)
                    name_to_pos_list[name].append(abs_pos)
            
            # Story 级聚合
            unique_names = sorted(name_to_emb_list.keys())
            curr_embs = []
            curr_pos = []
            
            for name in unique_names:
                curr_embs.append(torch.stack(name_to_emb_list[name]).mean(dim=0))
                curr_pos.append(sum(name_to_pos_list[name]) / len(name_to_pos_list[name]))
            
            # 异常处理 (空 Story)
            if not curr_embs:
                curr_embs = [torch.zeros(self.embed_dim, device=self.device)]
                curr_pos = [0.0]
                unique_names = ["UNK"]

            batch_embs.append(torch.stack(curr_embs))
            batch_pos.append(torch.tensor(curr_pos, device=self.device))
            batch_names.append(unique_names)
            
        return batch_embs, batch_pos, batch_names

    def forward(self, x, phase='train'):
        (contexts, queries, context_splits) = x
        
        # 1. 提取实体特征 (完整版)
        entity_embs_list, entity_pos_list, entity_names_list = self._get_entity_embeddings(contexts, context_splits)
        
        # Pad 到最大实体数
        max_len = max([e.shape[0] for e in entity_embs_list])
        B = len(entity_embs_list)
        
        padded_embs = torch.zeros(B, max_len, self.embed_dim, device=self.device)
        padded_pos = torch.zeros(B, max_len, device=self.device)
        pad_mask = torch.ones(B, max_len, device=self.device).bool() 
        
        for i, (embs, pos) in enumerate(zip(entity_embs_list, entity_pos_list)):
            L = embs.shape[0]
            padded_embs[i, :L, :] = embs
            padded_pos[i, :L] = pos
            pad_mask[i, :L] = False
            
        # 2. ALiBi Bias
        attn_bias = self._compute_alibi_bias(padded_pos)
        mask_bias = pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9
        attn_bias = attn_bias + mask_bias

        # 3. Universal Transformer 递归
        curr_state = padded_embs
        steps = self.steps_train if phase == 'train' else self.steps_test
        
        all_step_pair_logits = [] 
        
        for _ in range(steps):
            curr_state = self.reasoner(curr_state, attn_bias=attn_bias)
            
            # 预测 Pairwise 关系
            N = max_len
            x_i = curr_state.unsqueeze(2).expand(B, N, N, self.embed_dim)
            x_j = curr_state.unsqueeze(1).expand(B, N, N, self.embed_dim)
            pair_features = torch.cat([x_i, x_j], dim=-1)
            step_logits = self.classifier(pair_features)
            all_step_pair_logits.append(step_logits)

        # 4. 提取 Query Logits
        final_query_logits = []
        for i in range(B):
            sub_name, obj_name = queries[i]
            names = entity_names_list[i]
            try: sub_idx = names.index(sub_name)
            except: sub_idx = 0
            try: obj_idx = names.index(obj_name)
            except: obj_idx = 0
            
            logit = all_step_pair_logits[-1][i, sub_idx, obj_idx, :]
            final_query_logits.append(logit)
            
        final_query_logits = torch.stack(final_query_logits)
        
        return {
            "final_logits": final_query_logits,
            "all_step_logits": all_step_pair_logits, 
            "entity_names": entity_names_list
        }

# ==========================================
# 4. 训练逻辑
# ==========================================

class Trainer:
  def __init__(self, train_loader, test_loader, device, model_name, learning_rate, **args):
    self.device = device
    self.model = CLUTRRTransformerOOD(device=device, **args).to(device)
    self.model_name = model_name
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.max_accu = 0

  def compute_loss(self, y_pred, y_target, cot_targets):
    # 主任务 Loss
    final_logits = y_pred["final_logits"]
    main_loss = F.cross_entropy(final_logits, y_target.to(self.device))
    
    # CoT Loss
    aux_loss = 0.0
    step_logits_list = y_pred["all_step_logits"]
    entity_names_list = y_pred["entity_names"]
    num_valid_edges = 0
    last_step_logits = step_logits_list[-1] 
    
    for i, gt_relations in enumerate(cot_targets):
        names = entity_names_list[i]
        name2idx = {name: idx for idx, name in enumerate(names)}
        for (u, v, r_id) in gt_relations:
            if u in name2idx and v in name2idx:
                u_idx, v_idx = name2idx[u], name2idx[v]
                pred = last_step_logits[i, u_idx, v_idx, :].unsqueeze(0)
                target = torch.tensor([r_id], device=self.device)
                aux_loss += F.cross_entropy(pred, target)
                num_valid_edges += 1
                
    if num_valid_edges > 0: aux_loss = aux_loss / num_valid_edges
    return main_loss + 0.5 * aux_loss

  def train(self, num_epochs):
    for i in range(1, num_epochs + 1):
      train_loss, train_acc = self.run_epoch(i, self.train_loader, phase='train')
      test_loss, test_acc = self.run_epoch(i, self.test_loader, phase='test')
      wandb.log({"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
      if test_acc > self.max_accu:
          self.max_accu = test_acc
          print(f"New Best Accuracy: {test_acc:.2f}%")

# =======================================================
  # 1. 新增辅助函数：用于打印推理过程 (请加到 Trainer 类内部)
  # =======================================================
  def log_cot_examples(self, y_pred, queries, entity_names_list, step=0):
      # 使用 buffer 拼接字符串，一次性写入，避免多线程日志错乱
      log_msg = []
      log_msg.append(f"\n{'='*20} [Epoch {step} 推理可视化] {'='*20}")
      
      idx = 0
      names = entity_names_list[idx]
      all_step_logits = y_pred["all_step_logits"]
      id2rel = {v: k for k, v in relation_id_map.items()}
      
      sub, obj = queries[idx]
      log_msg.append(f"问题: {sub} 和 {obj} 是什么关系?")
      
      for t, step_logits in enumerate(all_step_logits):
          log_msg.append(f"Step {t+1}:")
          preds = torch.argmax(step_logits[idx], dim=-1)
          probs = torch.softmax(step_logits[idx], dim=-1)
          
          found_rel = False
          for i in range(len(names)):
              for j in range(len(names)):
                  if i == j: continue
                  r_id = int(preds[i, j].item()) # 记得加 int()
                  r_name = id2rel.get(r_id, "UNK")
                  confidence = probs[i, j, r_id].item()
                  
                  if r_name != "nothing" and confidence > 0.5: 
                      log_msg.append(f"  {names[i]} --[{r_name}]--> {names[j]} ({confidence:.2f})")
                      found_rel = True
          if not found_rel:
              log_msg.append("  (无明确关系)")
      
      log_msg.append("="*60 + "\n")
      
      # 核心：写入日志文件，而不是 print 到终端
      logger.info("\n".join(log_msg))

  # =======================================================
  # 2. 完整的 run_epoch 函数 (已嵌入可视化调用)
  # =======================================================
  def run_epoch(self, epoch, loader, phase='train'):
      if phase == 'train': self.model.train()
      else: self.model.eval()
    
      total_correct = 0; total_count = 0; total_loss = 0
      iterator = tqdm(loader, desc=f"{phase} {epoch}")
    
      for batch_idx, batch in enumerate(iterator): # 注意这里加了 enumerate
          (x_data, y_target, cot_targets) = batch
        
          if phase == 'train':
              self.optimizer.zero_grad()
              out = self.model(x_data, phase='train')
              loss = self.compute_loss(out, y_target, cot_targets)
              loss.backward()
              self.optimizer.step()
          else:
              with torch.no_grad():
                  out = self.model(x_data, phase='test')
                  loss = self.compute_loss(out, y_target, cot_targets)
        
          # 计算准确率
          preds = torch.argmax(out["final_logits"], dim=1).cpu()
          correct = (preds == y_target).sum().item()
          total_correct += correct
          total_count += len(y_target)
          total_loss += loss.item()
        
          # 更新进度条
          # 防止除零错误 (total_count / len(y_target) 等于当前的 batch 数量)
          num_batches = batch_idx + 1
          iterator.set_postfix(loss=total_loss/num_batches, acc=total_correct/total_count)

        # 逻辑：必须是 Test 阶段 + 必须是第 0 个 batch + 每隔 5 个 epoch 记录一次
          if phase == 'test' and batch_idx == 0 and epoch % 5 == 0:
              self.log_cot_examples(out, x_data[1], out["entity_names"], step=epoch)

      return total_loss / len(loader), total_correct / total_count

# ==========================================
# 5. 主入口
# ==========================================

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--n-epochs", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1e-5)
  parser.add_argument("--steps-train", type=int, default=4)
  parser.add_argument("--steps-test", type=int, default=12)
  parser.add_argument("--no-fine-tune-roberta", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  wandb.init(project="clutrr-transformer-ood", config=vars(args))
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/clutrr"))
  
  train_loader, test_loader = clutrr_loader(data_root, args.dataset, args.batch_size, 100)
  trainer = Trainer(
      train_loader, test_loader, device, "trans_ood", args.learning_rate,
      no_fine_tune_roberta=args.no_fine_tune_roberta,
      steps_train=args.steps_train, steps_test=args.steps_test
  )
  trainer.train(args.n_epochs)