import random
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset
from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np

class NegativeSampler(ABC):
    def __init__(self, dataset, device: torch.device):
        self.dataset = dataset
        self.device = device

    @abstractmethod
    def neg_sample(self, heads, rels, tails):
        """
        :param heads: 由 batch_size 个 head idx 组成的 tensor，size: [batch_size]
        :param rels: size [batch_size]
        :param tails: size [batch_size]
        """
        pass


def create_mapping(base_dir):
    """
    create mapping of `entity2id` and `relation2id`
    """
    entity2id = dict()
    with open(base_dir + 'entity2id.txt') as f:
        for line in f:
            parts = line.split('/')  
            if len(parts) > 1:
                entity = '/'.join(parts[:-1]) 
                entity_id = parts[-1].strip()  
                entity2id[entity] = int(entity_id)
  
    rel2id = dict()
    with open(base_dir + 'relation2id.txt') as f:
        for line in f:
            parts = line.split('/')  
            if len(parts) > 1:
                relation = '/'.join(parts[:-1])  
                relation_id = parts[-1].strip()  
                rel2id[relation] = int(relation_id)
    return entity2id, rel2id


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class RandomNegativeSampler(NegativeSampler):
    """
    Randomly replace the head or tail to perform sampling
    """

    def __init__(self, dataset, device):
        super(RandomNegativeSampler, self).__init__(dataset, device)

    def neg_sample(self, heads, rels, tails):
        ent_num = len(self.dataset.entity2id)
        head_or_tail = torch.randint(high=2, size=heads.size(), device=self.device)
        random_entities_head = torch.randint(high=ent_num, size=heads.size(), device=self.device)
        random_entities_tail = torch.randint(high=ent_num, size=heads.size(), device=self.device)
        for i in range(heads.size(0)):
            while random_entities_head[i] == heads[i]:
                random_entities_head[i] = torch.randint(high=ent_num, size=(1,), device=self.device)
            while random_entities_tail[i] == tails[i]:
                random_entities_tail[i] = torch.randint(high=ent_num, size=(1,), device=self.device)

        corrupted_heads = torch.where(head_or_tail == 1, random_entities_head, heads)
        corrupted_tails = torch.where(head_or_tail == 0, random_entities_tail, tails)
        return torch.stack([corrupted_heads, rels, corrupted_tails], dim=1)


class KRLDataset(Dataset):
    def __init__(self, triplets, entity2id, rel2id):
        super(KRLDataset, self).__init__()
        self.triples = []
        self.triplets = triplets
        self.entity2id = entity2id
        self.rel2id = rel2id

    def split_and_to_id(self, triplet) -> Tuple[int, int, int]:
        """
        :return: [head_id, rel_id, tail_id]
        """
        head, rel, tail = triplet
        head_id = self.entity2id[head]
        rel_id = self.rel2id[rel]
        tail_id = self.entity2id[tail]
        return (head_id, rel_id, tail_id)

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.triples)

    def read_triples(self):
        all_triples = []
        for lan, triplet_list in self.triplets.items():
            triples = [self.split_and_to_id(triplet) for triplet in triplet_list]
            all_triples.extend(triples)
        self.triples = all_triples
        triple = self.triples
        head_list = []
        rel_list = []
        tail_list = []

        for i in range(len(triple)):
            head_list.append(triple[i][0])
            rel_list.append(triple[i][1])
            tail_list.append(triple[i][2])

        head_tensor = torch.tensor(head_list)
        rel_tensor = torch.tensor(rel_list)
        tail_tensor = torch.tensor(tail_list)
        return head_tensor, rel_tensor, tail_tensor


class TransE(nn.Module):
    def __init__(
            self,
            ent_num,
            rel_num,
            device,
            norm,
            embed_dim,
            margin,
            entity2id,
            rel2id
    ):
        super(TransE, self).__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.norm = norm
        self.embed_dim = embed_dim
        self.margin = margin
        self.entity2id = entity2id
        self.rel2id = rel2id

        # Initialize ent_embedding
        self.ent_embedding = nn.Embedding(self.ent_num, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, 2, 1)

        # Initialize rel_embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, 2, 1)

        # loss function
        self.criterion = nn.MarginRankingLoss(margin=self.margin)

    def _distance(self, triples):
        """Calculate the distance of a batch's triplet

        :param triples: triples of a batch，size: [batch, 3]
        :return: size: [batch,]
        """
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]
        h_embs = self.ent_embedding(heads)  # h_embs: [batch, embed_dim]
        r_embs = self.rel_embedding(rels)
        t_embs = self.ent_embedding(tails)
        dist = h_embs + r_embs - t_embs  # [batch, embed_dim]
        return torch.norm(dist, p=self.norm, dim=1)

    def split_train_test(self, num, pos_heads, pos_tails, neg_heads, neg_tails):
        neg_num = num * 3
        train_pos_head_embeddings = self.ent_embedding(pos_heads[:-num])
        train_pos_tail_embeddings = self.ent_embedding(pos_tails[:-num])
        test_pos_head_embeddings = self.ent_embedding(pos_heads[-num:])
        test_pos_tail_embeddings = self.ent_embedding(pos_tails[-num:])
        train_neg_head_embeddings = self.ent_embedding(neg_heads[:-neg_num])
        train_neg_tail_embeddings = self.ent_embedding(neg_tails[:-neg_num])
        test_neg_head_embeddings = self.ent_embedding(neg_heads[-neg_num:])
        test_neg_tail_embeddings = self.ent_embedding(neg_tails[-neg_num:])
        train_label = []
        all_train_products = []
        for i in range(len(train_pos_head_embeddings)):
            pos_product = (train_pos_tail_embeddings[i] * train_pos_head_embeddings[i]).cpu().detach().numpy()
            all_train_products.append(pos_product)
            train_label.append(1)
            for j in range(3):  
                neg_product = (train_neg_tail_embeddings[i * 3 + j] * train_neg_head_embeddings[
                    i * 3 + j]).cpu().detach().numpy()
                all_train_products.append(neg_product)
                train_label.append(0)

        test_label = []
        all_test_products = []
        for i in range(num):
            pos_product = (test_pos_tail_embeddings[i] * test_pos_head_embeddings[i]).cpu().detach().numpy()
            all_test_products.append(pos_product)
            test_label.append(1)
            for j in range(3):  
                neg_product = (test_neg_tail_embeddings[i * 3 + j] * test_neg_head_embeddings[
                    i * 3 + j]).cpu().detach().numpy()
                all_test_products.append(neg_product)
                test_label.append(0)

        random_indices_test = random.sample(range(len(all_test_products)), len(all_test_products))
        shuffled_test_products = [all_test_products[i] for i in random_indices_test]
        shuffled_test_label = [test_label[i] for i in random_indices_test]
        all_test_products_np = np.array(shuffled_test_products)

        random_indices_train = random.sample(range(len(all_train_products)), len(all_train_products))
        shuffled_train_products = [all_train_products[i] for i in random_indices_train]
        shuffled_train_label = [train_label[i] for i in random_indices_train]
        all_train_products_np = np.array(shuffled_train_products)
        scaler = preprocessing.StandardScaler()
        all_train_products_np = scaler.fit_transform(all_train_products_np)
        all_test_products_np = scaler.fit_transform(all_test_products_np)

        return all_train_products_np, shuffled_train_label, all_test_products_np, shuffled_test_label

    def loss(self, pos_distances, neg_distances):
        """Calculate the loss of TransE training

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: loss
        """
        ones = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(pos_distances, neg_distances, ones)

    def forward(self, all_triplets):
        """Return model losses based on the input.

        :param all_triplets: the triplets from three modality
        :param pos_triples: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param neg_triples: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        data = KRLDataset(all_triplets, self.entity2id, self.rel2id)
        batch = data.read_triples()
        pos_heads, pos_rels, pos_tails = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        pos_triples = torch.stack([pos_heads, pos_rels, pos_tails], dim=1)
        train_neg_sampler = RandomNegativeSampler(data, self.device)
        neg_triples = train_neg_sampler.neg_sample(pos_heads, pos_rels, pos_tails)
        assert pos_triples.size()[1] == 3
        assert neg_triples.size()[1] == 3

        pos_distances = self._distance(pos_triples)
        neg_distances = self._distance(neg_triples)
        loss = self.loss(pos_distances, neg_distances)
        return loss

    def predict(self, test_triplets, gt_triplets):
        """sliding window(size = 2)"""
        macro_f1 = 0
        micro_f1 = 0
        acc = 0
        num = 0
        for i in range(len(test_triplets) - 2):
            num += 1
            selected_keys = [i, i+1, i+2]
            new_triplets = {}
            for key in selected_keys[:2]:
                new_triplets[key] = test_triplets[str(key)]
            new_triplets[selected_keys[2]] = gt_triplets[str(selected_keys[2])]

            pred_num = len(gt_triplets[str(i+2)]) 
            test_data = KRLDataset(new_triplets, self.entity2id, self.rel2id)
            batch = test_data.read_triples()
            pos_heads, pos_rels, pos_tails = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)

            adjacency_matrix = torch.zeros((self.ent_num, self.ent_num), dtype=torch.float32)
            for n in range(len(pos_heads)):
                head_idx = pos_heads[n].item()
                tail_idx = pos_tails[n].item()
                adjacency_matrix[head_idx][tail_idx] = 1  
            neg_nums = len(pos_heads) * 3
            rows, cols = torch.where(adjacency_matrix == 0)
            neg_heads = rows[:neg_nums].to(self.device)
            neg_tails = cols[:neg_nums].to(self.device)
            train_X, train_y, test_X, test_y = self.split_train_test(pred_num, pos_heads, pos_tails, neg_heads, neg_tails)
            bi_model = LogisticRegression(max_iter=1000)
            bi_model.fit(train_X, train_y)
            pred_y = bi_model.predict(test_X)
            micro_f1 += f1_score(test_y, pred_y, average='micro')
            macro_f1 += f1_score(test_y, pred_y, average='macro')
            acc += roc_auc_score(test_y, pred_y)
        acc /= num
        micro_f1 /= num
        macro_f1 /= num
        return acc, micro_f1, macro_f1
