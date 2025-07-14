import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import numpy as np

import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(AlignmentNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class FeedForward(nn.Module):
    def __init__(self, n_inp, n_out, activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        y = self.linear(x)
        y = self.activation(y)

        if indices is not None:
            if not torch.is_tensor(indices):
                indices = torch.tensor(indices, dtype=torch.long, device=y.device)
            if indices.numel() > 0:
                assert indices.max() < y.shape[-1], f"indices max {indices.max()} >= output dim {y.shape[-1]}"
                assert indices.min() >= 0, f"indices contains negative values"
                y = y[..., indices]
            else:
                raise ValueError("FeedForward received empty indices — nothing to predict.")

        if self.residual and x.size(-1) == y.size(-1):
            y = y + x

        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, adjacency):
        vsum = torch.sparse.mm(adjacency, emb)
        row_sum = torch.sparse.sum(adjacency, dim=1).to_dense().view(-1, 1)  # 每行和
        row_sum[row_sum == 0] = 1
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.f_k.weight.data)
        if self.f_k.bias is not None:
            self.f_k.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), dim=1)
        return logits


class GNNRepresentationGraphST(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNRepresentationGraphST, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.readout = AvgReadout()
        self.discriminator = Discriminator(hidden_dim)
        self.sigm = nn.Sigmoid()
        self.loss_CSL = nn.BCEWithLogitsLoss()

    @staticmethod
    def add_contrastive_mask(batch_size, device='cuda'):
        return torch.eye(batch_size, dtype=torch.float32, device=device)  # 对角线为正样本

    @staticmethod
    def add_contrastive_label(n_spot, device='cuda'):
        if n_spot == 0:
            raise ValueError("n_spot is 0 — contrastive label cannot be constructed.")

        # Safer: use torch instead of numpy
        one = torch.ones(n_spot, 1, dtype=torch.float32)
        zero = torch.zeros(n_spot, 1, dtype=torch.float32)
        label_CSL = torch.cat([one, zero], dim=1).to(device)
        return label_CSL

    def edge_index_to_adjacency(self, edge_index, num_nodes):

        row, col = edge_index
        values = torch.ones(len(row), device=edge_index.device)
        adjacency = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=values,
            size=(num_nodes, num_nodes)
        )
        return adjacency

    def construct_graphst_adjacency(self, locs, k=5, device='cuda'):

        locs = locs.cpu().numpy()
        n_neighbors = min(k, locs.shape[0])
        # print("n_neighbors:", n_neighbors)  # Debugging output
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(locs)
        _, indices = nbrs.kneighbors(locs)
        row = np.repeat(np.arange(len(locs)), n_neighbors)
        col = indices.flatten()
        edge_index = torch.tensor([row, col], dtype=torch.long, device=device)
        return edge_index

    def permutation(self, x, seed=42):

        np.random.seed(seed)
        ids = np.arange(x.shape[0])
        ids = np.random.permutation(ids)
        feature_permutated = x[ids]
        return feature_permutated

    def compute_loss(self, ret, ret_a, label, x, h):
        loss_sl_1 = self.loss_CSL(ret, label)
        loss_sl_2 = self.loss_CSL(ret_a, label)

        loss_feat = F.mse_loss(x, h)

        total_loss = 0.1*(loss_sl_1 + loss_sl_2) + 1*loss_feat
        return total_loss

    def forward(self, x, edge_index):

        num_nodes = x.shape[0]
        adjacency = self.edge_index_to_adjacency(edge_index, num_nodes).to(x.device)

        z = F.relu(self.gcn1(x, edge_index))
        hidden_emb = z

        h = F.relu(self.gcn2(z, edge_index))

        x_a = self.permutation(x)
        x_a = x_a.to(x.device)
        z_a = F.relu(self.gcn1(x_a, edge_index))
        emb_a = z_a

        g = self.readout(z, adjacency)
        g = self.sigm(g)
        g_a = self.readout(emb_a, adjacency)
        g_a = self.sigm(g_a)

        ret = self.discriminator(g, hidden_emb, emb_a)
        ret_a = self.discriminator(g_a, emb_a, hidden_emb)

        return hidden_emb, h, ret, ret_a

class AttentionFusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionFusionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, rep_x, rep_z):
        if rep_x.dim() == 4:
            batch_size, h, w, channels = rep_x.shape
            rep_x = rep_x.view(batch_size, h * w, channels)

        if rep_z.dim() == 4:
            batch_size, h, channels = rep_z.shape
            rep_z = rep_z.view(batch_size, h, channels)
        attn_output, _ = self.multihead_attn(query=rep_x, key=rep_z, value=rep_z)
        fused_rep = attn_output + rep_x
        fused_rep = self.layer_norm(fused_rep)
        fused_rep = F.relu(self.fc(fused_rep))

        return fused_rep


class GenePredictor(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super(GenePredictor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(256, 256),
            ELU(alpha=0.01, beta=0.01),
            nn.Linear(256, 256),
            ELU(alpha=0.01, beta=0.01),
            nn.Linear(256, 256)
        )
        self.layer2 = FeedForward(256, src_dim, activation=ELU(alpha=0.01, beta=0.01), residual=False)

        if src_dim != tgt_dim:
            self.source_to_target_mapping = nn.Linear(src_dim, tgt_dim)
        else:
            self.source_to_target_mapping = None

    def forward(self, x, indices=None, to_target_dim=False):
        residual = x
        x = self.layer1(x) + residual
        x = self.layer2(x, indices=indices)
        if to_target_dim and self.source_to_target_mapping is not None:
            x = self.source_to_target_mapping(x)
        return x


class VISD(pl.LightningModule):
    def __init__(self, lr, src_n_inp, tgt_n_inp, src_n_out, tgt_n_out, src_gnn_input_dim, tgt_gnn_input_dim, gnn_hidden_dim, gnn_output_dim, weight_decay=1e-5):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.src_gnn = GNNRepresentationGraphST(src_gnn_input_dim, gnn_hidden_dim, src_gnn_input_dim)
        self.tgt_gnn = GNNRepresentationGraphST(tgt_gnn_input_dim, gnn_hidden_dim, tgt_gnn_input_dim)
        self.src_feature_extraction = nn.Sequential(
            FeedForward(src_n_inp, 256),
            FeedForward(256, 256, residual=True),
            FeedForward(256, 256, residual=True),
            FeedForward(256, 256, residual=True))

        self.tgt_feature_extraction = nn.Sequential(
            FeedForward(tgt_n_inp, 256),
            FeedForward(256, 256, residual=True),
            FeedForward(256, 256, residual=True),
            FeedForward(256, 256, residual=True))
        self.gene_predict = GenePredictor(src_n_out, tgt_n_out)

        self.fusion_layer = AttentionFusionLayer(embed_dim=256, num_heads=4)
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.alignment_network = AlignmentNetwork(input_dim=256, hidden_dim=512)

    def src_forward_feature_extraction(self, x, edge_index, y):
        z, h, ret, ret_a = self.src_gnn(y, edge_index)

        loss_scl = self.src_gnn.compute_loss(ret, ret_a, self.src_gnn.add_contrastive_label(y.size(0), device='cuda'), y, h)
        rep_x = self.src_feature_extraction(x)
        rep_z = z.unsqueeze(1).expand(-1, x.size(1), -1)
        if rep_z.shape[0] < x.shape[0]:
            rep_z = rep_z.repeat((x.shape[0] // rep_z.shape[0] + 1), 1, 1)
            rep_z = rep_z[:x.shape[0]]
        elif rep_z.shape[0] > x.shape[0]:
            rep_z = rep_z[:x.shape[0]]

        rep = self.fusion_layer(rep_x, rep_z)
        return rep, loss_scl

    def tgt_forward_feature_extraction(self, x, edge_index, y):
        z, h, ret, ret_a = self.tgt_gnn(y, edge_index)

        loss_scl = self.tgt_gnn.compute_loss(ret, ret_a, self.src_gnn.add_contrastive_label(y.size(0), device='cuda'), y, h)

        rep_x = self.tgt_feature_extraction(x)
        rep_z = z.unsqueeze(1).expand(-1, x.size(1), -1)
        if rep_z.shape[0] < x.shape[0]:
            rep_z = rep_z.repeat((x.shape[0] // rep_z.shape[0] + 1), 1, 1)
            rep_z = rep_z[:x.shape[0]]
        elif rep_z.shape[0] > x.shape[0]:
            rep_z = rep_z[:x.shape[0]]

        rep = self.fusion_layer(rep_x, rep_z)
        return rep, loss_scl

    def training_step(self, batch, batch_idx):
        (x_source, y_source, label_source, locs_source), (x_target, y_target, locs_target) = batch
        device = x_source.device
        edge_index_source = self.src_build_adj_matrix(locs_source, device=device)
        edge_index_target = self.tgt_build_adj_matrix(locs_target, device=device)

        pretrain_feature_epochs = int(0.3 * self.trainer.max_epochs)
        pretrain_epochs = int(0.5 * self.trainer.max_epochs)

        if self.current_epoch < pretrain_feature_epochs:
            rep_source, loss_scl_source = self.src_forward_feature_extraction(x_source, edge_index_source, y_source)
            optimizer_src = self.optimizers()[0]
            optimizer_src.zero_grad()
            self.manual_backward(loss_scl_source)
            optimizer_src.step()

            rep_target, loss_scl_target = self.tgt_forward_feature_extraction(x_target, edge_index_target, y_target)
            optimizer_tgt = self.optimizers()[1]
            optimizer_tgt.zero_grad()
            self.manual_backward(loss_scl_target)
            optimizer_tgt.step()

            self.log('src_loss', loss_scl_source, prog_bar=True, on_step=False, on_epoch=True)
            self.log('tgt_loss', loss_scl_target, prog_bar=True, on_step=False, on_epoch=True)

            return loss_scl_source + loss_scl_target

        if pretrain_feature_epochs <= self.current_epoch < pretrain_epochs:
            rep_source, _ = self.src_forward_feature_extraction(x_source, edge_index_source, y_source)
            y_pred_source = self.gene_predict(rep_source)
            loss_mse = F.mse_loss(y_pred_source, label_source)
            for param in self.src_gnn.parameters():
                param.requires_grad = False
            for param in self.tgt_gnn.parameters():
                param.requires_grad = False
            for param in self.gene_predict.parameters():
                param.requires_grad = True
            optimizer_pretrain = self.optimizers()[2]
            optimizer_pretrain.zero_grad()
            self.manual_backward(loss_mse)
            optimizer_pretrain.step()

            self.log('pretrain_mse_loss', loss_mse, prog_bar=True, on_step=False, on_epoch=True)
            return loss_mse

        rep_source, _ = self.src_forward_feature_extraction(x_source, edge_index_source, y_source)
        rep_target, _ = self.tgt_forward_feature_extraction(x_target, edge_index_target, y_target)
        aligned_target, decoded_target = self.alignment_network(rep_target)
        mnn_pairs = self.compute_mnn_pairs_feature_dim(rep_source, aligned_target, seq_len=rep_source.size(1),
                                                       max_pairs=500)
        anchors, positives, negatives = [], [], []
        for target_batch, target_seq, source_batch, source_seq in mnn_pairs:
            anchors.append(aligned_target[target_batch, target_seq])
            positives.append(rep_source[source_batch, source_seq])
            negative_pool = rep_source[source_batch]
            neg_sim = F.cosine_similarity(aligned_target[target_batch, target_seq].unsqueeze(0), negative_pool, dim=-1)
            hardest_negative_idx = torch.argmin(neg_sim)
            negatives.append(negative_pool[hardest_negative_idx])
        if len(anchors) == 0:
            loss_triplet = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            anchor = torch.stack(anchors)
            positive = torch.stack(positives)
            negative = torch.stack(negatives)
            loss_triplet = triplet_loss_cosine(anchor, positive, negative)
        to_target_dim = y_source.shape[-1] != y_target.shape[-1]
        y_pred_target = self.gene_predict(aligned_target, to_target_dim=to_target_dim)
        loss_target_mse = F.mse_loss(y_pred_target.mean(-2), y_target)

        triplet_loss_weight = 1.0
        target_mse_loss_weight = 1.0
        loss_total = (triplet_loss_weight * loss_triplet + target_mse_loss_weight * loss_target_mse)
        optimizer_alignment = self.optimizers()[3]
        optimizer_alignment.zero_grad()
        self.manual_backward(loss_total)
        optimizer_alignment.step()

        self.log('triplet_loss', loss_triplet, prog_bar=True, on_step=False, on_epoch=True)
        self.log('target_mse_loss', loss_target_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('total_loss', loss_total, prog_bar=True, on_step=False, on_epoch=True)

        return loss_total

    def compute_mnn_pairs_feature_dim(self, rep_source, rep_target, seq_len, k=5, max_pairs=500):
        batch_size, seq_len, feature_dim = rep_source.shape
        rep_source_reshaped = rep_source.view(-1, feature_dim)
        rep_target_reshaped = rep_target.view(-1, feature_dim)

        distances = torch.cdist(rep_target_reshaped, rep_source_reshaped)
        topk_target_to_source = torch.topk(-distances, k, dim=1, largest=False).indices
        topk_source_to_target = torch.topk(-distances.t(), k, dim=1, largest=False).indices

        target_indices = torch.arange(distances.size(0), device=distances.device).unsqueeze(1).expand_as(
            topk_target_to_source)
        source_indices = topk_target_to_source

        mask = (target_indices.unsqueeze(2) == topk_source_to_target.t().unsqueeze(0)).any(dim=2)
        mnn_target_idx, mnn_source_idx = torch.where(mask)

        target_batch = mnn_target_idx // seq_len
        target_seq = mnn_target_idx % seq_len
        source_batch = mnn_source_idx // seq_len
        source_seq = mnn_source_idx % seq_len

        mnn_pairs = list(zip(target_batch.tolist(), target_seq.tolist(), source_batch.tolist(), source_seq.tolist()))

        if len(mnn_pairs) > max_pairs:
            indices = torch.randperm(len(mnn_pairs))[:max_pairs]
            mnn_pairs = [mnn_pairs[idx] for idx in indices]

        return mnn_pairs

    def src_build_adj_matrix(self, locs, k=5, device='cuda'):
        return self.src_gnn.construct_graphst_adjacency(locs, k, device)

    def tgt_build_adj_matrix(self, locs, k=5, device='cuda'):
        return self.tgt_gnn.construct_graphst_adjacency(locs, k, device)

    def predict_gene_expression(self, x_test, y_test, locs_test, indices=None):
        device = x_test.device
        if y_test.numel() > 0 and locs_test.numel() > 0:
            edge_index_test = self.tgt_build_adj_matrix(locs_test, device=device)
            rep_test, _ = self.tgt_forward_feature_extraction(x_test, edge_index_test, y_test)
        else:
            rep_test = self.tgt_feature_extraction(x_test)
        aligned_test, _ = self.alignment_network(rep_test)
        y_pred_test = self.gene_predict(aligned_test, indices)

        return y_pred_test


    def configure_optimizers(self):
        optimizer_src_feature_extraction = Adam(
            list(self.src_gnn.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        optimizer_tgt_feature_extraction = Adam(
            list(self.tgt_gnn.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )

        optimizer_pretrain = Adam(
            list(self.gene_predict.parameters()),  
            lr=self.lr, weight_decay=self.weight_decay
        )

        optimizer_alignment = Adam(
            list(self.alignment_network.parameters()) +
            list(self.tgt_feature_extraction.parameters()) +
            list(self.gene_predict.layer2.parameters()) +
            (list(self.gene_predict.source_to_target_mapping.parameters())
             if self.gene_predict.source_to_target_mapping is not None else []),
            lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler_src_feature = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_src_feature_extraction, mode='min', factor=0.5, patience=5, verbose=True
        )
        scheduler_tgt_feature = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_tgt_feature_extraction, mode='min', factor=0.5, patience=5, verbose=True
        )
        scheduler_pretrain = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_pretrain, mode='min', factor=0.5, patience=5, verbose=True
        )
        scheduler_alignment = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_alignment, mode='min', factor=0.5, patience=5, verbose=True
        )

        return [
            optimizer_src_feature_extraction,  
            optimizer_tgt_feature_extraction,  
            optimizer_pretrain,  
            optimizer_alignment  
        ], [
            scheduler_src_feature,  
            scheduler_tgt_feature,  
            scheduler_pretrain,  
            scheduler_alignment  
        ]


def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def triplet_loss_cosine(anchor, positive, negative, margin=0.5):
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = F.relu(margin - pos_sim + neg_sim)
    return loss.mean()


def check_gradients(model, model_name):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Gradient not updated for {model_name} parameter: {name}")
        elif torch.all(param.grad == 0):
            print(f"Gradient is zero for {model_name} parameter: {name}")
        else:
            print(f"Gradient updated for {model_name} parameter: {name}")