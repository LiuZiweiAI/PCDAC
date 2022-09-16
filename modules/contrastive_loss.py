import torch
import torch.nn as nn
import math

# Instance Loss
class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class weekLoss(nn.Module):
    def __init__(self, class_num, Truncated,batch_size, temperature, device, queue_batch, alpha):
        super(weekLoss, self).__init__()
        self.class_num = class_num
        self.batch_size = batch_size
        self.Truncated = Truncated
        self.queue_size = self.batch_size*5

        self.temperature = temperature
        self.device = device
        self.queue_batch = queue_batch
        self.alpha = alpha
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)


    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask


    def forward(self, epoch, step,z_w, c_i, c_j, c_w, queue_feats, queue_probs, queue_ptr):
        p_i = c_i.sum(0).view(-1)  # tensor.sum(0/1)：0：对每一列进行累加；1：对每一行进行累加。两者最终都为包含一个tensor的tensor：例：tensor([6, 6, 8])
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()  # tensor.size(0)：指tensor的第一维，通常指行

        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        ne_loss = ne_i + ne_j # - H(Y)
        with torch.no_grad():
            # DA
            list_w =[]
            list_w.append(c_w.mean(0))
            prob_avg = torch.stack(list_w, dim=0).mean(0)
            w_mean = c_w / prob_avg
            d_i = c_i*w_mean
            d_i = d_i / d_i.sum(dim=1, keepdim=True)
            d_j = c_j * w_mean
            d_j = d_j / d_j.sum(dim=1, keepdim=True)
            probs = w_mean / w_mean.sum(dim=1, keepdim=True)


            if epoch>0 or step>self.queue_batch: # memory-smoothing
                A = torch.exp(torch.mm(z_w, queue_feats.t())/self.temperature)
                A = A/A.sum(1,keepdim=True)
                probs = self.alpha*probs + (1-self.alpha)*torch.mm(A, queue_probs)

            n = c_w.size(0)
            queue_feats[queue_ptr:queue_ptr + n, :] = z_w
            queue_probs[queue_ptr:queue_ptr + n, :] = probs
            queue_ptr = (queue_ptr + n) % self.queue_size

        scores, lbs_u_guess = torch.max(probs, dim=1)
        mask_p = scores.ge(self.Truncated).float()
        loss_u = - torch.sum((torch.log(c_w) * probs), dim=1)*mask_p
        loss_u = loss_u.mean()
        print('loss_u:', loss_u*10)

        c_i = c_i.t()
        c_j = c_j.t()
        d_i = d_i.t()
        d_j = d_j.t()

        N = 2 * self.class_num
        c1 = torch.cat((c_i, d_j), dim=0)
        c2 = torch.cat((c_j, d_i), dim=0)

        sim1 = self.similarity_f(c1.unsqueeze(1), c1.unsqueeze(0)) / self.temperature
        sim2 = self.similarity_f(c2.unsqueeze(1), c2.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim1, self.class_num)
        sim_j_i = torch.diag(sim1, -self.class_num)
        sim2_i_j = torch.diag(sim2, self.class_num)
        sim2_j_i = torch.diag(sim2, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim1[self.mask].reshape(N, -1)

        positive_clusters2 = torch.cat((sim2_i_j, sim2_j_i), dim=0).reshape(N, 1)
        negative_clusters2 = sim2[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)

        labels2 = torch.zeros(N).to(positive_clusters2.device).long()
        logits2 = torch.cat((positive_clusters2, negative_clusters2), dim=1)

        loss_cluster = self.criterion(logits, labels) + self.criterion(logits2, labels2)
        loss_cluster /= 2*N

        loss = loss_cluster + ne_loss +loss_u*10 

        return loss, queue_feats, queue_probs, queue_ptr
