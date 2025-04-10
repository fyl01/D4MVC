# coding=gbk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import numpy as np
import random
import argparse
from load_data import DatasetSplit, load_data
import copy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from Nmetrics import cluster_metrics, purity, calculate_entropy, calculate_variance, calculate_gini
from scipy.optimize import linear_sum_assignment
from Network import AutoEncoder
import warnings
import time
warnings.filterwarnings("ignore")

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='BDGP', help="Dataset name to train on")
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--Dirichlet_alpha", type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")

    # Parameters for training
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--mse_epochs", default=500, help="Number of epochs for pretraining")
    parser.add_argument("--global_epochs", default=10000)
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate during clustering")
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--feature_dim', default=10, type=int, help="encode/decode dimension")
    parser.add_argument("--alpha", default=0.1)
    parser.add_argument("--interval_epoch", type=int, default=500)
    parser.add_argument("--b", type=float, default=0.9)
    parser.add_argument("--w", type=float, default=0.3)
    args = parser.parse_args()
    return args


def setup_seed():
    seed = 43
    if args.dataset == "BDGP":
        seed = 10
    if args.dataset == "Caltech-5V":
        seed = 50
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrain(ae, vi):
    ae.train()
    num_epochs = args.mse_epochs
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=args.lr)

    for epoch in range(num_epochs):
        for batch_idx, (xs, ys) in enumerate(data_loader_list[vi]):
            xs = xs.to(device)
            optimizer.zero_grad()
            zs, dec_enf, q = ae(xs)
            mseloss = criterion(xs, dec_enf)
            mseloss.backward()
            optimizer.step()

    # kmeans
    zs_list = []
    ys_list = []
    for batch_idx, (xs, ys) in enumerate(data_loader_list[vi]):
        xs = xs.to(device)
        zs, xrs, qs = ae(xs)
        zs_list.append(zs)
        ys_list.append(ys)
    zs_list = torch.cat(zs_list, dim=0)
    ys_list = torch.cat(ys_list, dim=0)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit(zs_list.detach().cpu().numpy())
    labels = kmeans.predict(zs_list.detach().cpu().numpy())
    print("Client %d, view %d" %(vi%args.num_users, vi//args.num_users+1), "single-view kmeans result(acc):", acc(labels, ys_list.detach().cpu().numpy()))
    cluster_centers = kmeans.cluster_centers_
    ae.centroids.data = copy.deepcopy(torch.tensor(cluster_centers)).to(device)
    # print(ae.centroids.data)     # shape: [k * encode_dim]


def localvalid(valid_model_list, valid_dataset_list):
    pf_list = []
    local_zs, local_ys, local_qs = [], [], []
    local_accs = []
    local_nmis = []
    local_aris = []
    for an in range(view * args.num_users):
        zs_list, ys_list, qs_list = [], [], []
        valid_model_list[an].eval()
        for batch_idx, (xs, ys) in enumerate(valid_dataset_list[an]):
            xs = xs.to(device)
            zs, xrs, qs = valid_model_list[an](xs)
            zs_list.append(zs)
            ys_list.append(ys)
            qs_list.append(qs)

        local_zs.append(torch.cat(zs_list, dim=0))
        local_ys.append(torch.cat(ys_list, dim=0))
        local_qs.append(torch.cat(qs_list, dim=0))

    # match
    for v in range(1, view):
        for i in range(args.num_users):
            idx = match(((local_qs[i].argmax(dim=1).detach().cpu().numpy())),
                        np.array((local_qs[v * args.num_users + i]).argmax(dim=1).detach().cpu().numpy()))  # .argmax(dim=1)
            local_qs[v * args.num_users + i] = local_qs[v * args.num_users + i][:, idx]


    for u in range(args.num_users):
        for v1 in range(1, view):
            local_zs[u] = torch.cat((local_zs[u], local_zs[u + v1 * args.num_users]), dim=1)
            local_qs[u] = (local_qs[u] + local_qs[u + v1 * args.num_users])

        local_pred = np.argmax(np.array(local_qs[u].cpu().detach().numpy()), axis=1)
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        kmeans.fit(local_zs[u].detach().cpu().numpy())
        labels = kmeans.predict(local_zs[u].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        p_f, _ = make_ps(local_zs[u], torch.tensor(cluster_centers).to(device))
        pf_list.append(p_f)
        test_acc = acc(local_ys[u].detach().numpy(), local_pred)
        test_nmi = normalized_mutual_info_score(local_ys[u].detach().numpy().squeeze(), local_pred)
        test_ari = adjusted_rand_score(local_ys[u].detach().numpy().squeeze(), local_pred)

        local_accs.append(test_acc)
        local_nmis.append(test_nmi)
        local_aris.append(test_ari)

    print('*' * 50)
    print('local acc', local_accs)
    print('local nmi', local_nmis)
    print('local ari', local_aris)
    total_acc = sum(x*y for x,y in zip(local_accs,samplenum))/sum(samplenum)
    total_nmi = sum(x*y for x,y in zip(local_nmis,samplenum))/sum(samplenum)
    total_ari = sum(abs(x)*y for x,y in zip(local_aris,samplenum))/sum(samplenum)
    print('total acc', total_acc)
    print('total nmi', total_nmi)
    print('total ari', total_ari)
    return pf_list


def match(y_pred, y_true):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind[1]


def ineqmatch(labels1, labels2):
    confusion_matrix = np.zeros((class_num, class_num), dtype=int)
    for l1 in labels1:
        for l2 in labels2:
            confusion_matrix[l1, l2] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    return col_ind


def make_ps(x, centroids):
    s = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - centroids, 2), 2))
    s = (s.t() / torch.sum(s, 1)).t()
    p = s ** 2 / s.sum(0)
    return (p.t() / p.sum(1)).t(), s

def acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        if args.dataset == 'BDGP' or args.dataset == 'Caltech-5V':
            w[y_pred[i], y_true[i]] += 1
        else:
            w[y_pred[i]-1, y_true[i]-1] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size

# only noniid clients
def globalvalid(valid_model_list, valid_dataset_list, noniidclient):
    local_zs, local_ys, local_qs = [], [], []
    for vi in range(view * args.num_users):
        zs_list, ys_list, qs_list = [], [], []
        valid_model_list[vi].eval()
        for batch_idx, (xs, ys) in enumerate(valid_dataset_list[vi]):
            xs = xs.to(device)
            zs, dec_enf, qs = valid_model_list[vi](xs)
            zs_list.append(zs)
            ys_list.append(ys)
            qs_list.append(qs)

        local_zs.append(torch.cat(zs_list, dim=0))
        local_ys.append(torch.cat(ys_list, dim=0))
        local_qs.append(torch.cat(qs_list, dim=0))

    for v in range(1, view):
        for i in range(args.num_users):
            idx = match((local_qs[i].argmax(dim=1).detach().cpu().numpy()),
                        (local_qs[v * args.num_users + i]).argmax(dim=1).detach().cpu().numpy())
            local_qs[v * args.num_users + i] = local_qs[v * args.num_users + i][:, idx]


    for u in range(args.num_users):
        for v1 in range(1, view):
            local_zs[u] = torch.cat((local_zs[u], local_zs[u + v1 * args.num_users]), dim=1)
            local_qs[u] = (local_qs[u] + local_qs[u + v1 * args.num_users])

    for i in range(1, args.num_users):
        idx = ineqmatch(local_qs[0].argmax(dim=1).detach().cpu().numpy(), local_qs[i].argmax(dim=1).detach().cpu().numpy())
        local_qs[i] = local_qs[i][:, idx]

    global_zs = torch.cat(local_zs[0:args.num_users], dim=0)
    global_ys = torch.cat(local_ys[0:args.num_users], dim=0)
    global_qs = torch.cat(local_qs[0:args.num_users], dim=0)

    total_pred = np.argmax(np.array(global_qs.cpu().detach().numpy()), axis=1)
    partial_zs, partial_ys, partial_qs = [], [], []
    for v in range(view):
        for u in noniidclient:
            partial_zs.append(local_zs[u + v * args.num_users])
            partial_qs.append(local_qs[u + v * args.num_users])
    partial_zs_concat = torch.cat(partial_zs[0:len(noniidclient)], dim=0)
    partial_qs_concat = torch.cat(partial_qs[0:len(noniidclient)], dim=0)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit(partial_zs_concat.detach().cpu().numpy())    # noniid samples
    cluster_centers = kmeans.cluster_centers_
    p_f, _ = make_ps(global_zs, torch.tensor(cluster_centers).to(device))
    return p_f


def formaltrain(model, vi, global_p, local_p, b):
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    c = 0
    for batch_idx, (xs, ys) in enumerate(data_loader_list[vi]):
        xs = xs.to(device)
        optimizer.zero_grad()
        _, xrs, q = model(xs)
        mseloss = criterion(xs, xrs)

        qs = q.shape[0]  # qs = batch_size
        gp_slice = global_p[c * qs: (c + 1) * qs, :].clone().detach()
        lp_slice = local_p[c * qs: (c + 1) * qs, :].clone().detach()
        global_kl = F.kl_div(F.log_softmax(q, dim=1), F.softmax(gp_slice, dim=1), reduction='batchmean')
        local_kl = F.kl_div(F.log_softmax(q, dim=1), F.softmax(lp_slice, dim=1), reduction='batchmean')
        c += 1
        loss = mseloss + args.alpha * (b*global_kl+(1-b)*local_kl)
        loss.backward()
        optimizer.step()


def noniid_user(users_model, r_num, local_labels, local_emb):
    distance_disp_list, compact_list, sep_list, dens_list = [], [], [], []
    for u in range(args.num_users):
        centers = users_model[u].concat_centr.data.cpu().detach()

        inter_cluster_distance_disparity, compactness, separation, density = cluster_metrics(local_emb[u].cpu().detach(), class_num, local_labels[u], centers)
        distance_disp_list.append(inter_cluster_distance_disparity)
        compact_list.append(compactness)
        sep_list.append(separation)
        dens_list.append(density)
    print("compactness", compact_list)

    metric2 = sorted(range(args.num_users), key=lambda i: compact_list[i])[:r_num]
    return metric2


def local_concat(ae_list):
    local_zs, local_ys = [], []
    for v in range(view):
        for i in range(args.num_users):
            zs_list = []
            ys_list = []
            ae_list[v*args.num_users+i].eval()
            for batch_idx, (xs, ys) in enumerate(data_loader_list[v * args.num_users + i]):
                xs = xs.to(device)
                zs, dec_enf, qs = ae_list[v*args.num_users+i](xs)
                zs_list.append(zs)
                ys_list.append(ys)
            local_zs.append(torch.cat(zs_list, dim=0))
            local_ys.append(torch.cat(ys_list, dim=0))
    for u in range(args.num_users):
        for v1 in range(1, view):
            local_zs[u] = torch.cat((local_zs[u], local_zs[u + v1 * args.num_users]), dim=1)
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        kmeans.fit(local_zs[u].detach().cpu().numpy())
        labels = kmeans.predict(local_zs[u].detach().cpu().numpy())
        local_labels[u] = labels
        cluster_centers = kmeans.cluster_centers_
        ae_list[u].concat_centr.data = copy.deepcopy(torch.tensor(cluster_centers)).to(device)

    return local_zs[:args.num_users]


if __name__ == '__main__':
    '''
    Dataset settings
     'BDGP'           # 2 views,  5 clusters, 2500 examples
     'Wiki'           # 2 views, 10 clusters, 2866 examples 
     'Cora'           # 4 views,  7 clusters, 2708 examples
     'Caltech-5V'     # 5 views,  7 clusters, 1400 examples
     'STL10'          # 3 views, 10 clusters, 13000 examples
     'CCV'            # 3 views, 20 clusters, 6773 examples
    '''
    args = args_parser()
    print('-' * 30, ' Parameters ', '-' * 30)
    print(args)
    print('-' * 75)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    dataset, dims, view, data_size, class_num = load_data(args.dataset, args.num_users, args.Dirichlet_alpha, args.w)

    samplenum = []
    for i in range(args.num_users):
        samplenum.append(len(dataset.user_data[i]))
    print("Sample number of each client",samplenum)

    data_loader_list = []
    for v in range(1, view + 1):
        for i in range(args.num_users):
            data_loader = DataLoader(
                DatasetSplit(getattr(dataset, 'x' + str(v)), dataset.y, dataset.user_data[i], dims[v - 1]),
                batch_size=args.batch_size, shuffle=False, drop_last=False)
            data_loader_list.append(copy.deepcopy(data_loader))
    setup_seed()
    T = 5
    for i in range(T):

        accs = []
        nmis = []
        aris = []

        ae_list = []
        ae_noise = []
        local_labels = {i: [] for i in range(args.num_users)}
        local_emb = []

        for v in range(view):
            for _ in range(args.num_users):
                ae_list.append(copy.deepcopy(AutoEncoder(class_num, args.feature_dim, dims[v]).to(device)))

        for vi in range(view * args.num_users):
            pretrain(ae_list[vi], vi)

        r_num = args.num_users - int(args.w * args.num_users)
        local_emb = local_concat(ae_list)
        noniidclient = noniid_user(ae_list, r_num, local_labels, local_emb)
        iidclient = np.setdiff1d(range(args.num_users), noniidclient)
        print(noniidclient, iidclient)

        b = args.b
        setup_seed()
        for me in range(args.global_epochs):
            if me % args.interval_epoch == 0:
                local_p = localvalid(ae_list, data_loader_list)
                global_p = globalvalid(ae_list, data_loader_list, noniidclient)
                for i in range(args.num_users):
                    idx = ineqmatch(global_p.argmax(dim=1).detach().cpu().numpy(), local_p[i].argmax(dim=1).detach().cpu().numpy())
                    local_p[i] = local_p[i][:, idx]
                local_p = torch.cat(local_p, dim=0)

            x_ind = 0
            for vi in range(view*args.num_users):
                y_ind = x_ind + len(dataset.user_data[vi % args.num_users])
                if vi % args.num_users in noniidclient:
                    formaltrain(ae_list[vi], vi, global_p[x_ind: y_ind], local_p[x_ind: y_ind], b)
                else:
                    formaltrain(ae_list[vi], vi, global_p[x_ind: y_ind], local_p[x_ind: y_ind], 1-b)
                x_ind = y_ind
                if (vi + 1) % args.num_users == 0:
                    x_ind = 0


