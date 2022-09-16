import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
import time

def train(epoch, center, queue_feats, queue_probs, queue_ptr,selected_label,classwise_acc,label_queue_size,select_ptr):
    loss_epoch = 0
    start = time.time()
    for step, ((x_i, x_j, x_w), _) in enumerate(cluster_data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda:0')
        x_j = x_j.to('cuda:0')
        x_w = x_w.to('cuda:0')
        z_i, z_j, z_w, c_i, c_j, c_w = model(x_i, x_j, x_w)
        loss_instance = criterion_instance(z_i, z_j)
        loss_week, queue_feats, queue_probs, queue_ptr, select_ptr, selected_label = criterion_week(epoch, center, step, z_i,
                                                                                               z_j, z_w, c_i, c_j, c_w,
                                                                                               queue_feats, queue_probs,
                                                                                               queue_ptr,
                                                                                               selected_label,
                                                                                               classwise_acc,
                                                                                               label_queue_size,
                                                                                               select_ptr)

        loss = loss_instance + loss_week
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(cluster_data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_week: {loss_week.item()}")
        loss_epoch += loss.item()
    end = time.time()
    print('迭代一轮所需时间：{}'.format(end - start))
    return loss_epoch, queue_feats, queue_probs, queue_ptr ,select_ptr,selected_label   # 返回一个epoch后所得的loss。

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        unlabeled_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=transform.Transforms(size=args.image_size),
        )
        cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        instance_dataset = unlabeled_dataset
        class_num = 10
    else:
        raise NotImplementedError
    cluster_data_loader = torch.utils.data.DataLoader(
        cluster_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    instance_data_loader = torch.utils.data.DataLoader(
        instance_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    # loss
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_week = contrastive_loss.weekLoss(class_num, args.Truncated, args.batch_size, args.cluster_temperature,
                                               loss_device, args.queue_batch, args.alpha).to(loss_device)

    # memory bank
    args.queue_size = args.queue_batch * args.batch_size
    queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
    queue_probs = torch.zeros(args.queue_size, class_num).cuda()
    queue_ptr = 0

    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, queue_feats, queue_probs, queue_ptr = train(epoch, queue_feats=queue_feats, queue_probs=queue_probs, queue_ptr=queue_ptr)

        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(cluster_data_loader)}")
    save_model(args, model, optimizer, args.epochs)
