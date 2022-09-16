import os
import numpy as np
import torch
import torchvision
import argparse
import time
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
#os.environ['CUDA_VISIBLE_DEVICES'] ='1'

def train(epoch, queue_feats, queue_probs, queue_ptr):

    loss_epoch = 0
    start = time.time()
    for step, ((x_i, x_j, x_w), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        x_w = x_w.to('cuda')

        z_i, z_j, z_w, c_i, c_j, c_w = model(x_i, x_j, x_w)
        loss_week, queue_feats, queue_probs, queue_ptr= criterion_week(epoch, step, z_w, c_i, c_j, c_w, queue_feats, queue_probs, queue_ptr)
        loss_instance = criterion_instance(z_i, z_j)
        loss = loss_instance + loss_week
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(loss)
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}"
                )
        loss_epoch += loss.item()
    end = time.time()
    print('迭代一轮所需时间：{}'.format(end - start))
    return loss_epoch, queue_feats, queue_probs, queue_ptr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--batch-size',default=120,type=int)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir+'ImageNet-10',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_dir+'imagenet-dogs',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset,
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

    #loss
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_week = contrastive_loss.weekLoss(class_num, args.Truncated,args.batch_size, args.cluster_temperature, loss_device, args.queue_batch, args.alpha).to(loss_device)

    # memory bank
    args.queue_size = args.queue_batch * args.batch_size
    queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
    queue_probs = torch.zeros(args.queue_size, class_num).cuda()
    queue_ptr = 0

    for epoch in range(args.start_epoch, args.epochs):

        lr = optimizer.param_groups[0]["lr"]   #
        loss_epoch, queue_feats, queue_probs, queue_ptr = train(epoch, queue_feats=queue_feats, queue_probs=queue_probs, queue_ptr=queue_ptr)

        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
