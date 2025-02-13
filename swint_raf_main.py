import argparse
import torch
import datetime
import time
import numpy as np
from tqdm import tqdm
import os.path as osp
from utils.tools import save_checkpoint, setup_logger, set_random_seed, AverageMeter, load_checkpoint
from utils.lr_scheduler import build_lr_scheduler
import datasets.get_dataset as dataset
import torchvision.transforms as transforms
from models.backbone_raf import load_clip_to_cpu, CLIP_text, SwinT_clip
from utils.loss import TransformLoss, SelfConLoss
from torch.nn import functional as F
from utils.accuracy import compute_accuracy, compute_class_accuracy
from collections import OrderedDict
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from yacs.config import CfgNode as CN


def setup_cfg():

    # Data root and dataloader
    cfg = CN()
    cfg.ROOT = "data/raf_db/Image"
    cfg.LABEL_TRAIN = "data/raf_db/train_label.txt"
    cfg.LABEL_TEST = "data/raf_db/test_label.txt"
    cfg.NUM_WORKERS = 8
    cfg.TRAIN_BATCH_SIZE = 64
    cfg.TEST_BATCH_SIZE = 100

    # Optim and lr
    cfg.OPTIM = CN()
    cfg.OPTIM.LR = 0.002
    cfg.OPTIM.MAX_EPOCH = 50
    cfg.OPTIM.WARMUP_EPOCH = 1
    cfg.OPTIM.WARMUP_CONS_LR = 1e-4
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9

    # Output
    cfg.TRAIN_PRINT_FREQ = 50
    cfg.OUTPUT_DIR = "outputs"
    cfg.SEED = 29
    cfg.USE_CUDA = True
    cfg.freeze()
    return cfg


def main():
    global best_acc
    cfg = setup_cfg()
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")

    # Load dataset
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225) 

    transform_train = transforms.Compose([
        transforms.Resize(112),
        transforms.RandomApply([   
            transforms.RandomCrop(112, padding=4) 
        ], p=0.5),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    train_dataset, test_dataset = dataset.get_dataset(cfg.ROOT, cfg.ROOT, cfg.LABEL_TRAIN, cfg.LABEL_TEST, transform_train=transform_train, transform_val=transform_val)

    # Build train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        sampler=RandomSampler(train_dataset),
        num_workers=cfg.NUM_WORKERS,
        drop_last=True,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    
    # Build test_loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        sampler=SequentialSampler(test_dataset), 
        num_workers=cfg.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )

    # Build the model
    clip_model = load_clip_to_cpu() 
    model_text = CLIP_text(clip_model).to(device) 
    model = SwinT_clip()
    model = model.to(device)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Build the optimizer and lr_scheduler
    optim = torch.optim.SGD(
                model.parameters(),
                lr = cfg.OPTIM.LR,
                momentum = cfg.OPTIM.MOMENTUM,
                weight_decay = cfg.OPTIM.WEIGHT_DECAY
                )
    sched = build_lr_scheduler(optim, cfg.OPTIM)

    criterion_t = TransformLoss()
    criterion_c = SelfConLoss()

    start_epoch = 0     
    best_result = -np.inf
    directory = cfg.OUTPUT_DIR
    time_start = time.time()
    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        train(train_loader, model_text, model, device, criterion_t, criterion_c, optim, sched, cfg, epoch)

        curr_result = test(test_loader, model_text, model, device, epoch)
        is_best = curr_result > best_result
        if is_best:
            best_result = curr_result
            save_model(model, optim, sched, epoch, directory, val_result=curr_result, model_name="model-best.pth.tar")

    print("Finish training")
    print("Deploy the model with the best performance")
    load_model(model, directory)
    test(test_loader, model_text, model, device, epoch)

    # Show elapsed time
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed: {elapsed}")


def train(train_loader, model_text, model, device, criterion_t, criterion_c, optim, sched, cfg, epoch):
    losses = AverageMeter()
    losses_s = AverageMeter()
    losses_t = AverageMeter()
    losses_c = AverageMeter()
    accuracy = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    num_batches = len(train_loader)

    end = time.time()
    model_text.eval()
    model.train()
    for batch_idx, (input, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
    
        input = input.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            feature_t, logit_scale = model_text()
            feature_l = [feature_t[x] for x in label]
            feature_l = torch.stack(feature_l)
                        
        feature_e, feature_n, output_e = model(input, feature_t, logit_scale)

        # Calculate transformation loss
        idx = torch.where(label != 6) #Index (6) for Neutral category
        nf_feature = feature_t[6, :].repeat(feature_l.shape[0], 1)
        delta_f = feature_e - feature_n
        delta_t = feature_l - nf_feature
        delta_f, delta_t = delta_f[idx[0]], delta_t[idx[0]]
        Lt = criterion_t(delta_f / delta_f.norm(dim=-1, keepdim=True), delta_t / delta_t.norm(dim=-1, keepdim=True)).mean()

        # Calculate cross-entropy loss and self-contrast objective
        Ls = F.cross_entropy(output_e, label)
        Lc = criterion_c(feature_l[idx[0]], feature_e[idx[0]], feature_n[idx[0]])

        # Total loss
        loss = Ls * 1.0 + Lt * 1.0 + Lc * 1.0

        acc = compute_accuracy(output_e, label)[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (batch_idx + 1) == num_batches:
            sched.step()

        batch_time.update(time.time() - end)

        accuracy.update(acc.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        losses_s.update(Ls.item(), input.size(0))
        losses_t.update(Lt.item(), input.size(0))
        losses_c.update(Lc.item(), input.size(0))
   
        meet_freq = (batch_idx + 1) % cfg.TRAIN_PRINT_FREQ == 0
        only_few_batches = num_batches < cfg.TRAIN_PRINT_FREQ
        if meet_freq or only_few_batches:
            nb_remain = 0
            nb_remain += num_batches - batch_idx - 1
            nb_remain += (
                cfg.OPTIM.MAX_EPOCH - epoch - 1
            ) * num_batches
            eta_seconds = batch_time.avg * nb_remain
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

            info = []
            info += [f"epoch [{epoch + 1}/{cfg.OPTIM.MAX_EPOCH}]"]
            info += [f"batch [{batch_idx + 1}/{num_batches}]"]
            info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
            info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
            info += [f"Ls {losses_s.avg:.4f}"]                    
            info += [f"Lt {losses_t.avg:.4f}"]
            info += [f"Lc {losses_c.avg:.4f}"]
            info += [f"loss {losses.avg:.3f}"]
            info += [f"acc {accuracy.avg:.3f}"]
            info += [f"lr {optim.param_groups[0]['lr']:.4e}"]
            info += [f"eta {eta}"]
            print(" ".join(info))

        end = time.time()
    return losses.avg, accuracy.avg    


def test(test_loader, model_text, model, device, epoch):
    batch_time = AverageMeter()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    model_text.eval()
    model.eval()
    print(f"Evaluation on the testing set")
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, label) in enumerate(tqdm(test_loader)):
            
            input = input.to(device)
            label = label.to(device)
           
            feature_t, logit_scale= model_text()  
            _, _, output = model(input, feature_t, logit_scale)

            pred = output.max(1)[1]
            matches = pred.eq(label).float()
            correct += int(matches.sum().item())
            total += label.shape[0]

            if batch_idx == 0:
                all_predicted = pred
                all_label = label
            else:
                all_predicted = torch.cat((all_predicted, pred), dim=0)
                all_label = torch.cat((all_label, label), dim=0)

            y_true.extend(label.data.cpu().numpy().tolist())
            y_pred.extend(pred.data.cpu().numpy().tolist())

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        results = OrderedDict()
        acc = 100.0 * correct / total
        averacc, averdata = compute_class_accuracy(all_predicted, all_label)
        results["accuracy"] = acc
        results["class_avg_accuracy"] = averacc
        results["Sur"], results["Fea"], results["Dis"], results["Hap"], results["Sad"], results["Ang"], results["Neu"] = averdata[0], averdata[1], averdata[2], averdata[3], averdata[4], averdata[5], averdata[6] 

        print(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* correct: {correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* class_avg_accuracy: {averacc:.2f}%\n"
            f"* Sur: {averdata[0]:.2f}%, "
            f" Fea: {averdata[1]:.2f}%, "
            f" Dis: {averdata[2]:.2f}%, "
            f" Hap: {averdata[3]:.2f}%, "
            f" Sad: {averdata[4]:.2f}%, "
            f" Ang: {averdata[5]:.2f}%, "
            f" Neu: {averdata[6]:.2f}%\n"
        )
        
        return list(results.values())[0]


def save_model(model, optim, sched, epoch, directory, is_best=False, val_result=None, model_name=""):
    save_checkpoint(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "val_result": val_result
        },
        osp.join(directory, 'image_encoder'),
        is_best=is_best,
        model_name=model_name,
    )


def load_model(model, directory, epoch=None):
    if not directory:
        print("Note that load_model() is skipped as no pretrained model is given")
        return

    names = ['image_encoder']
    model_file = "model-best.pth.tar"
    
    for name in names:
        model_path = osp.join(directory, name, model_file)
    
        if not osp.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
    
        model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    main()