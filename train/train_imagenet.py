import os
import sys
import time
import torch
import shutil
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
import dataloader.imagenet as data_sets
import tools.tools as transforms
import model.mobilenetv2 as m


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, predict = output.topk(max_k, 1, True, True)
        predict = predict.t()
        correct = predict.eq(target.view(1, -1).expand_as(predict))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch, gpu_id, print_freq=10):
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data', ':.2f')
    losses = AverageMeter('Loss', ':.2')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x_input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if gpu_id > -1:
            x_input = x_input.cuda(gpu_id, non_blocking=True)
            x_input = x_input.type(torch.cuda.FloatTensor)
            target = target.cuda(gpu_id, non_blocking=True)

        # compute output
        output = model(x_input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, top_k=(1, 5))
        losses.update(loss.item(), x_input.size(0))
        top1.update(acc1[0], x_input.size(0))
        top5.update(acc5[0], x_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, gpu_id, print_freq=10):
    batch_time = AverageMeter('Time', ':.2f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x_input, target) in enumerate(val_loader):
            if gpu_id > -1:
                x_input = x_input.cuda(gpu_id, non_blocking=True)
                x_input = x_input.type(torch.cuda.FloatTensor)
                target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(x_input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), x_input.size(0))
            top1.update(acc1[0], x_input.size(0))
            top5.update(acc5[0], x_input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, best_name_prefix, filename='checkpoint.pkl'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name_prefix + '_model_best.pkl')


def main(data_root, batch_size, epoch_number, gpu_id, f_name,
         lr=0.01, momentum=0.9, weight_decay=1e-4, act_id=1, init=1.0, arch="mobilenet"):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    # Gray = R*0.299 + G*0.587 + B*0.114
    # gray: mean = 0.458971, std = 0.225609
    # xy_axis: mean=[-0.00223214 -0.00223214], std = [0.28867226 0.28867226]
    # R,G,B,Gray
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.459],
                                     std=[0.229, 0.224, 0.225, 0.226])

    train_data_set = data_sets.ImageFolder(
        train_dir,
        224,
        transforms.Compose([
            # Resize according to aspect ratio
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        data_sets.ImageFolder(
            val_dir,
            224,
            transforms.Compose([
                # Resize according to aspect ratio
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    print("gpu_id =", gpu_id, " act_id =", act_id, "init =", init,
          "filename =", f_name, "lr =", lr, "num_classes =", data_sets.CLASS_NUMBER,
          "batch_size =", batch_size)
    model = m.MobileNet2(input_size=224, scale=1.0, act_id=act_id, init_value=init, num_classes=data_sets.CLASS_NUMBER)
    model.float()
    criterion = nn.CrossEntropyLoss()

    if gpu_id > -1:
        device = torch.device("cuda:" + str(gpu_id))
        model.to(device)
        model.float()
        criterion = criterion.cuda(gpu_id)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    best_acc1 = 0.0

    filename = f_name
    record = open(filename, "w")
    record.close()

    for epoch in range(epoch_number):

        adjust_learning_rate(optimizer, epoch, base_lr=lr)

        # train for one epoch
        t_acc1, t_acc5 = train(train_loader, model, criterion, optimizer, epoch, gpu_id)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, gpu_id)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        record = open(filename, "a+")
        record.write(str(epoch) + ","
                     + str(t_acc1.cpu().numpy())
                     + "," + str(t_acc5.cpu().numpy())
                     + "," + str(acc1.cpu().numpy()) + ","
                     + str(acc5.cpu().numpy()) + "\n")
        record.close()

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, f_name[:-4])


if __name__ == "__main__":

    act_id, gpu_id, init, f_name, lr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], float(
        sys.argv[5])
    main("/dataset/min_imagenet", batch_size=128, epoch_number=50, gpu_id=gpu_id,
         act_id=act_id, init=init, f_name=f_name, lr=lr)


