import sys
import time
import torch
import torch.nn as nn
import model.mobilenetv2 as m
import dataloader.cifar10 as r
from torch.autograd import Variable
from torch.utils.data import DataLoader

SEED = int(23356)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

EPOCH = 100
L_R = 2e-3  # 1e-4, 5e-4, 1e-3, 2e-3
BATCH_SIZE = 1024  # 1024, 256
VAL_BATCH_SIZE = 128

train_data = r.DataSet(is_train=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = r.DataSet(is_train=False)
val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE)


def poly_lr_scheduler(optimizer, init_lr, c_iter, max_iter, power):
    new_lr = init_lr * (1 - float(c_iter) / max_iter) ** power
    for para_group in optimizer.param_groups:
        para_group['lr'] = new_lr


def train(act_id, init, filename, lr):

    is_gpu = torch.cuda.is_available()
    print("is_gpu =", is_gpu, " act_id =", act_id, "init =", init,
          "filename =", filename, "lr =", lr, "num_classes =", r.CLASS_NUMBER)

    # select model
    model = m.MobileNet2(input_size=224, scale=1.0, act_id=act_id, init_value=init, num_classes=r.CLASS_NUMBER)
    model.float()

    if is_gpu:
        device = torch.device("cuda:0")
        model.to(device)
        model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=L_R, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(class_num=21)

    record = open(filename, "w")
    record.close()

    optimizer.zero_grad()
    for epoch in range(EPOCH):

        model = model.train()
        loss_value = 0.0
        acc_value = 0.0
        iter_count = 0
        epoch_start_time = time.time()

        for batch_data, batch_label in train_loader:
            start_time = time.time()

            # poly_lr_scheduler(optimizer, L_R, iter_count, max_iter=30000, power=0.9)

            if is_gpu:
                batch_data, batch_label = batch_data.cuda().type(torch.cuda.FloatTensor), \
                                          batch_label.cuda().type(torch.cuda.LongTensor)
            else:
                batch_data, batch_label = batch_data.type(torch.FloatTensor), batch_label.type(torch.LongTensor)

            batch_data, batch_label = Variable(batch_data), Variable(batch_label)
            out = model(batch_data)

            predict = torch.argmax(out, dim=1)
            correct = torch.sum(predict == batch_label).cpu().numpy()
            acc = correct / BATCH_SIZE

            loss = criterion(out, batch_label)
            loss_value += loss.data.item() * BATCH_SIZE
            acc_value += acc * BATCH_SIZE
            iter_count += BATCH_SIZE

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Iter: [%2d/%2d] || Time: %4.4f sec || Loss: %.5f(%.5f) || Acc: %.4f(%.4f)' % (
                iter_count, len(train_data), time.time() - start_time,
                loss.item(), loss_value / iter_count, acc, acc_value / iter_count))
        loss_value /= len(train_data)
        acc_value /= len(train_data)
        print('Epoch: [%2d/%2d] || Time: %4.4f sec || Loss: %.5f || Acc: %.4f'
              % (epoch, EPOCH, time.time() - epoch_start_time, loss_value, acc_value))

        if (epoch + 1) % 100 == 0:
            for para_group in optimizer.param_groups:
                para_group['lr'] *= 0.5
        for para_group in optimizer.param_groups:
            print("Epoch %2d, LR: %f" % (epoch, para_group['lr']))

        val_acc, val_loss = validate(model, val_loader, VAL_BATCH_SIZE, len(val_data), criterion)
        record = open(filename, "a+")
        record.write(
            str(epoch) + "," + str(acc_value) + "," + str(loss_value) + "," + str(val_acc) + "," + str(val_loss) + "\n")
        record.close()


def load_model(model_path):
    return torch.load(model_path)


def validate(model, data_loader, batch_size, data_size, criterion):
    is_gpu = torch.cuda.is_available()

    start_time = time.time()
    model = model.eval()
    acc_value = 0.0
    loss_value = 0.0

    for batch_data, batch_label in data_loader:
        if is_gpu:
            batch_data, batch_label = batch_data.cuda().type(torch.cuda.FloatTensor), \
                                      batch_label.cuda().type(torch.cuda.LongTensor)
        else:
            batch_data, batch_label = batch_data.type(torch.FloatTensor), batch_label.type(torch.LongTensor)
        batch_data, batch_label = Variable(batch_data), Variable(batch_label)
        batch_data, batch_label = Variable(batch_data), Variable(batch_label)
        out = model(batch_data)
        loss = criterion(out, batch_label)
        loss_value += loss.data.item() * batch_size

        predict = torch.argmax(out, dim=1)
        correct = torch.sum(predict == batch_label).cpu().numpy()
        acc_value += correct

    acc_value /= data_size
    loss_value /= data_size
    print( 'Validate Iter: Time: %4.4f sec || Loss: %.5f || Acc: %.4f'
           % (time.time() - start_time, loss_value, acc_value))

    return acc_value, loss_value


if __name__ == "__main__":
    # act_id, init_value, filename, lr=0.01
    train(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
