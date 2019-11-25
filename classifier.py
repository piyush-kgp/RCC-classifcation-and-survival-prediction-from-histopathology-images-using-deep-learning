

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torchvision
from PIL import Image
import argparse
import copy
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description='Process args for Binary Classifer')
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--val_dir", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=12)
parser.add_argument("--num_nodes", type=int, default=256)
parser.add_argument("--imagenet_model", type=str, default="resnet18")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--log_dir", type=str, default="runs/")
parser.add_argument("--save_prefix", type=str, required=True)
parser.add_argument("--model_checkpoint", type=str, required=False)
parser.add_argument("--optimzer_checkpoint", type=str, required=False)
parser.add_argument("--exp_lr_scheduler_checkpoint", type=str, required=False)


MODEL_DICT = {"resnet18" : models.resnet18(pretrained=True),
              # "alexnet" : models.alexnet(pretrained=True),
              # "squeezenet" : models.squeezenet1_0(pretrained=True),
              # "vgg16" : models.vgg16(pretrained=True),
              # "densenet" : models.densenet161(pretrained=True),
              # "inception" : models.inception_v3(pretrained=True),
              # "googlenet" : models.googlenet(pretrained=True),
              # "shufflenet" : models.shufflenet_v2_x1_0(pretrained=True),
              # "mobilenet" : models.mobilenet_v2(pretrained=True),
              # "resnext50_32x4d" : models.resnext50_32x4d(pretrained=True),
              # "wide_resnet50_2" : models.wide_resnet50_2(pretrained=True),
              # "mnasnet" : models.mnasnet1_0(pretrained=True)
             }


def train(model, train_dataloader, optimizer, criterion, device, epoch, scheduler, writer):
    num_batches = len(train_dataloader)
    model.train()
    running_correct = 0
    for batch_id, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(input=output, target=target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        running_correct += correct
        batch_size = len(data)
        running_loss = loss.item()/batch_size
        print("[Train] Epoch: {} [{}/{}]    Loss: {:.6f}   Batch Acc: {:.2f}".format(
              epoch, (batch_id+1)*batch_size, len(train_dataloader.dataset),
              running_loss, correct/batch_size*100), flush=True)
        writer.add_scalar('Loss/Train', running_loss, num_batches*epoch+batch_id)
        writer.add_scalar('Accuracy/Train', correct/batch_size*100, num_batches*epoch+batch_id)
    epoch_acc = running_correct/len(train_dataloader.dataset)*100
    epoch_loss = running_loss/len(train_dataloader.dataset)
    scheduler.step(epoch_loss)
    return epoch_acc


def test(model, val_dataloader, criterion, device, epoch, target_names, writer, save_prefix):
    model.eval()
    val_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(val_dataloader):
            # grid = torchvision.utils.make_grid(data)
            # writer.add_image('images', grid, 0)
            # # writer.add_graph(model, data)
            # writer.close()

            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            y_pred.append(output.argmax(dim=1))
            y_true.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("Validation Done: [{}/{}]".format((batch_id+1)*len(data), \
                  len(val_dataloader.dataset)), flush=True)

    val_loss /= len(val_dataloader.dataset)

    val_acc = 100.*correct/len(val_dataloader.dataset)
    print("[Test] Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
          epoch, val_loss, correct, len(val_dataloader.dataset), val_acc), flush=True
         )
    writer.add_scalar('Loss/Eval', val_loss, epoch)
    writer.add_scalar('Accuracy/Eval', val_acc, epoch)
    # for tag, param in model.named_parameters():
    #     writer.add_histogram(tag, param, epoch)
    y_true, y_pred = torch.cat(y_true).cpu().numpy(), torch.cat(y_pred).cpu().numpy()
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, digits=4)
    print(report, flush=True)
    return val_acc


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        super(MyDataset, self).__init__()
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        X, y = self.subset[index]
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.subset)



def roc_curve(y_true, y_pred):
    fpr = []
    tpr = []
    acc = []
    thresholds = np.arange(0.0, 1.01, .01)

    P = sum(y_true)
    N = len(y_true) - P

    for thresh in thresholds:
        FP=0
        TP=0
        for y_t, y_p in zip(y_true, y_pred):
            if y_p >= thresh:
                if y_t == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))
        acc.append((TP+N-FP)/len(y_true))
    # plt.plot(fpr, tpr)
    return tpr, fpr, acc, thresholds


def integrate(x_s, y_s):
    # integartion by trapezoidal rule
    x_diffs = [abs(x_s[i]-x_s[i-1]) for i in range(1,len(x_s))]
    y_sum = [y_s[i]+y_s[i-1] for i in range(1,len(y_s))]
    return .5*sum([a*b for a, b in zip(y_sum, x_diffs)])


class SlideDataset(Dataset):
    def __init__(self, file_paths, transform):
        super(SlideDataset, self).__init__()
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(Image.open(self.file_paths[idx]))

    def __len__(self):
        return len(self.file_paths)


def slide_wise_analysis(root, model, epoch, classes, transform, device, batch_size, num_char_slide, save_prefix):
    model.eval()
    isImage = lambda f: f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")

    slide_wise_output = dict()
    results_gt = []
    results_pred = []
    for class_id, class_ in enumerate(classes):
        slide_wise_output[class_] = dict()
        files = os.listdir(os.path.join(root, class_))
        files = [f for f in files if isImage(f)]
        slide_ids = set([f[:num_char_slide] for f in files])

        for slide_id in slide_ids:
            y_pred = []
            slide_files = [os.path.join(root, class_, f) for f in files if f.startswith(slide_id)]
            slide_files = sorted(slide_files)
            slide_dataset = SlideDataset(slide_files, transform)
            slide_dataloader = DataLoader(slide_dataset, batch_size=batch_size, shuffle=False)
            for batch in slide_dataloader:
                batch = batch.to(device)
                out = model(batch)
                y_pred.append(out.argmax(dim=1))

            y_pred = torch.cat(y_pred)
            fraction_positive = int(sum(y_pred).cpu().numpy())/len(y_pred)
            slide_wise_output[class_][slide_id] = fraction_positive

            results_gt.append(class_id)
            results_pred.append(fraction_positive)

    print("slide_wise_output", slide_wise_output, flush=True)
    print("results_gt", results_gt, flush=True)
    print("results_pred", results_pred, flush=True)

    tpr, fpr, accuracies, thresholds = roc_curve(results_gt, results_pred)
    roc_auc = integrate(x_s=fpr, y_s=tpr)

    acc_max = max(accuracies)
    threshold_opt = thresholds[accuracies.index(acc_max)]

    print("Slide Wise ROC-AUC = {}".format(roc_auc), flush=True)
    roc_auc = metrics.roc_auc_score(results_gt, results_pred)
    print("Slide Wise ROC-AUC by sklearn = {}".format(roc_auc), flush=True)
    print("Max Accuracy = {} at fraction = {}".format(acc_max, threshold_opt), flush=True)

    conf_mat = metrics.confusion_matrix(results_gt, results_pred>threshold_opt)
    print("Confusion Matrix\n", conf_mat, flush=True)
    kappa_score = metrics.cohen_kappa_score(results_gt, results_pred>threshold_opt)
    print("Cohen Kappa Score = {}".format(kappa_score), flush=True)


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    plt.savefig("{}_slide_roc_curve_epoch_{}.jpg".format(save_prefix, epoch))

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Accuracy vs Threshold')
    ax.plot(thresholds, accuracies, 'b', label = 'max acc = %0.2f at T =%0.2f' % \
            (max(accuracies), thresholds[accuracies.index(max(accuracies))]))
    ax.legend(loc = 'lower right')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Threshold Fraction of normal patches for WSI to be considered Normal')
    plt.savefig("{}_accuracy_vs_threshold_epoch_{}.jpg".format(save_prefix, epoch))



def main():
    args = parser.parse_args()

    img_dir = args.img_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_nodes = args.num_nodes
    learning_rate = args.learning_rate
    image_size = args.image_size
    num_classes = args.num_classes
    imagenet_model = args.imagenet_model
    log_dir = args.log_dir
    val_dir = args.val_dir
    save_prefix = args.save_prefix
    model_checkpoint = args.model_checkpoint
    optimzer_checkpoint = args.optimzer_checkpoint
    exp_lr_scheduler_checkpoint = args.exp_lr_scheduler_checkpoint

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ]),
    }

    if val_dir is not None:
        train_dataset = datasets.ImageFolder(root=img_dir, transform=data_transforms["train"])
        val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms["val"])
    else:
        dataset = datasets.ImageFolder(root=img_dir)
        num_train = int(len(dataset)*0.8)
        num_val = len(dataset) - num_train
        train_dataset, val_dataset = random_split(dataset, (num_train, num_val))

        train_dataset = MyDataset(train_dataset, data_transforms["train"])
        val_dataset = MyDataset(val_dataset, data_transforms["val"])


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = MODEL_DICT[imagenet_model]
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if model_checkpoint is not None:
        # TODO: This is because checkpoint dictionary has "module." in each key
        ckpt = torch.load(model_checkpoint)
        ckpt = {k[7:]: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print("[MSG] Loading {}".format(model_checkpoint), flush=True)

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    #
    # print(model, flush=True)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("DEVICE {}".format(device), flush=True)
    # model = nn.DataParallel(model).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params=model.module.fc.parameters(), lr=learning_rate, weight_decay=0.05)
    # exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)


    writer = SummaryWriter(log_dir=log_dir)

    classes=["cancer", "normal"]

    # #Load State Dicts
    # for item, ckpt in zip([model, optimizer, exp_lr_scheduler], \
    #     [model_checkpoint, optimzer_checkpoint, exp_lr_scheduler_checkpoint]):
    #     if ckpt is not None:
    #         try:
    #             item.load_state_dict(torch.load(ckpt))
    #             print("[MSG] Loading {}".item(ckpt), flush=True)
    #         except:
    #             print("[FAIL] Loading {} failed".format(ckpt), flush=True)
    #             pass


    # test(model, val_dataloader, criterion, device, -1, classes, writer)
    # val_acc_prev = 0
    # for epoch in range(0, num_epochs):
    #     train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
    #
    #     for item, name in zip([model, optimizer, exp_lr_scheduler], ["model", "optimizer", "exp_lr_scheduler"]):
    #         torch.save(item.state_dict(), "{}_{}_epoch_{}.pth".format(save_prefix, name, epoch))
    #
    #     val_acc = test(model, val_dataloader, criterion, device, epoch, classes, writer, save_prefix)
    #     # removing this condition for now
    #     # if val_acc-val_acc_prev <= 1: #<1% acc increase then stop
    #     #     break
    #     writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    #     # slide_wise_analysis(root=val_dir, model=model, epoch=epoch, classes=classes, \
    #     #                     transform=data_transforms["val"], device=device, \
    #     #                     batch_size=batch_size, num_char_slide=60, save_prefix=save_prefix)
    #     # val_acc_prev = val_acc
    #
    # print("[MSG]: Last FC layer Trained", flush=True)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer4[1].parameters():
        param.requires_grad = True
    print(model, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = list(model.module.fc.parameters()) + list(model.module.layer4[1].parameters())
    optimizer = optim.Adam(params=trainable_params, lr=learning_rate, weight_decay=0.05)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)

    for epoch in range(6, num_epochs):
        train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
        torch.save(model.state_dict(), "{}_model_epoch_{}.pth".format(save_prefix, epoch))
        val_acc = test(model, val_dataloader, criterion, device, epoch, classes, writer, save_prefix)
        writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    print("[MSG]: Last FC and Layer4[1] Trained", flush=True)
    #
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    # for param in model.layer4[1].parameters():
    #     param.requires_grad = True
    # for param in model.layer4[0].parameters():
    #     param.requires_grad = True
    # trainable_params = list(model.fc.parameters()) + list(model.layer4[1].parameters()) + \
    #                    list(model.layer4[0].parameters())
    # optimizer = optim.Adam(params=trainable_params, lr=learning_rate, weight_decay=0.05)
    # exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)
    #
    # for epoch in range(epoch, num_epochs):
    #     train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
    #     torch.save(model.state_dict(), "{}_model_epoch_{}.pth".format(save_prefix, epoch))
    #     torch.save(optimizer.state_dict(), "{}_optimizer_epoch_{}.pth".format(save_prefix, epoch))
    #     val_acc = test(model, val_dataloader, criterion, device, epoch, classes, writer, save_prefix)
    #     if val_acc-val_acc_prev <= 1: #<1% acc increase then stop
    #         break
    #     writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    #     slide_wise_analysis(root=val_dir, model=model, epoch=epoch, classes=classes, \
    #                         transform=data_transforms["val"], device=device, \
    #                         batch_size=batch_size, num_char_slide=60, save_prefix=save_prefix)
    #     val_acc_prev = val_acc
    #
    # print("[MSG]: Last FC, Layer4[1], Layer4[0] Trained", flush=True)
    #
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    # for param in model.layer4[1].parameters():
    #     param.requires_grad = True
    # for param in model.layer4[0].parameters():
    #     param.requires_grad = True
    # for param in model.layer3[1].parameters():
    #     param.requires_grad = True
    # trainable_params = list(model.fc.parameters()) + list(model.layer4[1].parameters()) + \
    #                    list(model.layer4[0].parameters()) + list(model.layer3[1].parameters())
    # optimizer = optim.Adam(params=trainable_params, lr=learning_rate, weight_decay=0.05)
    # exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)
    #
    #
    # for epoch in range(epoch, num_epochs):
    #     train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
    #     torch.save(model.state_dict(), "{}_model_epoch_{}.pth".format(save_prefix, epoch))
    #     torch.save(optimizer.state_dict(), "{}_optimizer_epoch_{}.pth".format(save_prefix, epoch))
    #     val_acc = test(model, val_dataloader, criterion, device, epoch, classes, writer, save_prefix)
    #     if val_acc-val_acc_prev <= 1: #<1% acc increase then stop
    #         break
    #     writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    #     slide_wise_analysis(root=val_dir, model=model, epoch=epoch, classes=classes, \
    #                         transform=data_transforms["val"], device=device, \
    #                         batch_size=batch_size, num_char_slide=60, save_prefix=save_prefix)
    #     val_acc_prev = val_acc
    #
    # print("[MSG]: Last FC, Layer4[1], Layer4[0], Layer3[1] Trained", flush=True)
    #
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    # for param in model.layer4[1].parameters():
    #     param.requires_grad = True
    # for param in model.layer4[0].parameters():
    #     param.requires_grad = True
    # for param in model.layer3[1].parameters():
    #     param.requires_grad = True
    # for param in model.layer3[0].parameters():
    #     param.requires_grad = True
    # trainable_params = list(model.fc.parameters()) + list(model.layer4[1].parameters()) + \
    #                    list(model.layer4[0].parameters()) + list(model.layer3[1].parameters()) + \
    #                    list(model.layer3[0].parameters())
    # optimizer = optim.Adam(params=trainable_params, lr=learning_rate, weight_decay=0.05)
    # exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)
    #
    # for epoch in range(epoch, num_epochs):
    #     train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
    #     torch.save(model.state_dict(), "{}_model_epoch_{}.pth".format(save_prefix, epoch))
    #     torch.save(optimizer.state_dict(), "{}_optimizer_epoch_{}.pth".format(save_prefix, epoch))
    #     val_acc = test(model, val_dataloader, criterion, device, epoch, classes, writer, save_prefix)
    #     if val_acc-val_acc_prev <= 1: #<1% acc increase then stop
    #         break
    #     writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    #     slide_wise_analysis(root=val_dir, model=model, epoch=epoch, classes=classes, \
    #                         transform=data_transforms["val"], device=device, \
    #                         batch_size=batch_size, num_char_slide=60, save_prefix=save_prefix)
    #     val_acc_prev = val_acc
    #
    # print("[MSG]: Last FC, Layer4[1], Layer4[0], Layer3[1], Layer3[0] Trained", flush=True)
    # torch.save(model.state_dict(), "{}_model_epoch_{}.pth".format(save_prefix, "fin"))
    # torch.save(optimizer.state_dict(), "{}_optimizer_epoch_{}.pth".format(save_prefix, "fin"))


if __name__=="__main__":
    main()
