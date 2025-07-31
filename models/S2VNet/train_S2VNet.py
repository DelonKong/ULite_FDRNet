import os
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import grouper, sliding_window, count_sliding_window


def train_S2VNet(network, optimizer, criterion, train_loader, val_loader, epoch, saving_path, device, scheduler=None):

    best_acc = -0.1
    losses = []

    for e in tqdm(range(1, epoch+1), desc="training the network"):
        network.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            # images, targets = images.cuda(), targets.cuda()
            optimizer.zero_grad()

            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, feature_abu, edm_per = network(images)

            band = re_unmix.shape[1] // 2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            re_unmix = re_unmix_nonlinear + output_linear

            # compute kl loss
            kl_div = -0.5 * (edm_var_2 + 1 - edm_var_1 ** 2 - edm_var_2.exp())
            kl_div = kl_div.sum() / batch_pred.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(0).cuda())

            # compute tv loss
            edm_per_diff = edm_per[1:, :] - edm_per[:(edm_per.shape[0] - 1), :]
            edm_per_diff = edm_per_diff.abs()
            loss_tv = edm_per_diff.mean()  # endmember tv_loss

            b_x, h_x, w_x = feature_abu.shape[0], feature_abu.shape[-2], feature_abu.shape[-1]
            h_tv = torch.pow((feature_abu[:, :, 1:, :] - feature_abu[:, :, :h_x - 1, :]), 2).sum()
            w_tv = torch.pow((feature_abu[:, :, :, 1:] - feature_abu[:, :, :, :w_x - 1]), 2).sum()
            loss_tv_abu = (h_tv + w_tv) / (b_x * 2 * h_x * w_x)  # abundance tv_loss

            sad_loss = torch.mean(torch.acos(torch.sum(images * re_unmix, dim=2) /
                                             (torch.norm(re_unmix, dim=1, p=2) * torch.norm(images, dim=2,
                                                                                            p=2) + 1e-5)))
            loss = criterion(batch_pred, targets) + sad_loss + 0.01 * kl_div + 0.01 * loss_tv + 0.01 * loss_tv_abu

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            train_info = "train at epoch {}/{}, loss={:.6f}"
            train_info = train_info.format(e, epoch,  mean_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []

        val_acc = validation_S2VNet(network, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        check_acc = 1
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        if best_acc > check_acc:
            print("======break=======")
            break
        save_checkpoint(network, is_best, saving_path, epoch=e, acc=best_acc)
        torch.cuda.empty_cache()


def validation_S2VNet(network, val_loader, device):
    num_correct = 0.
    total_num = 0.
    network.eval()
    for batch_idx, (images, targets) in enumerate(val_loader):
        images, targets = images.to(device), targets.to(device)
        # images, targets = images.cuda(), targets.cuda()
        re_unmix_nonlinear, re_unmix, outputs, edm_var_1, edm_var_2, _, _ = network(images)
        _, outputs = torch.max(outputs, dim=1)
        for output, target in zip(outputs, targets):
            num_correct = num_correct + (output.item() == target.item())
            # item()用于在只包含一个元素的tensor中提取值，注意是只包含一个元素，否则的话使用.tolist()
            total_num = total_num + 1
    overall_acc = num_correct / total_num
    return overall_acc


def test_S2VNet(network, model_dir, image, patch_size, n_classes, device, batch_size):
    network.load_state_dict(torch.load(model_dir + "/model_best.pth", weights_only=True))
    # network.load_state_dict(torch.load(model_dir + "/model_best.pth"), strict=False)
    network.eval()

    patch_size = patch_size
    batch_size = batch_size
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    # pad the image
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    probs = np.zeros(image.shape[:2] + (n_classes, ))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations,
                      desc="inference on the HSI"):
        with torch.no_grad():
            # batch : image[x:x + w, y:y + h], x, y, w, h
            # b[0] : image[x:x + w, y:y + h]
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            # data = data.cuda()
            re_unmix_nonlinear, re_unmix, output, edm_var_1, edm_var_2, _, _ = network(data)
            if isinstance(output, tuple):
                output = output[0]
            # output = output.to('cpu').numpy()
            output = output.cpu().numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out

    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_checkpoint(network, is_best, saving_path, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best validation OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))
    else:  # save the ckpt for each 10 epoch
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, 'model.pth'))


