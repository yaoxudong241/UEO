import os
import math
from decimal import Decimal
import time
import utility
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from model import vis_fea_map
import numpy as np
from model.util import BatchBlur
import lpips
import torch.nn.functional as F


def forward_chop(model, x, shave=6, min_size=10000,scale =4):
    # scale = 2#self.scale[self.idx_scale]
    n_GPUs = 1#min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        u_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)

            sr_batch = model(lr_batch, scale)

            # sr_batch = model(lr_batch)
            sr_list.extend(sr_batch[0].chunk(n_GPUs, dim=0))
            u_list.extend(sr_batch[1].chunk(n_GPUs, dim=0))
    else:
        sr_list,u_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size, scale= scale) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output_U = x.new(b, c, h, w)

    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    output_U[:, :, 0:h_half, 0:w_half] \
        = u_list[0][:, :, 0:h_half, 0:w_half]
    output_U[:, :, 0:h_half, w_half:w] \
        = u_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_U[:, :, h_half:h, 0:w_half] \
        = u_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_U[:, :, h_half:h, w_half:w] \
        = u_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output, output_U


class GaussianBlurConv(torch.nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0).cuda()
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.Gauss_Blur = GaussianBlurConv(args.n_colors)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.Sigmoid = torch.nn.Sigmoid()

        self.D = lpips.LPIPS(net='vgg', lpips=False).to("cuda")

    def test(self):
        epoch = self.scheduler.last_epoch

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        total_time = 0
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename,) in enumerate(tqdm_test):

                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:

                        lr = self.prepare([lr])[0]


                    tic = time.time()
                    # sr = hr


                    # sr = self.model(lr, idx_scale)
                    # sr_img,U = forward_chop(self.model, lr, shave=8, min_size=10000,scale =4)
                    sr_img, U = forward_chop(self.model, lr, shave=18, min_size=10000, scale=4)
                    sr = [sr_img, U]


                    total_time += time.time() - tic
                    if isinstance(sr, list):
                        var = sr[1]
                        sr = sr[0]
                        if var.size(1) > 1:
                            convert = var.new(1, 3, 1, 1)
                            convert[0, 0, 0, 0] = 65.738
                            convert[0, 1, 0, 0] = 129.057
                            convert[0, 2, 0, 0] = 25.064
                            var.mul_(convert).div_(256)
                            var = var.sum(dim=1, keepdim=True)
                    else:
                        sr = sr
                        var = None
                    sr = utility.quantize(sr, self.args.rgb_range)

                    # lr = lr
                    # hr = hr

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                    if var is not None:
                        vis_fea_map.draw_features(var.cpu().numpy(),
                                                  "{}/results/SR/{}/X{}/{}_var.png".format(self.args.save,
                                                                                               self.args.testset,
                                                                                               self.args.scale[0],
                                                                                               filename))
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(total_time), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model(lr, idx_scale)
            if isinstance(sr, list):

                # step1
                if self.args.stage == 'step1':
                    s = torch.exp(-sr[1])
                    sr_ = torch.mul(sr[0], s)
                    hr_ = torch.mul(hr, s)
                    # loss = self.loss(sr_,hr_) + 2* torch.mean(sr[1])

                    # gram
                    loss = 0.5*self.loss(sr_, hr_) + 0.5 * torch.mean(sr[1])

                # step2
                elif self.args.stage == 'step2':
                    b, c, h, w = sr[1].shape
                    # s1 = sr[1].view(b, c, -1)
                    # pmin = torch.min(s1, dim=-1)
                    # pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                    # s = sr[1]
                    # s = s - pmin+1
                    # sr_ = torch.mul(sr[0], s)
                    # hr_ = torch.mul(hr, s)
                    # loss = self.loss(sr_, hr_)

                    u_sig = torch.sigmoid(sr[1])
                    b, ch, w, h = u_sig.size()
                    s = w * h
                    sum_x = torch.sum(torch.sum(u_sig, dim=3), dim=2)  # *0.99+0.005
                    # sum_x = torch.sum(x)*0.99+0.005

                    alpha = F.relu((0.5 * s - sum_x) / (s - sum_x))
                    alpha = torch.unsqueeze(torch.unsqueeze(alpha, dim=2), dim=3)

                    beta = F.relu((sum_x - 0.5 * s) / sum_x)
                    beta = torch.unsqueeze(torch.unsqueeze(beta, dim=2), dim=3)

                    w = u_sig + alpha * (1 - u_sig) - beta * u_sig

                    ww = w.clone()
                    his, bin = torch.histogram(ww.cpu(), bins=100, range=(0., 1.))
                    # plt.plot(bin[:-1], torch.log10(his))
                    # plt.xlabel('fixedsum out')
                    # plt.savefig('f/fig_fixedsum_out.jpg')
                    # show_tensor_images(w * 2 - 1.0, filename='f/fixedsum_out.jpg')
                    w = 0.999 * w + 0.0005
                    n = torch.rand_like(w) * 0.999 + 0.0005

                    v = 10 * torch.log(w * n / ((1 - w) * (1 - n)))
                    v_ = torch.abs(v)
                    w = torch.exp((v - v_) / 2) / (torch.exp(-(v + v_) / 2) + torch.exp((v - v_) / 2))

                    # loss_fn_vgg = lpips.LPIPS(net='vgg', lpips=False).to("cuda")

                    # l = torch.abs(sr[0] - hr)
                    # combined1 = (1 - w) * sr[0].detach() + w * hr  # + noise
                    # combined2 = w * sr[0].detach() + (1 - w) * hr  # + noise
                    #
                    # loss_weight = torch.mean(
                    #     torch.log(torch.abs(self.D(combined2, hr, normalize=False)) + 0.001) - torch.log(
                    #         torch.abs(self.D(combined1, hr, normalize=False)) + 0.001))

                    w1 = w.view(b, c, -1)
                    pmin = torch.min(w1, dim=-1)
                    pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)

                    w = w - pmin + 1
                    sr_ = torch.mul(sr[0], w)
                    hr_ = torch.mul(hr, w)
                    loss = self.loss(sr_, hr_)

                else:
                    loss = self.loss(sr[0], hr)
            else:
                loss = self.loss(sr, hr)

            # if loss.item() < self.args.skip_threshold * self.error_last:
            loss.backward()
            self.optimizer.step()
            # else:
            #     print('Skip this batch {}! (Loss: {})'.format(
            #         batch + 1, loss.item()
            #     ))

            timer_model.hold()

            if (batch + 1) * self.args.batch_size % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
