import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
def cutout_aware(image, kpoint, mask_size_min=3, mask_size_max=15, p=0.2, cutout_inside=True, mask_color=(0, 0, 0),
                cut_n=10, batch_size=2, bbox_trg=None, n_pts=None, cutout_size_min=0.03, cutout_size_max=0.1):
    # mask_size = torch.randint(low=mask_size_min, high=mask_size_max, size=(1,))

    # mask_size_half = mask_size // 2
    # offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image, x, y, mask_size, mask_size_half):
        # image = np.asarray(image).copy()

        if torch.rand(1) > p:
            return image

        _, h, w = image.shape

        # if cutout_inside:
        #     cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        #     cymin, cymax = mask_size_half, h + offset - mask_size_half
        # else:
        #     cxmin, cxmax = 0, w + offset
        #     cymin, cymax = 0, h + offset

        cx = x  # torch.randint(low=cxmin[0], high=cxmax[0], size=(1,))
        cy = y  # torch.randint(low=cymin[0], high=cymax[0], size=(1,))
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[:, ymin:ymax, xmin:xmax] = 0  # mask_color
        return image

    image_new = []
    for b in range(batch_size):
        bbox = bbox_trg[b]
        max_mask_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        buffer = image[b]
        kpoint_n = len(kpoint)
        # non_zero_index = torch.nonzero(kpoint[b][0])
        non_zero_index = torch.arange(n_pts[b])
        # cut_n_rand = torch.randint(1, cut_n, size=(1,))

        for n in range(len(non_zero_index)):
            if math.isclose(cutout_size_min, cutout_size_max):
                mask_size = torch.ones(1,) * max_mask_size * cutout_size_min
            else:                   
                mask_size = torch.randint(low=int(max_mask_size * cutout_size_min), 
                                        high=int(max_mask_size * cutout_size_max), size=(1,))
            # mask_size = int(max_mask_size * 0.1)
            mask_size_half = mask_size // 2
            buffer = _cutout(buffer, int(kpoint[b][0][n]), int(kpoint[b][1][n]),
                            mask_size, mask_size_half)
        image_new.append(buffer)
    image_new = torch.stack(image_new, dim=0)
    return image_new

def keypoint_cutmix(image_s, kpoint_s,
                    image_t, kpoint_t,
                    mask_size_min=5, mask_size_max=20, p=0.3,
                    batch_size=20, n_pts=None):
    def _cutmix(image1, x1, y1,
                image2, x2, y2, mask_size, mask_size_half):
        # image = np.asarray(image).copy()

        image_new1 = torch.clone(image1)
        image_new2 = torch.clone(image2)

        if torch.rand(1) > p:
            return image1, image2

        _, h1, w1 = image1.shape
        _, h2, w2 = image2.shape

        cx1 = x1
        cy1 = y1
        xmin1 = cx1 - mask_size_half
        ymin1 = cy1 - mask_size_half
        xmax1 = xmin1 + mask_size
        ymax1 = ymin1 + mask_size
        xmin1 = max(0, xmin1)
        ymin1 = max(0, ymin1)
        xmax1 = min(w1, xmax1)
        ymax1 = min(h1, ymax1)

        cx2 = x2
        cy2 = y2
        xmin2 = cx2 - mask_size_half
        ymin2 = cy2 - mask_size_half
        xmax2 = xmin2 + mask_size
        ymax2 = ymin2 + mask_size
        xmin2 = max(0, xmin2)
        ymin2 = max(0, ymin2)
        xmax2 = min(w2, xmax2)
        ymax2 = min(h2, ymax2)

        if (ymax1 - ymin1) != (ymax2 - ymin2) or (xmax1 - xmin1) != (xmax2 - xmin2):
            return image1, image2

        image_new1[:, ymin1:ymax1, xmin1:xmax1] = image2[:, ymin2:ymax2, xmin2:xmax2]
        image_new2[:, ymin2:ymax2, xmin2:xmax2] = image1[:, ymin1:ymax1, xmin1:xmax1]
        return image_new1, image_new2

    image_new1 = []
    image_new2 = []
    for b in range(batch_size):

        buffer1 = image_s[b]
        buffer2 = image_t[b]

        # kpoint_n = len(kpoint_s)
        # non_zero_index_s = torch.nonzero(kpoint_s[b][0])
        # non_zero_index_t = torch.nonzero(kpoint_t[b][0])
        non_zero_index_s = torch.arange(n_pts[b])
        non_zero_index_t = torch.arange(n_pts[b])

        for n in range(len(non_zero_index_t)):
            mask_size = torch.randint(low=mask_size_min, high=mask_size_max, size=(1,))
            mask_size_half = mask_size // 2
            buffer1, buffer2 = _cutmix(buffer1, int(kpoint_s[b][0][n]), int(kpoint_s[b][1][n]),
                                    buffer2, int(kpoint_t[b][0][n]), int(kpoint_t[b][1][n]),
                                    mask_size, mask_size_half)
        image_new1.append(buffer1)
        image_new2.append(buffer2)
    image_new1 = torch.stack(image_new1, dim=0)
    image_new2 = torch.stack(image_new2, dim=0)
    return image_new1, image_new2


def transform_by_grid(src:torch.Tensor,
                        theta=None,
                        mode='aff',
                        interpolation_mode='bilinear',
                        padding_factor=1.0,
                        crop_factor=1.0,
                        use_mask=True):
    mode_list = []
    if mode == 'aff' or mode == 'tps':
        mode_list=[mode]
        theta_list = [theta]
    if mode == 'afftps':
        mode_list = ['aff', 'tps']
        theta_list = [theta[:,:6], theta[:,6:]]
    for i in range(len(mode_list)):
        theta = theta_list[i].float()
        sampling_grid = generate_grid(src.size(), theta, mode=mode_list[i])
        sampling_grid = sampling_grid.cuda()
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        src = F.grid_sample(src, sampling_grid.float(), align_corners=False, mode=interpolation_mode)
        mask = torch.autograd.Variable(torch.ones(src.size())).cuda()
        if use_mask:
            mask = F.grid_sample(mask, sampling_grid)
            mask[mask < 0.9999] = 0
            mask[mask > 0] = 1
            src = mask*src
    return src

#Return transformed grid#
def generate_grid(img_size, theta=None, mode='aff'):
    out_h, out_w = img_size[2], img_size[3]
    gridGen = TpsGridGen(out_h, out_w)

    if mode == 'aff':
        return F.affine_grid(theta.view(-1,2,3), img_size)
    elif mode == 'tps':
        return gridGen(theta.view(-1,18,1,1))
    else:
        raise NotImplementedError

"""
https://github.com/PruneTruong/GLU-Net
"""
class TpsGridGen(nn.Module):
    """
    Adopted version of synthetically transformed pairs dataset by I.Rocco
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        # return torch.cat((points_X_prime, points_Y_prime), 3)
        return torch.cat((points_X_prime, points_Y_prime), 3).cuda()
