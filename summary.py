import utilities
import os
import matplotlib.pyplot as plt
import torch
import pytorch_ssim
import numpy as np

EVALUATE_DIR = "../evaluate_min/"

if __name__ == '__main__':
    total_psnr = 0.0
    total_ssim = 0.0
    for x in range(0, 100):
        real = plt.imread(EVALUATE_DIR + str(x) + "_real.png")
        recon = plt.imread(EVALUATE_DIR + str(x) + "_recon.png")

        psnr_val = utilities.cal_psnr(real * 255.0, recon * 255.0)
        real = np.transpose(real, [2, 0, 1])
        real = np.expand_dims(real, 0)
        recon = np.transpose(recon, [2, 0, 1])
        recon = np.expand_dims(recon, 0)

        ssim_val = pytorch_ssim.ssim(torch.from_numpy(real), torch.from_numpy(recon))
        total_psnr = psnr_val + total_psnr
        total_ssim = ssim_val + total_ssim
    print(total_psnr / 100.0, total_ssim / 100.0)
