import skimage

SR_image = skimage.io.imread('../data/test/HR_image.png')
GROUND_TRUTH = skimage.io.imread('../data/ground_truth/202598.png')

# MSE
mse = skimage.measure.compare_mse(GROUND_TRUTH, SR_image)

# PSNR
psnr = skimage.measure.compare_psnr(GROUND_TRUTH, SR_image, data_range=255)

# SSMI
ssmi = skimage.measure.compare_ssim(GROUND_TRUTH, SR_image)

message = 'mse={}, '.format(mse) + 'psnr={}, '.format(
    psnr) + 'ssmi={}.'.format(ssmi)

print(message)