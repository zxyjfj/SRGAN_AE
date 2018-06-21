from skimage import io, measure

SR_image = io.imread('../data/test/HR_Image.png')
GROUND_TRUTH = io.imread('../data/ground_truth/202598.png')

# MSE
mse = measure.compare_mse(GROUND_TRUTH, SR_image)

# PSNR
psnr = measure.compare_psnr(GROUND_TRUTH, SR_image, data_range=255)

# SSMI
ssmi = measure.compare_ssim(GROUND_TRUTH, SR_image, multichannel=True)

message = 'mse={:5f}, '.format(mse) + 'psnr={:5f}, '.format(
    psnr) + 'ssmi={:5f}.'.format(ssmi)

print(message)