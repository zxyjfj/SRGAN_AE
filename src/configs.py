# data path and log path
TRAIN_DATA_PATH = '../data/train'
TEST_DATA_PATH = '../data/test'
INFERENCES_SAVE_PATH = '../imgs'
TRAIN_SUMMARY_PATH = '../summary'
CHECKPOINTS_PATH = '../checkpoints'

reconstruction_loss_weight = 40

# 避免出现 log(0)
EPS = 1e-12
BATCH_SIZE = 64
# 低分辨图片的大小
INPUT_SIZE = 32
SCALE_FACTOR = 4
# 高分辨图片的大小
LABEL_SIZE = SCALE_FACTOR * INPUT_SIZE
NUM_CHENNELS = 3
# 需要保存的模型的数
MAX_CKPT_TO_KEEP = 50
LEARN_RATE = 5e-5
NUM_EPOCH = 100

# patch generation
PATCH_SIZE = 128

# data queue
MIN_QUEUE_EXAMPLES = 1024
NUM_PROCESS_THREADS = 3

# data argumentation
MAX_RANDOM_BRIGHTNESS = -1
# RANDOM_CONTRAST_RANGE = [0.8, 1.2]
RANDOM_CONTRAST_RANGE = [0.8]
GAUSSIAN_NOISE_STD = -1
JPEG_NOISE_LEVEL = -1