# data path and log path
TRAINING_DATA_PATH = '../data/train'
INFERENCES_SAVE_PATH = '../data/imgs'
TRAINING_SUMMARY_PATH = '../summary'
CHECKPOINTS_PATH = '../checkpoints'

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
LEARN_RATE = 8e-4
NUM_EPOCH = 100