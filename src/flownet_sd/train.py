from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_sd import FlowNetSD

# Create a new network
net = FlowNetSD()

# Load a batch of data
input_a, input_b, flow = load_batch(
    FLYING_CHAIRS_DATASET_CONFIG, 'train', net.global_step)

# Train on the data
net.train(
    log_dir='./logs/flownet_sd',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow
)
