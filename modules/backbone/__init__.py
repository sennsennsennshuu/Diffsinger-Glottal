from .wavenet import WaveNet
from .lynxnet import LYNXNet

BACKBONES = {
    "wavenet": WaveNet,
    "lynxnet": LYNXNet,
}
