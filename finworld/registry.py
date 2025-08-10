
from mmengine.registry import Registry

DATASET = Registry('data', locations=['finworld.data'])
ENVIRONMENT = Registry('environment', locations=['finworld.environment'])
COLLATE_FN = Registry('collate_fn', locations=['finworld.data'])
DATALOADER = Registry('dataloader', locations=['finworld.data'])
DOWNLOADER = Registry('downloader', locations=['finworld.downloader'])
PROCESSOR = Registry('processor', locations=['finworld.processor'])
FACTOR = Registry('factor', locations=['finworld.factor'])
SCALER = Registry('scaler', locations=['finworld.data'])

EMBED = Registry('embed', locations=['finworld.models'])
ENCODER = Registry('encoder', locations=['finworld.models'])
DECODER = Registry('decoder', locations=['finworld.models'])
PROVIDER = Registry('provider', locations=['finworld.provider'])
DIFFUSION = Registry('diffusion', locations=['finworld.diffusion'])
MODEL = Registry('model', locations=['finworld.models'])
QUANTIZER = Registry('quantizer', locations=['finworld.models'])
PREDICTOR = Registry('predictor', locations=['finworld.models'])
AGENT = Registry('agent', locations=['finworld.agent'])
TOOL = Registry('tool', locations=['finworld.tools'])

PLOT = Registry('plot', locations=['finworld.plot'])

DOWNSTREAM = Registry('downstream', locations=['finworld.downstream'])
REDUCER = Registry('reducer', locations=['finworld.reducer'])

LOSS = Registry(name='loss_func', locations=['finworld.loss'])
METRIC = Registry(name='metric', locations=['finworld.metric'])
OPTIMIZER = Registry(name='optimizer', locations=['finworld.optimizer'])
SCHEDULER = Registry(name='scheduler', locations=['finworld.scheduler'])
TRAINER = Registry(name='trainer', locations=['finworld.trainer'])
EVALUATOR = Registry(name='evaluator', locations=['finworld.evaluator'])

REWARDS = Registry(name='rewards', locations=['finworld.rewards'])

TASK = Registry(name='task', locations=['finworld.task'])
