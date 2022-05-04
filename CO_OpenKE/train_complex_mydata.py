import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gethered')
parser.add_argument('--dim', type=int, default=200)
args = parser.parse_args()

dataset = args.dataset
dim = args.dim

print('model: complEx')

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "benchmarks/{}/".format(dataset),
	# nbatches = 100,
	batch_size=512,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("benchmarks/{}/".format(dataset), "link")

# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = dim
)

# define the loss function
model = NegativeSampling(
	model = complEx,
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(),
	regul_rate = 1.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 0.5, use_gpu = True, opt_method = "adagrad")
trainer.run()
# complEx.save_checkpoint('../COKG/complEx_{}_dim{}.ckpt'.format(dataset, dim))
complEx.save_parameters('../../../data/COKG_data/complEx.vec')

# # test the model
# complEx.load_checkpoint('./checkpoint/complEx_{}_dim400.ckpt'.format(dataset))
# tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)
