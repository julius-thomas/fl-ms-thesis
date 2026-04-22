import os
import sys
import time
import torch
import argparse
import traceback

from importlib import import_module
from torch.utils.tensorboard import SummaryWriter

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model 




def main(args, writer):
    """Main program to run federated learning.
    
    Args:
        args: user input arguments parsed by argparser
        writer: `torch.utils.tensorboard.SummaryWriter` instance for TensorBoard tracking
    """
    # set seed for reproducibility
    set_seed(args.seed)

    # get dataset
    server_dataset, client_datasets = load_dataset(args)

    # check all args before FL
    args = check_args(args)
    
    # get model
    model, args = load_model(args)

    # create central server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}Server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model=model)
    
    # federated learning
    for curr_round in range(1, args.R + 1):
        ## update round indicator
        server.round = curr_round

        ## update after sampling clients randomly
        selected_ids = server.update() 

        ## evaluate on clients not sampled (for measuring generalization performance)
        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            server.evaluate(excluded_ids=selected_ids)
    else:
        ## wrap-up
        server.finalize()



if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    #####################
    # Drift adaptation  #
    #####################
    parser.add_argument('--drift_adaptation', help='use drift adaptation for learning rate', action='store_true')
    parser.add_argument('--drift_adaptation_mode', help='type of LR adaptation; `original` (model-based, server-side), `custom` (loss-based, per-client)', type=str, default='custom')
    parser.add_argument('--b1', help='EMA coefficient for mean tracking', type=float, choices=[Range(0.01, 0.99)], default=0.5)
    parser.add_argument('--b2', help='EMA coefficient for variance tracking', type=float, choices=[Range(0.01, 0.99)], default=0.5)
    parser.add_argument('--b3', help='EMA coefficient for variance ratio tracking', type=float, choices=[Range(0.01, 0.99)], default=0.5)

    #####################
    #   Concept drift   #
    #####################
    parser.add_argument('--concept_drift', help='introduce concept drift to the selected dataset', action='store_true')
    parser.add_argument('--drift_duration', help='duration of the concept drift in rounds', type=int, default=50)
    parser.add_argument('--drift_start', help='round number in which the concept drift gets initiated', type=int, default=50)
    parser.add_argument('--drift_mode', help='type of concept drift; `soft` (mild blur), `hard` (strong blur), `sudden` (label swap), `custom` (dataset-specific real-world drift)', type=str, default='hard')

    #####################
    #  Active sampling  #
    #####################
    parser.add_argument('--active_sampling', help='apply loss-based active client sampling', action='store_true')
    parser.add_argument('--sampling_fraction', help='fraction of sampled clients used for active sampling', type=float, default=0.3)
    parser.add_argument('--sampling_type', help='type of client selection; `max` (top-K by loss), `stoch` (Boltzmann softmax)', type=str, default='max')
    parser.add_argument('--temp', help='Boltzmann temperature for stochastic sampling', type=float, default=0.5)
    parser.add_argument('--candidate_sampling', help='how to form the initial candidate set before active sampling; `uniform` (random) or `ucb` (UCB1 over all clients)', type=str, default='uniform', choices=['uniform', 'ucb'])
    parser.add_argument('--ucb_c', help='UCB exploration constant (only used when --candidate_sampling ucb)', type=float, default=1.0)
    parser.add_argument('--ucb_signal', help='reward signal used to update UCB mu_i; `loss` (pre-training loss on candidate pool), `delta_loss` (L_before - L_after on final-selected), `param_drift` (||w_local - w_global|| on final-selected)', type=str, default='loss', choices=['loss', 'delta_loss', 'param_drift'])
    parser.add_argument('--ucb_window', help='sliding-window size for UCB reward memory (number of rounds); 0 = infinite (plain UCB, keep all history)', type=int, default=0)

    #####################
    # Default arguments #
    #####################
    parser.add_argument('--exp_name', help='name of the experiment', type=str, required=True)
    parser.add_argument('--seed', help='global random seed', type=int, default=5959)
    parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`, `mps`', type=str, default='cpu')
    parser.add_argument('--bf16', help='use bfloat16 mixed precision (requires CUDA)', action='store_true')
    parser.add_argument('--data_path', help='path to save & read raw data', type=str, default='./data')
    parser.add_argument('--log_path', help='path to save logs', type=str, default='./log')
    parser.add_argument('--result_path', help='path to save results', type=str, default='./result')
    parser.add_argument('--no_save_model', help='skip writing the final global model checkpoint (.pt) to --result_path', action='store_true')
    parser.add_argument('--no_save_results', help='skip writing the final results JSON to --result_path', action='store_true')
    parser.add_argument('--use_tb', help='use TensorBoard for log tracking (if passed)', action='store_true')
    parser.add_argument('--tb_port', help='TensorBoard port number (valid only if `use_tb`)', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address (valid only if `use_tb`)', type=str, default='0.0.0.0')
    
    #####################
    # Dataset arguments #
    #####################
    ## dataset configuration arguments
    parser.add_argument('--dataset', help='''name of dataset to use for an experiment (NOTE: case sensitive)
    - image classification datasets in `torchvision.datasets`,
    - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
    - among [ TinyImageNet | CINIC10 | SpeechCommands | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM | MIMIC4 ]
    ''', type=str, required=True)
    parser.add_argument('--test_size', help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local holdout set)', type=float, choices=[Range(-1, 1.)], default=0.2)
    parser.add_argument('--rawsmpl', help='a fraction of raw data to be used (valid only if one of `LEAF` datasets is used)', type=float, choices=[Range(0., 1.)], default=1.0)
    
    ## data augmentation arguments
    parser.add_argument('--resize', help='resize input images (using `torchvision.transforms.Resize`)', type=int, default=None)
    parser.add_argument('--crop', help='crop input images (using `torchvision.transforms.CenterCrop` (for evaluation) and `torchvision.transforms.RandomCrop` (for training))', type=int, default=None)
    parser.add_argument('--imnorm', help='normalize channels with mean 0.5 and standard deviation 0.5 (using `torchvision.transforms.Normalize`, if passed)', action='store_true')
    parser.add_argument('--randrot', help='randomly rotate input (using `torchvision.transforms.RandomRotation`)', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip input horizontaly (using `torchvision.transforms.RandomHorizontalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randvf', help='randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help='randomly change the brightness and contrast (using `torchvision.transforms.ColorJitter`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--precomputed', help='use precomputed resized PNGs from CheXpert/png_<size>/. Implies --resize <size> and --imnorm; skips the runtime Resize transform. Only valid for CheXpert.', type=int, nargs='?', const=224, default=None)
    parser.add_argument('--num_workers', help='DataLoader worker processes per client loader. 0 keeps loading in the main process (safe at large K). Raise only when K is small enough that K*num_workers stays well under OS fd/process limits.', type=int, default=0)

    ## statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''type of data split scenario
    - `iid`: statistically homogeneous setting,
    - `unbalanced`: unbalanced in sample counts across clients,
    - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
    - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
    - `pre`: pre-defined data split scenario,
    - `custom`: dataset-specific real-world heterogeneity (e.g. Dirichlet-over-care-unit for MIMIC4)
    ''', type=str, choices=['iid', 'unbalanced', 'patho', 'diri', 'pre', 'custom'], required=True)
    parser.add_argument('--mincls', help='the minimum number of distinct classes per client (valid only if `split_type` is `patho` or `diri`)', type=int, default=2)
    parser.add_argument('--cncntrtn', help='a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)', type=float, default=0.1)
    
    
    ###################
    # Model arguments #
    ###################
    ## model
    parser.add_argument('--model_name', help='a model to be used (NOTE: case sensitive)', type=str,
        choices=[
            'TwoNN', 'TwoCNN', 'SimpleCNN', 'FEMNISTCNN', 'Sent140LSTM',
            'LeNet', 'MobileNet', 'SqueezeNet',
            'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
            'ResNet10', 'ResNet18', 'ResNet34',
            'ShuffleNet', 'MobileNeXt', 'SqueezeNeXt', 'MobileViT', 
            'StackedLSTM', 'StackedTransformer', 'LogReg', 'M5',
            'DistilBert', 'SqueezeBert', 'MobileBert'
        ],
        required=True
    )
    parser.add_argument('--hidden_size', help='hidden channel size for vision models, or hidden dimension of language models', type=int, default=64)
    parser.add_argument('--dropout', help='dropout rate', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--use_model_tokenizer', help='use a model-specific tokenizer (if passed)', action='store_true')
    parser.add_argument('--use_pt_model', help='use a pre-trained model weights for fine-tuning (if passed)', action='store_true')
    parser.add_argument('--seq_len', help='maximum sequence length used for text datasets', type=int, default=512)
    parser.add_argument('--num_layers', help='number of layers in recurrent cells', type=int, default=2)
    parser.add_argument('--num_embeddings', help='size of an embedding layer', type=int, default=1000)
    parser.add_argument('--embedding_size', help='output dimension of an embedding layer', type=int, default=512)
    parser.add_argument('--init_type', help='weight initialization method', type=str, default='xavier', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'truncnorm', 'none'])
    parser.add_argument('--init_gain', type=float, default=1.0, help='magnitude of variance used for weight initialization')
    
    ######################
    # Learning arguments #
    ######################
    ## federated learning settings
    parser.add_argument('--algorithm', help='federated learning algorithm to be used', type=str,
        choices=['fedavg', 'fedsgd', 'fedprox', 'fedavgm'], 
        required=True
    )
    parser.add_argument('--eval_type', help='''the evaluation type of a model trained from FL algorithm
    - `local`: evaluation of personalization model on local hold-out dataset  (i.e., evaluate personalized models using each client\'s local evaluation set)
    - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the global model using separate holdout dataset located at the server)
    - 'both': combination of `local` and `global` setting
    ''', type=str,
        choices=['local', 'global', 'both'],
        required=True
    )
    parser.add_argument('--eval_fraction', help='fraction of randomly selected (unparticipated) clients for the evaluation (valid only if `eval_type` is `local` or `both`)', type=float, choices=[Range(1e-8, 1.)], default=1.)
    parser.add_argument('--eval_every', help='frequency of the evaluation (i.e., evaluate peformance of a model every `eval_every` round)', type=int, default=1)
    parser.add_argument('--eval_metrics', help='metric(s) used for evaluation', type=str,
        choices=[
            'acc1', 'acc5', 'auroc', 'auprc', 'youdenj', 'f1', 'precision', 'recall',
            'seqacc', 'mse', 'mae', 'mape', 'rmse', 'r2', 'd2',
            'mlacc', 'mlauroc'
        ], nargs='+', required=True
    )
    parser.add_argument('--K', help='number of total cilents participating in federated training', type=int, default=100)
    parser.add_argument('--R', help='number of total federated learning rounds', type=int, default=1000)
    parser.add_argument('--C', help='sampling fraction of clients per round (full participation when 0 is passed)', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--E', help='number of local epochs', type=int, default=5)
    parser.add_argument('--B', help='local batch size (full-batch training when zero is passed)', type=int, default=10)
    parser.add_argument('--beta1', help='server momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    
    # optimization arguments
    parser.add_argument('--no_shuffle', help='do not shuffle data when training (if passed)', action='store_true')
    parser.add_argument('--optimizer', help='type of optimization method (NOTE: should be a sub-module of `torch.optim`, thus case-sensitive)', type=str, default='SGD', required=True)
    parser.add_argument('--max_grad_norm', help='a constant required for gradient clipping', type=float, choices=[Range(0., float('inf'))], default=0.)
    parser.add_argument('--weight_decay', help='weight decay (L2 penalty)', type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--momentum', help='momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--lr', help='learning rate for local updates in each client', type=float, choices=[Range(0., 100.)], default=0.01, required=True)
    parser.add_argument('--lr_decay', help='decay rate of learning rate', type=float, choices=[Range(0., 1.)], default=1.0)
    parser.add_argument('--lr_decay_step', help='intervals of learning rate decay', type=int, default=20)
    parser.add_argument('--criterion', help='objective function (NOTE: should be a submodule of `torch.nn`, thus case-sensitive)', type=str, required=True)
    parser.add_argument('--mu', help='constant for proximity regularization term (valid only if the algorithm is `fedprox`)', type=float, choices=[Range(0., 1e6)], default=0.01)

    # parse arguments
    args = parser.parse_args()

    # --precomputed implies --resize <size> --imnorm (those preproc steps are baked
    # into the cached PNGs, so re-applying Resize at runtime would be wasted work).
    if args.precomputed is not None:
        args.resize = args.precomputed
        args.imnorm = True

    # make path for saving losses & metrics & models (skip entirely if nothing will be written)
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}_{curr_time}')
    if not (args.no_save_model and args.no_save_results):
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
        
    # make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    # initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)
    
    # check TensorBoard execution
    tb = TensorBoardRunner(args.log_path, args.tb_host, args.tb_port) if args.use_tb else None

    # define writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_path, f'{args.exp_name}_{curr_time}'), filename_suffix=f'_{curr_time}')

    # run main program
    torch.autograd.set_detect_anomaly(True)
    try:
        main(args, writer)
        if args.use_tb:
            tb.finalize()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        if args.use_tb:
            tb.interrupt()
        sys.exit(1)
