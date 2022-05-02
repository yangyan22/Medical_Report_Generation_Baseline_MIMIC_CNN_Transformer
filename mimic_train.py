import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from modules.models import Model


def parse_agrs():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/public/home/yangyan/YangYan/images_downsampled',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/public/home/yangyan/YangYan/MIMIC_R2_mesh_token.json',
                        help='the path to the directory containing the data.')

    # Data input settings
    # parser.add_argument('--image_dir', type=str, default='/media/camlab1/doc_drive/MIMIC_data/images_downsampled')
    # parser.add_argument('--ann_path', type=str,
    #                     default='/media/camlab1/doc_drive/MIMIC_data/R2/MIMIC_R2_token_mesh.json')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='densenet121', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers_encoder', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--num_layers_decoder', type=int, default=6, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # Sample related  
    parser.add_argument('--sample_method', type=str, default='greedy', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=1, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=3, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='./results/mimic_50/', help='the patch to save the models.')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_1', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=10, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=1,
                        help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--mode', type=str, default="CNN-Transformer", help='CNN-Transformer or CNN-TD')
    parser.add_argument('--seed', type=int, default=456789, help='.')
    parser.add_argument('--resume', type=str)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--gpus', type=str, default='2, 3')
    parser.add_argument('--gpus_id', type=list, default=[0, 1])
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # build model architecture
    model = Model(args, tokenizer)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, drop_last=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False, drop_last=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
