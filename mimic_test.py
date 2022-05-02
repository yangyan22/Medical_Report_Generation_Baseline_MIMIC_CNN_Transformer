import argparse
from tqdm import tqdm
import torch
import csv
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.models import Model
from modules.metrics import compute_scores
import json
import os
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


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
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')  # Batch_size should be appropriate ! multiple GPUs should be able to divide the data

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default="densenet121", help='the visual extractor to be used.')
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
    parser.add_argument('--n_gpu', type=int, default=3, help='the sample number per image.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='greedy')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')
    parser.add_argument('--restore_dir', type=str, default='./results/mimic_50/', help='the patch to save the models.')
    parser.add_argument('--mode', type=str, default="CNN-TD", help='CNN-Transformer or CNN-TD')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--gpus', type=str, default='2, 3')
    parser.add_argument('--gpus_id', type=list, default=[0, 1])
    parser.add_argument('--checkpoint_dir', type=str, default='current_checkpoint.pth', help='the patch to save the models.')
    parser.add_argument('--gn', type=str, default='_gn_18_2.csv')

    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    model_path = os.path.join(args.restore_dir, args.checkpoint_dir)
    if args.n_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 0, 1, 2, 3 single-GPU
        device = torch.device('cuda:0')  # always: 0
        checkpoint = torch.load(model_path)
        print("Checkpoint loaded from epoch {}".format(checkpoint['epoch']))
        model = Model(args, tokenizer)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)  # the position of environ is important!
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # select multiple GPUs
        device = torch.device('cuda:0')  # always: 0
        checkpoint = torch.load(model_path)
        print("Checkpoint loaded from epoch {}".format(checkpoint['epoch']))
        model = Model(args, tokenizer)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=args.gpus_id)  # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" device_ids=[0, 1] 1 equals to GPU: 2
        print("GPUs_Used: {}".format(args.n_gpu))

    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        test_gts, test_res = [], []
        record_path_gt = os.path.join(args.restore_dir, args.dataset_name + '_gt.csv')
        record_path_gn = os.path.join(args.restore_dir, args.dataset_name + args.gn)
        with open(record_path_gt, "w", newline="") as f_gt:
            file_gt = csv.writer(f_gt)
            with open(record_path_gn, "w", newline="") as f_gn:
                file_gn = csv.writer(f_gn)
                with tqdm(desc='Epoch %d - Testing', unit='it', total=len(test_dataloader)) as pbar:
                    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_dataloader):
                        images, reports_ids, reports_masks = images.to(device), reports_ids.to(
                            device), reports_masks.to(device)
                        output = model(images, mode='sample')
                        if args.n_gpu > 1:
                            reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                            ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        else:
                            reports = model.tokenizer.decode_batch(output.cpu().numpy())
                            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        test_gts.extend(ground_truths)  # gt
                        test_res.extend(reports)  # gn
                        pbar.update()

                        i = 0
                        for id in images_id:
                            # print(id)
                            # print('Pred Sent.{}'.format(reports[i]))
                            # print('Real Sent.{}'.format(ground_truths[i]))
                            # print('\n')
                            # whether to save the results in csv files
                            # file_gt.writerow([ground_truths[i]])
                            file_gn.writerow([reports[i]])
                            i = i + 1
                test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                   {i: [re] for i, re in enumerate(test_res)})
        print(test_met)
        # whether to save the results in json files
        # with open(os.path.join(args.restore_dir, 'data_gt.json'), 'w') as f:
        #     json.dump(test_gts, f)
        # with open(os.path.join(args.restore_dir, 'data_re.json'), 'w') as f:
        #     json.dump(test_res, f)
        # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
