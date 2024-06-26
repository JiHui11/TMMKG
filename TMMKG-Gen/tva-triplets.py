from __future__ import print_function
from dataloader import Dataset
import argparse
import torch
import numpy as np
from utils import *
import random
from ast_model import AudioTaggingModel
from alignment import alignment_triples_vat, alignment_triples_va, alignment_triples_vt, load_word_embeddings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MMKG_generation')
parser.add_argument('--dir_video', type=str, default="../data/visual_feature.pkl", help='visual features')
parser.add_argument('--dir_audio', type=str, default='../data/audio_feature.pkl', help='audio features')
parser.add_argument('--dir_labels', type=str, default='../data/label.pkl', help='labels of dataset')

parser.add_argument('--dir_order_train', type=str, default='../data/train_set.pkl', help='indices of training samples')
parser.add_argument('--dir_order_test', type=str, default='../data/test_set.pkl', help='indices of testing samples')

parser.add_argument('--batch_size', type=int, default=100, help='number of batch size')
parser.add_argument('--threshold', type=float, default=0.095, help='key-parameter for pruning process')
parser.add_argument('--entity_threshold', type=float, default=0.40, help='threshold for entity alignment process')
parser.add_argument('--audios-dir', type=str, default='../data/audios-2000', help='audios')
parser.add_argument('--videos-dir', type=str, default='../data/videos-2000', help='videos')
parser.add_argument('--checkpoint_path', type=str, default='../data/best_audio_model-0.197583.pth', help="the best ast model")
parser.add_argument('--label_csv', type=str, default='../data/labels.csv', help="types of audio labels")
parser.add_argument('--split_audios_folder', type=str, default='../extract_audios', help="split_audios_folder")

parser.add_argument('--status', type=str, default='test', help="the mode of operation: 'train' for training kg, 'test' for testing kg")
parser.add_argument('--text_triples_dir', type=str, default='../data/text_triples_2000.json')
parser.add_argument('--train_visual_triples_dir', type=str, default='../data/train_video_triples.json', help="File path to the JSON containing visual triples for training phase.")
parser.add_argument('--test_visual_triples_dir', type=str, default='../data/test_video_triples.json', help="File path to the JSON containing visual triples for testing phase.")
parser.add_argument('--gt_triples_dir', type=str, default='../data/ground_truth.json')

parser.add_argument('--kgg_mode', type=str, default='two_modality', help="the mode of kg generation")
parser.add_argument('--type_two_mode', type=str, default='vt', help="the types of two modalities formed by combining the three modalities pairwise")

parser.add_argument('--glove_file_path', type=str, default='../data/glove.6B.300d.txt', help="the path of glove")


FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)
torch.cuda.manual_seed_all(FixSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def generate_TMMKG(args, net_model):
    Data = Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, batch_size=args.batch_size, label_dir=args.dir_labels,
                        order_dir=args.dir_order_test, status=args.status) # Status of the dataset, can be 'train' or 'test'

    nb_batch = Data.__len__() // args.batch_size  

    # Read visual triples from JSON file
    # Use args.test_visual_triples_dir for testing and args.train_visual_triples_dir for training
    if args.status == 'train':
        all_visual_triples = read_json(args.train_visual_triples_dir)  # Use training visual triples
    elif args.status == 'test':
        all_visual_triples = read_json(args.test_visual_triples_dir)   # Use testing visual triples

    keys = list(all_visual_triples.keys())
    print(len(keys))
    print('nb_batch:', nb_batch)
    SHUFFLE_SAMPLES = True
    num = 0
    for i in range(nb_batch):
        audio, video, labels, segment_label, segment_avps_gt = Data_train.get_batch(i, SHUFFLE_SAMPLES) 
        for n in range(args.batch_size):
            keys_video = list(video.keys())
            keys_auido = list(audio.keys())
            filename_a = keys_auido[n]
            filename_v = keys_video[n]
            if filename_v not in keys:
                continue
            else:
                num += 1
                audio_input = torch.from_numpy(audio[filename_a]).float().unsqueeze(0).cuda() # [1,t,128]
                video_input = torch.from_numpy(video[filename_v]).float().unsqueeze(0).cuda() # [1,t,7,7,512]
                """temporal interval"""
                fusion, out_prob, cross_att, selected_time_segments, _ = net_model(audio_input, video_input, args.threshold)
                """temporal knowledge graph"""
                kg = extract_dynamic_knowledge_graph(filename_a, filename_v, selected_time_segments, word_embeddings,
                                                          all_text_triples, all_visual_triples, all_gt_triples, args)
                
                # Check if the length of the knowledge graph triplets is less than 3
                # We exclude triplets with length less than 3 because we plan to perform link prediction
                # with a dynamic time window of size 2 in the subsequent steps.
                if len(kg[0]) < 3:
                    print(filename_v)
                    num -= 1
                else:
                    tva_triplets[filename_v] = kg[0]
                    gt_triplets[filename_v] = kg[1]
    print(num)
    print(len(gt_triplets))
    
    output_file_path = f'../data/tva_triples_{args.status}.json'

    with open(output_file_path, 'w') as json_file:
        json.dump(tva_triplets, json_file, indent=4)

    print(f"JSON 文件已保存到 {output_file_path}")

    output_file_path1 = f'../data/gt_triples_{args.status}.json'

    with open(output_file_path1, 'w') as file:
        json.dump(gt_triplets, file, indent=4)

    print(f"JSON 文件已保存到 {output_file_path1}")


def extract_dynamic_knowledge_graph(filename_a, filename_v, selected_time_segments, word_embeddings,
                                        all_text_triples, all_visual_triples, all_gt_triples, args):
    all_triples = []
    extract_audio_segments(filename_a, selected_time_segments, args.audios_dir, args.split_audios_folder)
    audio_tagging_model = AudioTaggingModel(args.checkpoint_path, args.label_csv)
    audio_labels = audio_tagging_model.forward(args.split_audios_folder, filename_a)

    text_triplets = all_text_triples[filename_v]
    visual_triples, gt_triples, segment_indices_list = extract_visual_triples(filename_v, selected_time_segments,
                                                                              args.videos_dir, all_visual_triples,
                                                                              all_gt_triples)

    if args.kgg_mode == "one_modality":
        all_triples = visual_triples
    elif args.kgg_mode == "two_modality":
        if args.type_two_mode == 'vt':
            all_triples = alignment_triples_vt(args, text_triplets, visual_triples, word_embeddings)
        elif args.type_two_mode == 'va':
            all_triples = alignment_triples_va(args, audio_labels, visual_triples, word_embeddings, segment_indices_list)
    elif args.kgg_mode == "three_modality":
        all_triples = alignment_triples_vat(args, text_triplets, audio_labels, visual_triples, word_embeddings, segment_indices_list)
    return all_triples, gt_triples


if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)
    net_model = torch.load('PSP_30_fully_0.095.pt')
    net_model.eval()
    word_embeddings = load_word_embeddings(args.glove_file_path)
    all_text_triples = read_json(args.text_triples_dir)
    all_gt_triples = read_json(args.gt_triples_dir)
    tva_triplets = {}
    gt_triplets = {}
    generate_TMMKG(args, net_model)



