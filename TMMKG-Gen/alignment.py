import torch
import numpy as np
import torch.nn.functional as F


def load_word_embeddings(glove_file_path):
    word_embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            word_embeddings[word] = vector
    return word_embeddings


def process_entity_similarity(args, entities_a, entities_b, word_embeddings, triples):
    threshold = args.entity_threshold
    new_rel = 'related to'
    for entity_a in entities_a:
        if '_' in entity_a:
            token_a = entity_a.split('_')[0]
        else:
            token_a = entity_a  
        for entity_b in entities_b:
            if '_' in entity_b:
                token_b = entity_b.split('_')[0]
            else:
                token_b = entity_b
            if token_a in word_embeddings:
                vector_a = word_embeddings[token_a]
            else:
                lw_token = sorted(token_a.split(' '), key=lambda x: len(x), reverse=True)[0]
                vector_a = word_embeddings[lw_token]

            if token_b in word_embeddings:
                vector_b = word_embeddings[token_b]
            else:
                lw_token = sorted(token_b.split(' '), key=lambda x: len(x), reverse=True)[0]
                vector_b = word_embeddings[lw_token]
            a = torch.tensor(vector_b)
            b = torch.tensor(vector_a)
            similarity = F.cosine_similarity(a, b, dim=0)
            if similarity.item() == 1.0:
                continue
            else:
                if similarity.item() > threshold:
                    new_triple = (entity_a, new_rel, entity_b)
                    if new_triple not in triples:
                        triples.append(new_triple)
    return triples


def alignment_triples_vt(args, text_triplets, video_temporal_triples, word_embeddings):
    entities_a = []
    all_triples = {}
    for triplet in text_triplets:
        head_entity, _, tail_entity = triplet
        if head_entity not in entities_a:
            entities_a.append(head_entity)
        if tail_entity not in entities_a:
            entities_a.append(tail_entity)
    for key, video_triples in video_temporal_triples.items():
        entities_b = []
        triples = video_triples + text_triplets
        for triplet in video_triples:
            head_entity, _, tail_entity = triplet
            if head_entity not in entities_b:
                entities_b.append(head_entity)
            if tail_entity not in entities_b:
                entities_b.append(tail_entity)
        temporal_triples = process_entity_similarity(args, entities_a, entities_b, word_embeddings, triples)

        all_triples[key] = temporal_triples
    return all_triples


def alignment_triples_va(args, audio_labels, video_temporal_triples, word_embeddings, segment_indices_list):
    all_triples = {}
    for key, video_triples in video_temporal_triples.items():
        num = segment_indices_list[key]
        entities_b = []
        entities_b.append(audio_labels[num])
        entities_a = []
        triples = video_triples
        for triplet in video_triples:
            head_entity, _, tail_entity = triplet
            if head_entity not in entities_a:
                entities_a.append(head_entity)
            if tail_entity not in entities_a:
                entities_a.append(tail_entity)

        temporal_triples = process_entity_similarity(args, entities_a, entities_b, word_embeddings, triples)
        all_triples[key]= temporal_triples
    return all_triples


def alignment_triples_vat(args, text_triplets, audio_labels, video_temporal_triples, word_embeddings, segment_indices_list):
    triples_at = alignment_triples_va(args, audio_labels, video_temporal_triples, word_embeddings, segment_indices_list)
    all_triples = alignment_triples_vt(args, text_triplets, triples_at, word_embeddings)
    return all_triples

