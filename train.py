import argparse
import json
import nltk
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import data_processing
import models


def apply_decoder(decoder, x, hidden, encoder_hiddens, lemmas, mode="simple"):
    if mode == "simple":
        output, hidden = decoder(x, hidden)
        log_p = F.log_softmax(output, dim=-1)
        return output, hidden, log_p
    elif mode == "attention":
        output, hidden = decoder(x, hidden, encoder_hiddens)
        log_p = F.log_softmax(output, dim=-1)
        return output, hidden, log_p
    elif mode == "pointer":
        output, hidden = decoder(x, hidden, encoder_hiddens, lemmas)
        log_p = torch.log(output)
        return output, hidden, log_p
    else:
        raise ValueError("'{}' is not a valid value for 'mode'".format(mode))


def eval_model(
        word_encoder,
        tags_encoder,
        decoder,
        test_sampler,
        use_cuda,
        decoder_type):
    
    accuracy = []
    levenstein = []
    losses = []
    
    mapper = test_sampler.mapper
    for i, (lemmas, _, forms, tags) in enumerate(test_sampler.data_iter()):
        parsed_forms = mapper.parse_words(forms.numpy())

        if use_cuda:
            lemmas = lemmas.cuda()
            forms = forms.cuda()
            tags = tags.cuda()

        batch_size = lemmas.size(0)
        
        all_hiddens, last_hidden = word_encoder(lemmas)
        tags_encoded = tags_encoder(tags)
        hidden = torch.cat((last_hidden, tags_encoded), dim=-1)
        loss = 0.0
        
        words = []
        end_words = [False] * lemmas.size(0)
        output = torch.LongTensor([mapper.letters_mapping["BEG"]] * batch_size)
        words.append(output.numpy())


        if use_cuda:
            output = output.cuda()

        for j in range(forms.size(1) - 1):
            output, hidden, log_p = apply_decoder(decoder, output, hidden, all_hiddens, lemmas, decoder_type)
            loss += F.nll_loss(log_p, forms[:, j + 1])
            output = torch.argmax(output, -1)

            if use_cuda:
                words.append(output.cpu().numpy())
            else:
                words.append(output.numpy())

            end_words = [end_words[i] or words[-1][i] == mapper.letters_mapping["END"] for i in range(len(end_words))]
            if all(end_words):
                break

        words = np.array(words).T
        words = mapper.parse_words(words)

        accuracy.append(np.mean([word == form for word, form in zip(words, parsed_forms)]))
        levenstein.append(np.mean([nltk.edit_distance(word, form) for word, form in zip(words, parsed_forms)]))
        losses.append(loss.item() / lemmas.size(1))
    
    return np.mean(losses), np.mean(accuracy), np.mean(levenstein)


def train_epoch(
        n_epoch,
        word_encoder,
        tags_encoder,
        decoder,
        train_sampler,
        test_sampler,
        tags_opt,
        words_opt,
        decoder_opt,
        test_eval_every,
        use_cuda,
        decoder_type):

    for i, (lemmas, _, forms, tags) in enumerate(train_sampler.data_iter()):
        if use_cuda:
            lemmas = lemmas.cuda()
            forms = forms.cuda()
            tags = tags.cuda()

        batch_size = lemmas.size(0)

        tags_opt.zero_grad()
        words_opt.zero_grad()
        decoder_opt.zero_grad()
        
        all_hiddens, last_hidden = word_encoder(lemmas)
        tags_encoded = tags_encoder(tags)
        hidden = torch.cat((last_hidden, tags_encoded), dim=-1)
        loss = 0.0
        
        if i % 5 == 0:
            for j in range(forms.size(1) - 1):
                output, hidden, log_p = apply_decoder(decoder, forms[:, j], hidden, all_hiddens, lemmas, decoder_type)
                loss += F.nll_loss(log_p, forms[:, j + 1])
        else:
            output = torch.LongTensor([mapper.letters_mapping["BEG"]] * batch_size)
            if use_cuda:
                output = output.cuda()
            for j in range(forms.size(1) - 1):
                output, hidden, log_p = apply_decoder(decoder, output, hidden, all_hiddens, lemmas, decoder_type)
                loss += F.nll_loss(log_p, forms[:, j + 1])
                output = torch.argmax(output, -1)

        loss.backward()
        
        tags_opt.step()
        words_opt.step()
        decoder_opt.step()
        
        log_line = "Epoch {epoch}. Iteration {current}/{total}\tTrain loss = {loss:.6f}".format(
            epoch=n_epoch,
            current=i + 1,
            total=len(train_sampler),
            loss=loss.item() / lemmas.size(1)
        )
        
        if (i + 1) % test_eval_every == 0:
            test_loss, accuracy, levenstein = eval_model(
                word_encoder,
                tags_encoder,
                decoder, 
                test_sampler,
                use_cuda,
                decoder_type
            )

            log_line += "\tTest loss = {loss}\tTest accuracy = {accuracy}\tTest edit-distance = {levenstein}".format(
                loss=test_loss,
                accuracy=accuracy,
                levenstein=levenstein
            )
        
        print(log_line)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-data", required=True, type=argparse.FileType("r"), help="Input train data JSON")
    parser.add_argument("-v", "--val-data", required=True, type=argparse.FileType("r"), help="Input validate data JSON")
    parser.add_argument("-m", "--mapper", required=True, type=str, help="Mapper")
    parser.add_argument("-n", "--n-epochs", required=True, type=int, help="Number of epochs to train")
    parser.add_argument("-b", "--batch-size", required=True, type=int, help="Batch size")
    parser.add_argument("-e", "--eval-every", required=True, type=int, help="How often to eval")
    parser.add_argument("-l", "--first-log-line", default="NEW LOG", type=str, help="First line of a log")
    parser.add_argument("-o", "--output-path", required=True, type=str, help="Output path for saving models weights")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for Adam optimizer")
    parser.add_argument("--decoder-type", type=str, choices=["simple", "attention", "pointer"], help="Decoder type")
    parser.add_argument("--use-cuda", action="store_true", help="Use cuda")
    args = parser.parse_args()

    train = json.load(args.train_data)
    val = json.load(args.val_data)
    mapper = data_processing.Mapper.from_json(args.mapper)

    train_sampler = data_processing.InflectionSampler(train, mapper, batch_size=args.batch_size)
    val_sampler = data_processing.InflectionSampler(val, mapper, batch_size=args.batch_size)

    print(args.first_log_line)
    sys.stdout.flush()

    word_encoder = models.WordEncoder(mapper.n_letters, 300, 300)
    tags_encoder = models.TagsEncoder(mapper.n_features, 150, 300)
    if args.decoder_type == "simple":
        decoder = models.SimpleDecoder(mapper.n_letters, 300, 600)
    elif args.decoder_type == "attention":
        decoder = models.AttentionDecoder(mapper.n_letters, 300, 600, 300, 300)
    elif args.decoder_type == "pointer":
        decoder = models.PointerDecoder(mapper.n_letters, 300, 600, 300, 300, args.use_cuda)

    if args.use_cuda:
        word_encoder.cuda()
        tags_encoder.cuda()
        decoder.cuda()

    word_opt = Adam(word_encoder.parameters(), lr=args.lr)
    tags_opt = Adam(tags_encoder.parameters(), lr=args.lr)
    decoder_opt = Adam(decoder.parameters(), lr=args.lr)

    for i in range(args.n_epochs):
        train_epoch(
            n_epoch=i + 1,
            word_encoder=word_encoder,
            tags_encoder=tags_encoder,
            decoder=decoder,
            train_sampler=train_sampler,
            test_sampler=val_sampler,
            tags_opt=tags_opt,
            words_opt=word_opt,
            decoder_opt=decoder_opt,
            test_eval_every=args.eval_every,
            use_cuda=args.use_cuda,
            decoder_type=args.decoder_type
        )

        output_path = os.path.join(args.output_path, "epoch_{}_weights.pth".format(i + 1))
        torch.save({
            "word_encoder": word_encoder.state_dict(),
            "tags_encoder": tags_encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "word_opt": word_opt.state_dict(),
            "tags_opt": tags_opt.state_dict(),
            "decoder_opt": decoder_opt.state_dict()           
        }, output_path)

    print("END LOG")