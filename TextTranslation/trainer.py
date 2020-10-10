import random
import torch
import torch.nn as nn
from torch import optim
from datasets import readLangs, SOS_token, EOS_token, MAX_LENGTH
from models import EncoderRNN, AttenDecoderRNN, DecoderRNN
from utils import timeSince
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = MAX_LENGTH + 1

lang1 = "en"
lang2 = "cn"
path = "en-cn.txt"
input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
print(len(pairs))
print(input_lang.n_words)
print(input_lang.index2word)

print(output_lang.n_words)
print(output_lang.index2word)

def listTotensor(input_lang, data):
    indexes_in = [input_lang.word2index[word] for word in data.split(" ")]
    indexes_in.append(EOS_token)

    input_tensor = torch.tensor(indexes_in, dtype=torch.long, device=device).view(-1, 1)
    return input_tensor
def tensorsFromPair(pair):
    input_tensor = listTotensor(input_lang, pair[0])
    output_tensor = listTotensor(output_lang, pair[1])

    return (input_tensor, output_tensor)

def loss_func(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_len = input_tensor.size(0)
    output_len = output_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    for ei in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device)

    use_teacher_forcing = True if random.random() < 0.5 else False
    loss = 0
    if use_teacher_forcing:
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output_tensor[di])
            decoder_input = output_tensor[di]

    else:
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output_tensor[di])
            topV, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_len
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttenDecoderRNN(hidden_size, output_lang.n_words, max_len=MAX_LENGTH, dropout_p=0.1).to(device)

lr = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)

scheduler_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
scheduler_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=1, gamma=0.95)

criterion = nn.NLLLoss()

n_iters = 1000000

training_pairs = [
    tensorsFromPair(random.choice(pairs)) for i in range(n_iters)
]

print_every = 100
save_every = 1000
print_loss_total = 0
start = time.time()
for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    output_tensor = training_pair[1]

    loss = loss_func(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    print_loss_total += loss
    if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print("{},{},{},{}".format(timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))

    if iter % save_every == 0:
        torch.save(encoder.state_dict(), "models/encoder_{}.pth".format(iter))
        torch.save(decoder.state_dict(), "models/decoder_{}.pth".format(iter))

    if iter % 10000:
        scheduler_encoder.step()
        scheduler_decoder.step()
