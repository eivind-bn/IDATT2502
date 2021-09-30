import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, label_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.fc1 = nn.Linear(128, label_size)

    # Batch size is the number of sequences to be passed to the LSTM.
    # When training, batch size is typically > 1, but the batch size is 1 when generating
    def reset(self, batch_size=1):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, batch_size, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.fc1(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


chars = {
    ' ': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'h': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'a': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    't': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    'r': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    'c': [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    'f': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    'l': [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    'm': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    'p': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    's': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    'o': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    'n': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
}


char_encoding_size = len(chars)
index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']


emojis = {
    'hat': '\U0001F3A9',
    'cat': '\U0001F408',
    'rat': '\U0001F400',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}


emoji_enc = {
    'hat':[1., 0., 0., 0., 0., 0., 0.],
    'rat': [0., 1., 0., 0., 0., 0., 0.],
    'cat': [0., 0., 1., 0., 0., 0., 0.],
    'flat': [0., 0., 0., 1., 0., 0., 0.],
    'matt': [0., 0., 0., 0., 1., 0., 0.],
    'cap': [0., 0., 0., 0., 0., 1., 0.],
    'son': [0., 0., 0., 0., 0., 0., 1.]
}

emoji_encoding_size = len(emojis)
index_to_emoji = [emojis['hat'], emojis['rat'],
                  emojis['cat'], emojis['flat'],
                  emojis['matt'], emojis['cap'],
                  emojis['son']]

x_train = torch.tensor([
    [[chars['h']], [chars['a']], [chars['t']], [chars[' ']]],
    [[chars['r']], [chars['a']], [chars['t']], [chars[' ']]],
    [[chars['c']], [chars['a']], [chars['t']], [chars[' ']]],
    [[chars['f']], [chars['l']], [chars['a']], [chars['t']]],
    [[chars['m']], [chars['a']], [chars['t']], [chars['t']]],
    [[chars['c']], [chars['a']], [chars['p']], [chars[' ']]],
    [[chars['s']], [chars['o']], [chars['n']], [chars[' ']]]])

y_train = torch.tensor([
    [emoji_enc['hat'], emoji_enc['hat'], emoji_enc['hat'], emoji_enc['hat']],
    [emoji_enc['rat'], emoji_enc['rat'], emoji_enc['rat'], emoji_enc['rat']],
    [emoji_enc['cat'], emoji_enc['cat'], emoji_enc['cat'], emoji_enc['cat']],
    [emoji_enc['flat'], emoji_enc['flat'], emoji_enc['flat'], emoji_enc['flat']],
    [emoji_enc['matt'], emoji_enc['matt'], emoji_enc['matt'], emoji_enc['matt']],
    [emoji_enc['cap'], emoji_enc['cap'], emoji_enc['cap'], emoji_enc['cap']],
    [emoji_enc['son'], emoji_enc['son'], emoji_enc['son'], emoji_enc['son']]])

model = LongShortTermMemoryModel(char_encoding_size, emoji_encoding_size)
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(len(x_train)):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


def run(arg: str):
    model.reset()
    y = ''
    for i, c in enumerate(arg):
        y = model.f(torch.tensor([[chars[c]]]))

    print(index_to_emoji[y.argmax()])



while True:
    run(input('Type emoji-name:'))

