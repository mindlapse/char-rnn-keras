import keras
from keras import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

class NameGenerator(object):

    def __init__(self, filename, max_len):
        self.filename = filename
        text = open('data/names.txt.cleaned.txt').read()
        self.char_map = sorted(list(set(text)))
        self.char_map = ['_'] + self.char_map
        self.null_index = 0
        self.num_chars = len(self.char_map)
        self.max_len = 32


    def create_training_set(self):

        with open(self.filename) as f:
            lines = f.readlines()

            # Turn each name into a list of character indexes
            X_chars = []
            y_chars = []
            for line in lines:
                line = line.strip()
                X_chars += [[self.char_map.index(c) for c in line]]
                y_chars += [[self.char_map.index(c) for c in line[1:]] + [0]]


            X_padded = pad_sequences(X_chars, padding='post', truncating='post', \
                                     maxlen=self.max_len, value=self.null_index)

            y_padded = pad_sequences(y_chars, padding='post', truncating='post', \
                                     maxlen=self.max_len, value=self.null_index)

            self.X = np.array([to_categorical(int_vec, num_classes=self.num_chars) for int_vec in X_padded])
            self.y = np.array([to_categorical(int_vec, num_classes=self.num_chars) for int_vec in y_padded])




    def create_model(self, hidden_units=512, dropout=.3):

        self.model = Sequential([

            LSTM(hidden_units, input_shape=(None, self.num_chars), return_sequences=True),
            Dropout(dropout),

            LSTM(hidden_units, return_sequences=True),
            Dropout(dropout),

            LSTM(hidden_units, return_sequences=True),
            Dropout(dropout),

            TimeDistributed(Dense(self.num_chars)),

            Activation('softmax')

        ])



    def compile_model(self, optimizer='rmsprop'):
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        print(self.model.summary())


    def sample_output(self, times=5):
        for i in range(times):
            str = np.random.choice(self.char_map)

            for i in range(self.max_len):
                input = np.zeros((1, len(str), self.num_chars))

                for t, char in enumerate(str):
                    input[0, t, self.char_map.index(char)] = 1

                probs = self.model.predict(input, verbose=0)
                # print(np.array(probs).shape)
                # print(probs[0][i][:4])
                # print(np.random.multinomial(1, probs[0][i] / (1+1e-5)))

                str += self.char_map[np.argmax(np.random.multinomial(1, probs[0][i] / (1 + 1e-5)))]

            # Sample from the probability distribution
            print(str)

    def train_model(self, epochs=50, batch_size=128):

        for epoch in range(epochs):
            print('-' * 10 + ' Iteration: {} '.format(epoch) + '-' * 10)



            history = self.model.fit(self.X, self.y, batch_size=batch_size, epochs=1, verbose=1)
            print('loss is {}'.format(history.history['loss'][0]))


if __name__ == "__main__":

    filename = 'data/names.txt.cleaned.txt'


    ng = NameGenerator(filename, max_len=32)

    ng.create_training_set()
    ng.create_model( hidden_units=512, dropout=.3 )
    ng.compile_model( optimizer='rmsprop' )
    ng.train_model( epochs=5000, batch_size=10 )

