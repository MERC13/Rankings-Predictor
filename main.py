""" 
Data scraping?
Padding
Embedding: Neural network grabs matchup and also includes other statistics (win-rate, number of points)
Transformer architecture?
Data augmentation/simulation
"""


from data import tournaments, win_counts
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Flatten, Reshape, LSTM

# Constants
MAX_TEAMS = 126
NUM_ROUNDS = 6
MAX_PAIRINGS_PER_ROUND = 63
EMBEDDING_DIM = 32
PAD_VALUE = 0

# 1. Preprocessing ------------------------------------------------------------
def preprocess_input(tournaments):
    """Convert list of tournaments to padded numerical format"""
    processed = []
    for tournament in tournaments:
        tournament_data = []
        for round_pairings in tournament:
            # Pad round to 63 pairings with [PAD_VALUE, PAD_VALUE]
            padded_round = np.full((MAX_PAIRINGS_PER_ROUND, 2), PAD_VALUE, dtype=int)
            for i, (team1, team2) in enumerate(round_pairings):
                padded_round[i] = [team1, team2]
            tournament_data.append(padded_round)
        processed.append(np.array(tournament_data))
    return np.array(processed)





# 2. Neural Network Architecture ----------------------------------------------
def build_model(number_of_rounds, number_of_matchups_per_round):
    input_shape = (number_of_rounds, MAX_TEAMS)
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(64)(x)
    outputs = Dense(MAX_TEAMS, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae')
    return model






# 3. Usage ------------------------------------------------------------
tournaments = tournaments()
win_counts = win_counts()

# Preprocess
X = preprocess_input(tournaments)
y = np.array(win_counts)
y_padded = np.zeros((y.shape[0], MAX_TEAMS))
y_padded[:, :y.shape[1]] = y
y = y_padded
number_of_rounds = len(tournaments[0])
number_of_matchups_per_round = len(tournaments[0][0])
num_tournaments = len(tournaments)
X = X.reshape((X.shape[0], number_of_rounds, -1))

# Initialize model
model_path = 'rankings_predictor_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded from disk.")
else:
    model = build_model(number_of_rounds, number_of_matchups_per_round)
    print("New model initialized.")
model.summary()


# Train
model.fit(X, y, epochs=200, batch_size=32, validation_split=0)

model.save('rankings_predictor_model.h5')

# Predict
predicted_wins = model.predict(X)
for i, wins in enumerate(predicted_wins):
    print(f"Predicted wins for tournament {i + 1}:")
    for team_id, win_count in enumerate(wins):
        print(f"  Team {team_id + 1}: {win_count:.6f} wins")
