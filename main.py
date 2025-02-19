""" 
Data scraping?
Padding
Embedding: Neural network grabs matchup and also includes other statistics (win-rate, number of points)
Transformer architecture?
Data augmentation/simulation
"""


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
# Sample input (3 tournaments)
tournaments = [
    [
        # Tournament 1
        # Round 1
        [
            [9, 20], [41, 42], [40, 22], [16, 34], [30, 28], [8, 37],
            [5, 7], [5, 19], [23, 1], [15, 24], [17, 35], [14, 6],
            [27, 20], [25, 10], [33, 21], [12, 36], [2, 26], [13, 37],
            [38, 4], [29, 19], [18, 32], [3, 0]
        ],
        # Round 2
        [
            [24, 13], [4, 29], [37, 38], [19, 8], [21, 14], [26, 18],
            [6, 41], [10, 5], [22, 2], [35, 15], [1, 17], [20, 25],
            [7, 30], [36, 16], [28, 9], [32, 12], [34, 40], [19, 23],
            [37, 27], [20, 5], [42, 3], [0, 33]
        ],
        # Round 3
        [
            [35, 32], [6, 20], [5, 9], [26, 34], [2, 30], [18, 33],
            [13, 14], [15, 37], [37, 7], [4, 20], [17, 19], [8, 38],
            [41, 1], [10, 24], [23, 21], [19, 29], [12, 16], [3, 22],
            [25, 5], [27, 40], [42, 28], [36, 0]
        ],
        # Round 4
        [
            [37, 6], [43, 15], [7, 41], [20, 27], [25, 8], [30, 5],
            [3, 18], [32, 28], [1, 12], [33, 10], [34, 23], [16, 19],
            [14, 4], [19, 35], [5, 37], [9, 36], [29, 26], [24, 2],
            [21, 42], [28, 13], [38, 17], [40, 0]
        ],
        # Round 5
        [
            [9, 24], [4, 28], [7, 10], [23, 27], [26, 37], [19, 22],
            [37, 29], [43, 12], [5, 28], [32, 42], [34, 25], [6, 33],
            [19, 3], [36, 8], [41, 16], [2, 18], [38, 30], [13, 35],
            [1, 14], [20, 5], [15, 17], [21, 0]
        ],
        # Round 6
        [
            [10, 1], [30, 19], [17, 5], [2, 15], [8, 9], [18, 4],
            [24, 23], [43, 41], [3, 7], [28, 6], [36, 13], [21, 4],
            [27, 19], [37, 32], [14, 2], [25, 37], [16, 33], [29, 20],
            [5, 38], [26, 34], [42, 35], [40, 0]
        ]
    ],
    [
        [
            (9,46),(53,15),(7,32),(54,5),(55,56),(10,1),(22,6),(51,23),(41,39),(42,29),(30,43),(57,38),(31,34),(47,11),(48,25),(8,14),(37,33),(2,50),(58,36),(45,27),(4,35),(49,59),(12,28),(26,18),(60,17),(3,24),(19,20),(61,44),(52,62),(21,63),(40,13)
        ],
        [
            (64,22),(43,22),(9,15),(10,23),(46,12),(11,65),(29,52),(50,7),(51,50),(13,49),(62,19),(8,19),(47,20),(66,34),(2,27),(30,34),(38,4),(59,48),(67,48),(18,54),(63,25),(56,37),(24,45),(1,26),(35,33),(68,31),(39,54)
        ],
        [
            
        ],
    ]
]

# Corresponding outputs (number of wins per team)
win_counts = [
    # Tournament 1
    [
        6, 6, 5, 5, 5, 5, 5, 5, 4, 4,  
        4, 4, 4, 4, 4, 4, 3, 3, 3, 3,  
        3, 3, 3, 2, 2, 2, 2, 2, 2, 2,  
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  
        1, 1, 1
    ]
]

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
