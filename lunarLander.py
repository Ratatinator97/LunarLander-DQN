import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt

# Definition de l'agent
class Agent:
    def __init__(self, state_size, action_size):

        # Dimensionement des entrées et sorties de notre agent
        self.state_size = state_size
        self.action_size = action_size

        # Les hyperparametres
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.001  # Valeur classique
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.993  # Valeur déterminée par bijection

        self.model = self._build_model(
            False
        )  # Mettre a 'True' pour recuperer les valeurs des poids de l'essai precedent

    def _build_model(self, getweights=False):

        # Reseau profond de neuronnes
        model = Sequential()
        model.add(
            Dense(64, input_dim=self.state_size, activation="relu")
        )  # J'ai essaye de reduire le nombre de poids
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        if getweights:
            # On recupere les poids
            model.load_weights("weights_backup.h5")
            # On suppose les poids optimaux, on reduit donc au maximum l'exploration
            self.exploration_rate = self.exploration_min
        return model

    def act(self, state):
        # Exploration ?
        if np.random.rand() <= self.exploration_rate:
            # Action au hasard
            return random.randrange(self.action_size)
        # Sinon on predit la qualite des actions a faire et on prend la meilleure
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        # On sauvegarde dans la memoire l'experience
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        # Etape cruciale de l'apprentissage par renforcement
        # On mets a jour les poids (on apprend)

        if len(self.memory) < sample_batch_size:
            # On attend d'avoir un minimum d'elements dans la memoire
            return
        # On recupere l'equivalent d'un batch dans la memoire
        batch = random.sample(self.memory, sample_batch_size)

        # On recupere les variables dans le batch
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])

        # On les decorrele
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # L'ajustement du Q (qualite d'une action) va etre plus ou moins forte en fonction du gamma
        # On met a jour Q en fonction de la Qualite de la meilleure action future
        # et en ajoutant la recompense a l'etat present
        q_update = rewards + self.gamma * (
            np.amax(self.model.predict_on_batch(next_states), axis=1)
        ) * (1 - dones)
        q_values = self.model.predict_on_batch(states)
        indice = np.array([i for i in range(sample_batch_size)])
        q_values[[indice], [actions]] = q_update  # Mise a jour
        
        # On met a jour le modele
        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            # On reduit le taux d'exploration au fur et a mesure de l'entrainement
            self.exploration_rate *= self.exploration_decay

    def save(self):
        # Sauvegarde des poids
        self.model.save("weights_backup.h5")

    def get_trained_model(self):
        # Recuperation des poids
        trained_model = load_model("weights_backup.h5")
        return trained_model


class LunarLander:
    def __init__(self):

        self.sample_batch_size = 64
        self.episodes = 400  # J'ai estime que aux alentours de 300 episodes le reseau commence a etre tres bon
        self.env = gym.make("LunarLander-v2")

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        # On cree l'agent et on lui donne sa taille (entree + sorties)
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            loss = []  # On initialise la loss sur une experience entiere
            for index_episode in range(self.episodes):

                # On remet a zero l'environement a chaque episode
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                score = 0
                done = False  # Variable qui indique si l'episode est fini ou non
                # (Crash ou atterissage)

                for i in range(
                    2000
                ):  # Je considere que 2000 actions sont suffisantes pour atterrir
                    # Je reduits aussi afin de ne pas eterniser l'entrainement

                    # Decommenter la ligne suivante pour le mode graphique (plus lent et plus de ressources)
                    #self.env.render()

                    # On choisit la meilleure action
                    action = self.agent.act(state)

                    # On recupere les donnees de l'environement selon notre action
                    next_state, reward, done, _ = self.env.step(action)

                    score += reward
                    next_state = np.reshape(next_state, [1, self.state_size])

                    # On memorise l'experience
                    self.agent.remember(state, action, reward, next_state, done)
                    # On passe a l'etat suivant
                    state = next_state
                    # On rejoue l'experience
                    self.agent.replay(self.sample_batch_size)

                    if done:
                        print("Episode n°{} Score: {}".format(index_episode, score))
                        break
                # On ajoute le score de l'episode aux autres scores
                loss.append(score)

        finally:
            self.agent.save()
            return loss


if __name__ == "__main__":
    lunarlander = LunarLander()
    loss = lunarlander.run()

    # Affiche un graphique (episode) => (score)
    plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()
