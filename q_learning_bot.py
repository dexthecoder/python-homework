import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
import pygame
import os

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)

class QBot:
    def __init__(self, x, y, radius, color, bot_id=None):
        self.x = x
        self.y = y
        self.radius = radius
        # RGB değerlerinin minimum değerlerini sağlamak için renkleri parlaklaştırır
        r, g, b = color
        self.color = (max(r, 150), max(g, 150), max(b, 150))
        self.speed = 3
        self.pieces = [(x, y, radius)]
        self.bot_id = bot_id if bot_id is not None else random.randint(0, 999999)
        
        # Model dizini oluşturulur
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bot_models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Dünya boyutları
        self.WORLD_WIDTH = 800
        self.WORLD_HEIGHT = 600
        
        # Q-learning parametreleri
        self.input_size = 9
        self.output_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        
        # Kaydedilmiş modelleri yüklemeye çalışır
        self.load_models()
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        
        # Model kaydedilme sayacı
        self.train_counter = 0
        self.save_interval = 1000

    def save_models(self):
        """Save both policy and target networks"""
        try:
            policy_path = os.path.join(self.models_dir, f'policy_net_{self.bot_id}.pth')
            target_path = os.path.join(self.models_dir, f'target_net_{self.bot_id}.pth')
            params_path = os.path.join(self.models_dir, f'params_{self.bot_id}.pth')
            
            # Eğer dizin silinmişse yeniden oluşturulur
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
            
            torch.save(self.policy_net.state_dict(), policy_path)
            torch.save(self.target_net.state_dict(), target_path)
            
            # Eğitim parametreleri kaydedilir
            torch.save({
                'epsilon': self.epsilon,
                'memory': list(self.memory)
            }, params_path)
            
        except Exception as e:
            print(f"Error saving models for bot {self.bot_id}: {e}")

    def load_models(self):
        """Load both policy and target networks if they exist"""
        try:
            policy_path = os.path.join(self.models_dir, f'policy_net_{self.bot_id}.pth')
            target_path = os.path.join(self.models_dir, f'target_net_{self.bot_id}.pth')
            params_path = os.path.join(self.models_dir, f'params_{self.bot_id}.pth')
            
            if os.path.exists(policy_path) and os.path.exists(target_path):
                self.policy_net.load_state_dict(torch.load(policy_path))
                self.target_net.load_state_dict(torch.load(target_path))
                print(f"Loaded existing model for bot {self.bot_id}")
                
                # Eğitim parametreleri yüklenir
                if os.path.exists(params_path):
                    params = torch.load(params_path)
                    self.epsilon = params['epsilon']
                    self.memory = deque(params['memory'], maxlen=10000)
                    print(f"Loaded training parameters for bot {self.bot_id}")
                    
        except Exception as e:
            print(f"Error loading models for bot {self.bot_id}: {e}")
            # Yükleme başarısız olursa yeni başlatılmış ağlar kullanılır

    def get_state(self, foods, players):
        # En yakın yem bulunur
        min_food_dist = float('inf')
        food_dx = food_dy = 0
        for food in foods:
            dist = math.sqrt((food.x - self.x)**2 + (food.y - self.y)**2)
            if dist < min_food_dist:
                min_food_dist = dist
                food_dx = food.x - self.x
                food_dy = food.y - self.y

        # Duvar uzaklıkları
        wall_left = self.x
        wall_right = self.WORLD_WIDTH - self.x
        wall_up = self.y
        wall_down = self.WORLD_HEIGHT - self.y

        # En yakın oyuncu uzaklıkları ve boyutu
        min_player_dist = float('inf')
        player_dx = player_dy = 0
        relative_size = 1.0  # Varsayılan olarak nötr boyut oranı
        
        for player in players:
            if player != self:
                dist = math.sqrt((player.x - self.x)**2 + (player.y - self.y)**2)
                if dist < min_player_dist:
                    min_player_dist = dist
                    player_dx = player.x - self.x
                    player_dy = player.y - self.y
                    # Bu bot ve en yakın oyuncu arasındaki boyut oranı hesaplanır
                    if player.pieces and self.pieces:  # Her iki parça da var mı kontrol edilir
                        player_size = player.pieces[0][2]  # En yakın oyuncunun ilk parçasının yarıçapı
                        my_size = self.pieces[0][2]  # Bu bot'un ilk parçasının yarıçapı
                        relative_size = my_size / player_size  # >1 ise büyük, <1 ise küçük

        state = torch.FloatTensor([
            food_dx/self.WORLD_WIDTH, food_dy/self.WORLD_HEIGHT,  # yem yönü
            wall_left/self.WORLD_WIDTH, wall_right/self.WORLD_WIDTH,  # duvar uzaklıkları
            wall_up/self.WORLD_HEIGHT, wall_down/self.WORLD_HEIGHT,
            player_dx/self.WORLD_WIDTH, player_dy/self.WORLD_HEIGHT,  # en yakın oyuncu yönü
            relative_size  # boyut oranı
        ]).to(self.device)

        return state

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def move(self, action):
        # Hareketi (0,1,2,3) formatına çevirir
        if action == 0:  # yukarı
            dx, dy = 0, -1
        elif action == 1:  # aşağı   
            dx, dy = 0, 1
        elif action == 2:  # sol
            dx, dy = -1, 0
        else:  # sağ
            dx, dy = 1, 0

        new_pieces = []
        for i, (x, y, r) in enumerate(self.pieces):
            new_x = x + dx * self.speed
            new_y = y + dy * self.speed
            
            # Bot'un sınırlar içinde kalması sağlanır
            new_x = max(r, min(self.WORLD_WIDTH - r, new_x))
            new_y = max(r, min(self.WORLD_HEIGHT - r, new_y))
            
            new_pieces.append((new_x, new_y, r))
        
        if new_pieces:
            self.pieces = new_pieces
            self.x, self.y = self.pieces[0][0], self.pieces[0][1]

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.stack(next_states)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values)

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon azalır
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Sayacı artır ve düzenli olarak kaydeder
        self.train_counter += 1
        if self.train_counter % self.save_interval == 0:
            self.save_models()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Hedef ağ güncellendikten sonra modeller kaydedilir
        self.save_models()

    def draw(self, screen, camera):
        for x, y, r in self.pieces:
            screen_x, screen_y = camera.apply(x, y)
            screen_r = camera.apply_radius(r)
            pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), int(screen_r))
            
            # Bot ID eklendi
            font = pygame.font.Font(None, max(20, int(screen_r)))
            text = font.render(str(self.bot_id), True, (0, 0, 0))
            text_rect = text.get_rect(center=(screen_x, screen_y))
            screen.blit(text, text_rect) 