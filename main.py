import pygame                           # oyun ortamının geliştirilmesinde kullanıldı
import random                           # rastgele renk seçimi gibi random değerlerin üretiminde kullanıldı
import numpy as np                      # q learning de yer alan matris işlemleri için kullanıldı
import math                             # matematiksel hesaplamalar ve fonksiyonlar için kullanıldı
from q_learning_bot import QBot         # botun yer aldığı dosya
import os                               # kayıtlı dosyalara erişmede ve kaydetmede kullanıldı
import glob                             # belirli uzantılara sahip dosyaları bulmak için kullanıldı
import sys                              # hata ayıklama işlemlerinde vs. kullanıldı
import traceback                        # hataların detaylı çözümlemesinde kullanıldı

try:
    print("Starting the game...")

    # Pygame Başlatılıyor
    pygame.init()
    print("Pygame initialized successfully")

    # SDL video driver ayarlanıyor
    os.environ['SDL_VIDEODRIVER'] = 'windows'
    
    print("Environment variables set")

    # Sabitler
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    PLAYER_START_RADIUS = 16
    FOOD_RADIUS = 8
    BOT_START_RADIUS = 16
    FOOD_COUNT = 100
    BOT_COUNT = 10
    WORLD_WIDTH = 800
    WORLD_HEIGHT = 600

    print("Creating display...")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    print("Display created successfully")
    pygame.display.set_caption("Agar.io Clone with Q-Learning Bots")

except Exception as e:
    print("Error occurred:")
    print(str(e))
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)

# Renkler
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GRID_COLOR = (230, 230, 230)

# Oyuncu Renkleri
COLORS = {
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'ORANGE': (255, 165, 0),
    'PURPLE': (128, 0, 128)
}

class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.zoom_speed = 0.1
        self.min_zoom = 0.1
        self.max_zoom = 3.0

    def update(self, player):
        # playerin toplam alanı ve kütle merkezini bul
        total_mass = 0
        center_x = 0
        center_y = 0
        max_radius = 0
        total_area = 0  # tüm parçaların toplam alanını bul
        
        # Toplam kütleyi, merkezi ve alanı hesapla
        for x, y, r in player.pieces:
            area = math.pi * r * r
            total_area += area
            mass = r * r  # Alana orantılı kütle
            center_x += x * mass
            center_y += y * mass
            total_mass += mass
            max_radius = max(max_radius, r)
        
        if total_mass > 0:
            center_x /= total_mass
            center_y /= total_mass

            # Oyuncunun merkezde kalmasını sağlamak için kamera pozisyonu doğrudan ayarlandı
            self.x = center_x - WINDOW_WIDTH / (2 * self.zoom)
            self.y = center_y - WINDOW_HEIGHT / (2 * self.zoom)
        
        # Oyuncu boyutuna ve ekran boyutlarına göre hedef yakınlaştırma hesaplandı
        # screen_area = WINDOW_WIDTH * WINDOW_HEIGHT
        min_dimension = min(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Tüm parçaların uyması için istenen görüş yarıçapı hesaplandı
        view_radius = 0
        for x, y, r in player.pieces:
            dist_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            view_radius = max(view_radius, dist_from_center + r)
        
        # Görünüm yarıçapına dolgu eklendş
        view_radius *= 1.5  # 50% dolgu
        
        # Yakınlaştırma görüş yarıçapına göre hesaplandı
        target_zoom = min_dimension / (view_radius * 4)
        
        # Yakınlaştırmanın sınırlar içinde kalması sağlandı
        self.target_zoom = min(max(target_zoom, self.min_zoom), self.max_zoom)
        
        # Mevcut yakınlaştırmayı hedef yakınlaştırmaya daha akışkan bir şekilde entegre edin
        self.zoom += (self.target_zoom - self.zoom) * self.zoom_speed

    def apply(self, x, y):
        # Dünya koordinatlarını ekran koordinatlarına çevirir
        screen_x = (x - self.x) * self.zoom
        screen_y = (y - self.y) * self.zoom
        return int(screen_x), int(screen_y)

    def apply_radius(self, r):
        # yarıçapı zoom'a göre ölçeklenir
        return r * self.zoom

class Player:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        # RGB değerlerinin minimum değerlerini sağlamak için renkleri parlaklaştırır
        r, g, b = color
        self.color = (max(r, 150), max(g, 150), max(b, 150))
        self.base_speed = 3
        self.pieces = [(x, y, radius)]

    def move(self, dx, dy):
        new_pieces = []
        for i, (x, y, r) in enumerate(self.pieces):
            current_dx = dx
            current_dy = dy
            
            # yarıçapa göre hız hesaplanır
            current_speed = self.calculate_speed(r)
            
            new_x = x + current_dx * current_speed
            new_y = y + current_dy * current_speed
            
            # Duvarlara çarparsa geri döner
            if new_x - r < 0:
                new_x = r
            elif new_x + r > WORLD_WIDTH:
                new_x = WORLD_WIDTH - r
            
            if new_y - r < 0:
                new_y = r
            elif new_y + r > WORLD_HEIGHT:
                new_y = WORLD_HEIGHT - r
            
            new_pieces.append((new_x, new_y, r))
        
        if new_pieces:
            self.pieces = new_pieces
            self.x, self.y = self.pieces[0][0], self.pieces[0][1]

    def calculate_speed(self, radius):
        # Hız yarıçapa ters orantılıdır
        # Daha küçük parçalar daha hızlı hareket eder, daha büyük parçalar daha yavaş hareket eder
        min_speed = 1.5  # Çok büyük parçalar için minimum hız
        max_speed = 4.5  # Çok küçük parçalar için maksimum hız
        
        # Yarıçapa göre hız hesaplanır
        speed = self.base_speed * (PLAYER_START_RADIUS / radius)
        
        # Hızın min ve max değerleri arasında sınırlanır
        return max(min_speed, min(speed, max_speed))

    def draw(self, screen, camera):
        for x, y, r in self.pieces:
            screen_x, screen_y = camera.apply(x, y)
            screen_r = camera.apply_radius(r)
            pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), int(screen_r))
            
            # 'P' etiketi eklendi
            font = pygame.font.Font(None, max(20, int(screen_r)))
            text = font.render('P', True, (0, 0, 0))
            text_rect = text.get_rect(center=(screen_x, screen_y))
            screen.blit(text, text_rect)

class Food:
    def __init__(self):
        self.respawn()
        
    def respawn(self):
        self.x = random.randint(FOOD_RADIUS, WORLD_WIDTH - FOOD_RADIUS)
        self.y = random.randint(FOOD_RADIUS, WORLD_HEIGHT - FOOD_RADIUS)
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def draw(self, screen, camera):
        screen_x, screen_y = camera.apply(self.x, self.y)
        screen_r = camera.apply_radius(FOOD_RADIUS)
        pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), int(screen_r))

def calculate_reward(bot, foods, players):
    # Temel ödül
    reward = 0
    
    # Yemlere yakın olma ödülü
    min_food_dist = float('inf')
    for food in foods:
        dist = math.sqrt((food.x - bot.x)**2 + (food.y - bot.y)**2)
        min_food_dist = min(min_food_dist, dist)
    
    # Yemlere yakın olma ödülü yarıçapa ters orantılıdır
    reward += 1.0 / (min_food_dist + 1)
    
    # Büyük oyunculara yakın olma cezası
    for player in players:
        if player != bot:
            dist = math.sqrt((player.x - bot.x)**2 + (player.y - bot.y)**2)
            if dist < max(player.radius, bot.radius) * 2:
                if player.radius > bot.radius:
                    reward -= 0.5  # Büyük oyunculara yakın olma cezası
                else:
                    reward += 0.3  # Küçük oyunculara yakın olma ödülü
    
    return reward

def draw_grid(screen, camera):
    # Grid çizgileri çizilir
    grid_size = 100
    
    # Görünür grid aralığı hesaplanır
    start_x = max(0, int(camera.x / grid_size)) * grid_size
    end_x = min(WORLD_WIDTH, int((camera.x + WINDOW_WIDTH / camera.zoom) / grid_size + 1) * grid_size)
    start_y = max(0, int(camera.y / grid_size)) * grid_size
    end_y = min(WORLD_HEIGHT, int((camera.y + WINDOW_HEIGHT / camera.zoom) / grid_size + 1) * grid_size)
    
    # Dikey çizgiler çizilir
    for x in range(start_x, end_x, grid_size):
        start_pos = camera.apply(x, start_y)
        end_pos = camera.apply(x, end_y)
        pygame.draw.line(screen, GRID_COLOR, start_pos, end_pos, 1)
    
    # Yatay çizgiler çizilir
    for y in range(start_y, end_y, grid_size):
        start_pos = camera.apply(start_x, y)
        end_pos = camera.apply(end_x, y)
        pygame.draw.line(screen, GRID_COLOR, start_pos, end_pos, 1)

def get_random_color():
    """COLORS dizisinden rastgele bir renk seçer"""
    return random.choice(list(COLORS.values()))

def get_available_bot_ids():
    """Get a list of bot IDs that have existing models"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bot_models')
    if not os.path.exists(models_dir):
        return []
    
    # policy net dosyaları bulunur ve bot ID'leri çıkarılır
    model_files = glob.glob(os.path.join(models_dir, 'policy_net_*.pth'))
    bot_ids = []
    for file in model_files:
        try:
            # Dosya adından bot ID'si çıkarılır (policy_net_X.pth)
            filename = os.path.basename(file)
            bot_id = int(filename.split('policy_net_')[1].split('.pth')[0])
            # Bu bot için gerekli tüm dosyalar var mı kontrol edilir
            if (os.path.exists(os.path.join(models_dir, f'target_net_{bot_id}.pth')) and
                os.path.exists(os.path.join(models_dir, f'params_{bot_id}.pth'))):
                bot_ids.append(bot_id)
        except:
            continue
    
    return sorted(bot_ids)

def create_bot(x, y, bot_id):
    """Create a bot with the given ID or reuse an existing model"""
    return QBot(x, y, BOT_START_RADIUS, get_random_color(), bot_id=bot_id)

def respawn_all_entities(player, bots):
    """Respawn all entities when someone gets too big"""
    # Oyuncu yeniden oluşturulur
    x = random.randint(PLAYER_START_RADIUS, WORLD_WIDTH - PLAYER_START_RADIUS)
    y = random.randint(PLAYER_START_RADIUS, WORLD_HEIGHT - PLAYER_START_RADIUS)
    player.pieces = [(x, y, PLAYER_START_RADIUS)]
    player.x = x
    player.y = y
    player.color = get_random_color()
    
    # Tüm botlar yeniden oluşturulur
    for bot in bots:
        x = random.randint(BOT_START_RADIUS, WORLD_WIDTH - BOT_START_RADIUS)
        y = random.randint(BOT_START_RADIUS, WORLD_HEIGHT - BOT_START_RADIUS)
        bot.pieces = [(x, y, BOT_START_RADIUS)]
        bot.x = x
        bot.y = y
        bot.color = get_random_color()

def main():
    clock = pygame.time.Clock()
    camera = Camera()

    # Oyuncu haritanın merkezinde rastgele bir renkle oluşturulur
    player = Player(WORLD_WIDTH//2, WORLD_HEIGHT//2, PLAYER_START_RADIUS, get_random_color())
    
    # Mevcut modellere göre kullanılabilir bot ID'leri alınır
    available_bot_ids = get_available_bot_ids()
    
    # Botlar haritanın düzgün dağıtılır
    bots = []
    for i in range(BOT_COUNT):
        x = random.randint(BOT_START_RADIUS, WORLD_WIDTH - BOT_START_RADIUS)
        y = random.randint(BOT_START_RADIUS, WORLD_HEIGHT - BOT_START_RADIUS)
        
        # Eğer mevcut model sayısından daha fazla bot varsa
        if i >= len(available_bot_ids):
            # Sonraki kullanılabilir ID bulunur (mevcut modellere çakışma önlemek için)
            new_id = max(available_bot_ids + [i]) + 1 if available_bot_ids else i
            bot = create_bot(x, y, new_id)
        else:
            # Mevcut model kullanılır
            bot = create_bot(x, y, available_bot_ids[i])
        
        bots.append(bot)
    
    # Yemler haritanın düzgün dağıtılır
    foods = [Food() for _ in range(FOOD_COUNT)]
    for food in foods:
        food.x = random.randint(FOOD_RADIUS, WORLD_WIDTH - FOOD_RADIUS)
        food.y = random.randint(FOOD_RADIUS, WORLD_HEIGHT - FOOD_RADIUS)

    running = True
    training_interval = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Çıkış yapmadan önce tüm bot modelleri kaydedilir
                for bot in bots:
                    bot.save_models()
                running = False

        # Sürekli klavye girişi işlenir
        keys = pygame.key.get_pressed()
        dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]
        
        # Diagonal hareket normalleştirilir
        if dx != 0 and dy != 0:
            dx /= math.sqrt(2)
            dy /= math.sqrt(2)
            
        player.move(dx, dy)
        
        # Kamera güncellenir
        camera.update(player)

        # Botlar güncellenir
        all_players = [player] + bots
        for bot in bots:
            state = bot.get_state(foods, all_players)
            action = bot.select_action(state)
            bot.move(action)
            reward = calculate_reward(bot, foods, all_players)
            next_state = bot.get_state(foods, all_players)
            bot.memory.append((state, action, reward, next_state))
            
            if training_interval % 10 == 0:
                bot.train(bot.batch_size)
            
            if training_interval % 1000 == 0:
                bot.update_target_network()

        # Herhangi bir varlık çok büyük olduğunda yeniden oluşturulur
        max_allowed_radius = WORLD_WIDTH / 4  # Yarım yarım dünya genişliği
        for entity in all_players:
            for _, _, radius in entity.pieces:
                if radius >= max_allowed_radius:
                    print(f"Entity reached {radius} radius! Respawning all entities...")
                    respawn_all_entities(player, bots)
                    break

        # Yemlere çarpmak
        all_entities = [player] + bots
        for food in foods:
            for entity in all_entities:
                for i, (x, y, r) in enumerate(entity.pieces):
                    dist = math.sqrt((food.x - x)**2 + (food.y - y)**2)
                    if dist < r + FOOD_RADIUS:
                        new_r = math.sqrt(r ** 2 + (FOOD_RADIUS * 0.8) ** 2)
                        entity.pieces[i] = (x, y, new_r)
                        food.x = random.randint(FOOD_RADIUS, WORLD_WIDTH - FOOD_RADIUS)
                        food.y = random.randint(FOOD_RADIUS, WORLD_HEIGHT - FOOD_RADIUS)

        # Oyuncu ve botlar arası çarpışmalar kontrol edilir
        for i, entity1 in enumerate(all_entities):
            for j, entity2 in enumerate(all_entities[i+1:], i+1):
                for p1_idx, (x1, y1, r1) in enumerate(entity1.pieces):
                    for p2_idx, (x2, y2, r2) in enumerate(entity2.pieces):
                        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if dist < max(r1, r2):
                            if r1 > r2:
                                new_r = math.sqrt(r1 ** 2 + (r2 * 0.8) ** 2)
                                entity1.pieces[p1_idx] = (x1, y1, new_r)
                                entity2.pieces.pop(p2_idx)
                                if not entity2.pieces:
                                    if entity2 in bots:
                                        # Bot'un ID'si alınır
                                        old_bot_id = entity2.bot_id
                                        bots.remove(entity2)
                                        x = random.randint(BOT_START_RADIUS, WORLD_WIDTH - BOT_START_RADIUS)
                                        y = random.randint(BOT_START_RADIUS, WORLD_HEIGHT - BOT_START_RADIUS)
                                        # Yeni bot oluşturulur
                                        new_bot = create_bot(x, y, old_bot_id)
                                        bots.append(new_bot)
                                    else:
                                        x = random.randint(PLAYER_START_RADIUS, WORLD_WIDTH - PLAYER_START_RADIUS)
                                        y = random.randint(PLAYER_START_RADIUS, WORLD_HEIGHT - PLAYER_START_RADIUS)
                                        entity2.pieces = [(x, y, PLAYER_START_RADIUS)]
                                        entity2.x = x
                                        entity2.y = y
                                        # Oyuncu yeni rastgele renk ile yeniden oluşturulur
                                        entity2.color = get_random_color()
                            elif r2 > r1:
                                new_r = math.sqrt(r2 ** 2 + (r1 * 0.8) ** 2)
                                entity2.pieces[p2_idx] = (x2, y2, new_r)
                                entity1.pieces.pop(p1_idx)
                                if not entity1.pieces:
                                    if entity1 in bots:
                                        # Bot'un ID'si alınır
                                        old_bot_id = entity1.bot_id
                                        bots.remove(entity1)
                                        x = random.randint(BOT_START_RADIUS, WORLD_WIDTH - BOT_START_RADIUS)
                                        y = random.randint(BOT_START_RADIUS, WORLD_HEIGHT - BOT_START_RADIUS)
                                        # Yeni bot oluşturulur
                                        new_bot = create_bot(x, y, old_bot_id)
                                        bots.append(new_bot)
                                    else:
                                        x = random.randint(PLAYER_START_RADIUS, WORLD_WIDTH - PLAYER_START_RADIUS)
                                        y = random.randint(PLAYER_START_RADIUS, WORLD_HEIGHT - PLAYER_START_RADIUS)
                                        entity1.pieces = [(x, y, PLAYER_START_RADIUS)]
                                        entity1.x = x
                                        entity1.y = y
                                        # Oyuncu yeni rastgele renk ile yeniden oluşturulur
                                        entity1.color = get_random_color()

        # Her şey çizilir
        screen.fill(WHITE)
        
        # Grid çizilir
        draw_grid(screen, camera)
        
        # Yemler çizilir
        for food in foods:
            food.draw(screen, camera)
        
        # Oyuncu ve botlar çizilir
        player.draw(screen, camera)
        for bot in bots:
            bot.draw(screen, camera)
        
        pygame.display.flip()
        clock.tick(60)
        training_interval += 1

    pygame.quit()

if __name__ == "__main__":
    main() 