# Agar.io Klonu - Q-Learning Botlarla

Bu proje, popüler Agar.io oyununun basit bir klonudur ve PyTorch kullanarak oyunu oynamayı öğrenen Q-learning tabanlı yapay zekâ botlarını içermektedir.

---

## 🎮 Özellikler

- Oyuncu tarafından kontrol edilen, hareket edebilen ve bölünebilen hücre
- Q-Learning kullanan yapay zekâ kontrollü botlar
- Toplanabilir yiyecek parçacıkları
- Oyuncular ve botlar arasında çarpışma algılama
- Yiyecek veya diğer oyuncular tüketildikçe dinamik boyut artışı

---

## 🧰 Gereksinimler

- Python 3.8 veya üzeri
- PyGame
- PyTorch
- NumPy

---

## ⚙️ Kurulum

1. Bu Git deposunu klonlayın:
   ```bash
   git clone https://github.com/kullanici-adi/agar-ai-clone.git
   cd agar-ai-clone
Gerekli Python paketlerini yükleyin:

bash
Kopyala
Düzenle
pip install -r requirements.txt
▶️ Nasıl Oynanır?
Oyunu başlatmak için aşağıdaki komutu çalıştırın:

bash
Kopyala
Düzenle
python game.py
🎮 Kontroller
Yön tuşları: Hücrenizi hareket ettirin

🧠 Oyun Kuralları
Yiyecek parçacıklarını toplayarak büyüyün

Daha büyük hücreler daha küçükleri yiyebilir

Daha büyük hücrelerden kaçının

Botların zamanla öğrenip gelişen stratejilerini gözlemleyin

🤖 Q-Learning Uygulaması
Yapay zekâ botları Deep Q-Learning algoritmasını kullanmaktadır. Özellikleri:

Durum Uzayı (State Space):

En yakın yiyecek parçacığına olan mesafe

Duvarlara olan mesafe

Diğer oyunculara olan mesafe

Aksiyon Uzayı (Action Space):

Yukarı, aşağı, sola, sağ olmak üzere 4 yönlü hareket

**Ödül Fonksiyonu:

Yiyeceğe yakınlık → pozitif ödül

Daha büyük oyunculara yakınlık → negatif ödül

Daha küçük oyunculara yakınlık → pozitif ödül

Deneyim tekrarı (Experience Replay)

Kararlılığı artırmak için hedef ağ (Target Network)

👥 Ekip Üyeleri

032390060 - Yusuf İslam Çelik

032390080 - Arda İnanç

032390025 - Polat Ceylan

032390079 - Yusuf Cihan Yılmaz

