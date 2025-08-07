# Çoklu Kamera ile Gerçek Zamanlı Hız Tespiti ve Raporlama Sistemi

Bu proje, Python tabanlı bir bilgisayarlı görü sistemidir. Birden fazla video akışını eş zamanlı olarak işleyerek hız ihlali yapan araçları tespit eder, plakalarını okur ve resimli bir Excel raporu oluşturur.

![sample_screenshot](https://i.imgur.com/gK9f3Yg.jpeg)
*(Bu kısma kendi projenizden bir ekran görüntüsü ekleyebilirsiniz.)*

## Ana Özellikler

- **Eş Zamanlı Çoklu Video İşleme:** Birden fazla kamera akışını, her biri ayrı bir işlem parçacığında (thread) olmak üzere aynı anda işler.
- **Yüksek Doğrulukta Hız Ölçümü:** Her video için ayrı ayrı yapılandırılabilen sanal çift çizgi metodu ile anlık hız yerine, belirli bir mesafedeki **ortalama hızı** hesaplayarak çok daha doğru sonuçlar üretir.
- **Yapay Zeka Tabanlı Tespit ve Takip:**
  - **Araç Tespiti:** YOLOv8 modeli ile videodaki araçları yüksek başarıyla tespit eder.
  - **Araç Takibi:** Supervision ve ByteTrack kullanarak tespit edilen her araca benzersiz bir kimlik atar ve takibini yapar.
- **Gelişmiş Plaka Okuma:**
  - **Plaka Tespiti:** Plakalar için eğitilmiş özel bir YOLO modeli kullanır.
  - **Görüntü İyileştirme:** Düşük çözünürlüklü veya bulanık plaka görüntülerinin kalitesini **Real-ESRGAN** modeli ile artırır.
  - **OCR:** Google Cloud Vision API kullanarak iyileştirilmiş görüntüden plaka metnini yüksek doğrulukla okur.
- **Web Tabanlı Canlı İzleme:** Flask ile oluşturulmuş basit bir web arayüzü üzerinden tüm kamera akışları canlı olarak izlenebilir.
- **Otomatik Raporlama:** Tespit edilen tüm ihlalleri; ihlal zamanı, araç resmi, plaka resmi, hız ve plaka metni gibi detaylarla birlikte tek bir Excel (`.xlsx`) dosyasına kaydeder.

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### **Ön Gereksinimler**

- **Python:** Bu proje **Python 3.10.x** veya **3.11.x** versiyonları ile geliştirilmiş ve test edilmiştir.
- **Git:** Projeyi klonlamak için gereklidir.

### 1. Projeyi Klonlayın

```bash
git clone [https://github.com/utkuhann/Radar_Esbas_taslak.git](https://github.com/utkuhann/Radar_Esbas_taslak.git)
cd Radar_Esbas_taslak
```

### 2. Sanal Ortam Oluşturun (Tavsiye Edilir)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

Gerekli tüm Python kütüphanelerini, versiyonları sabitlenmiş olan `requirements.txt` dosyası ile tek seferde yükleyin.

```bash
pip install -r requirements.txt
```

**Önemli Not (PyTorch Kurulumu):** `requirements.txt` dosyasındaki PyTorch versiyonu genellikle CPU içindir. Eğer sisteminizde NVIDIA GPU varsa ve CUDA ile hızlandırmadan faydalanmak istiyorsanız, [PyTorch'un resmi web sitesine](https://pytorch.org/get-started/locally/) gidin, kendi CUDA versiyonunuza uygun kurulum komutunu alın ve PyTorch'u bu komutla yeniden kurun.

### 4. Modelleri İndirin

Projenin ihtiyaç duyduğu önceden eğitilmiş modelleri indirip proje klasörüne yerleştirin.

- `yolov8n.pt` (Ultralytics tarafından sağlanır, ilk çalıştırmada otomatik inebilir)
- `lapi.pt` (Sizin özel plaka tespit modeliniz)
- `RealESRGAN_x4plus.pth` (Real-ESRGAN modeli)

Bu dosyaları projenin ana dizininde veya kod içinde belirttiğiniz bir klasörde bulundurduğunuzdan emin olun.

### 5. Google Cloud Kimlik Doğrulaması

Bu proje plaka okuma için Google Cloud Vision API kullanmaktadır.

1.  Google Cloud üzerinden bir servis hesabı oluşturun ve bir JSON anahtar dosyası indirin.
2.  İndirdiğiniz JSON dosyasının yolunu bir ortam değişkeni olarak ayarlayın.

    ```bash
    # Windows (Komut İstemi)
    set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\keyfile.json"

    # macOS / Linux
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
    ```

## Yapılandırma

`hiz_denetimi_multi_video_2line_web.py` dosyasını açarak `VIDEO_CONFIGS` listesini kendi ihtiyaçlarınıza göre düzenleyin.

```python
VIDEO_CONFIGS = [
    {
        "path": "path/to/your/video1.mp4",
        "line_top_ratio": 0.45,         # Üst çizginin yüksekliği (0.0-1.0)
        "line_bottom_ratio": 0.75,      # Alt çizginin yüksekliği (0.0-1.0)
        "distance_m": 15                # İki çizgi arasındaki gerçek mesafe (metre)
    },
    # ... diğer videolar için ...
]
```
- **`distance_m`**: Bu değerin doğruluğu, hız hesaplamasının doğruluğunu doğrudan etkiler. Lütfen bu mesafeyi dikkatlice ölçerek girin.
- `HIZ_LIMITI_KMH` gibi diğer global ayarları da aynı dosyadan değiştirebilirsiniz.

## Çalıştırma

Tüm kurulum ve yapılandırma adımları tamamlandıktan sonra, aşağıdaki komutla uygulamayı başlatın:

```bash
python hiz_denetimi_multi_video_2line_web.py
```

Terminalde web sunucusunun başladığına dair bir mesaj göreceksiniz. Ardından, bir web tarayıcısı açarak `http://127.0.0.1:5000` adresine gidin.

Tüm video akışları bittiğinde, tespit edilen ihlallerin tamamını içeren Excel raporu (`birlesik_ihlal_raporu_dogru_hesap.xlsx`) projenin ana dizininde oluşturulacaktır.