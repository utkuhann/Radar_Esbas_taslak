import os
import requests
from tqdm import tqdm
from urllib.parse import urlparse

# 1. Adım: İndirilecek dosyaların URL'lerini bir sözlükte (dictionary) tanımlayın.
# GitHub Releases'den veya başka bir servisten aldığınız doğrudan indirme linklerini buraya yapıştırın.
VIDEO_URLS = {
    "traffic.mp4": "https://github.com/utkuhann/Radar_Esbas_taslak/releases/download/videos/traffic.mp4",
    "traffic2.mp4": "https://github.com/utkuhann/Radar_Esbas_taslak/releases/download/videos/traffic2.mp4",
    "traffic3.mp4": "https://github.com/utkuhann/Radar_Esbas_taslak/releases/download/videos/traffic3.mp4",
    "traffic4.mp4": "https://github.com/utkuhann/Radar_Esbas_taslak/releases/download/videos/traffic4.mp4"
}

# 2. Adım: Dosyaların indirileceği klasörü belirleyin.
DOWNLOAD_DIR = "data"

def download_file(url, directory):
    """Belirtilen URL'den dosyayı indirir ve ilerleme çubuğu gösterir."""
    
    # Dosya adını URL'den al
    local_filename = os.path.basename(urlparse(url).path)
    file_path = os.path.join(directory, local_filename)

    # Eğer dosya zaten varsa, indirmeyi atla
    if os.path.exists(file_path):
        print(f"'{local_filename}' zaten mevcut, indirme atlanıyor.")
        return

    print(f"'{local_filename}' indiriliyor...")
    
    try:
        # Sunucudan stream (akış) olarak veri iste
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Hata varsa (4xx veya 5xx) exception fırlat
            
            # Dosya boyutunu al (ilerleme çubuğu için)
            total_size = int(r.headers.get('content-length', 0))
            
            # tqdm ile bir ilerleme çubuğu oluştur
            # desc: ilerleme çubuğunun önündeki açıklama
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=local_filename)
            
            # Dosyayı parça parça diske yaz
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): # 8KB'lık parçalar
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print(f"HATA: '{local_filename}' indirilirken bir sorun oluştu.")
            else:
                # Başarılı indirme mesajını ilerleme çubuğu satırının üzerine yazmamak için boşluk bırak
                print(f"'{local_filename}' başarıyla indirildi.")

    except requests.exceptions.RequestException as e:
        print(f"HATA: '{local_filename}' indirilirken bir sorun oluştu: {e}")


# 3. Adım: Ana script mantığı
# HATA BURADAYDI: "_main_" yerine "__main__" olmalı (çift alt çizgi)
if __name__ == "__main__":
    # İndirme klasörünün var olduğundan emin ol
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"'{DOWNLOAD_DIR}' klasörü oluşturuluyor.")
        os.makedirs(DOWNLOAD_DIR)

    # Tanımlanan tüm videoları döngüye al ve indir
    for filename, url in VIDEO_URLS.items():
        download_file(url, DOWNLOAD_DIR)

    print("\nTüm işlemler tamamlandı.")
