import os
import urllib.request
import time
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

papers = {
    # arXiv — direct PDF
    "01_Pesonen_2025_WACV.pdf": "https://arxiv.org/pdf/2408.10843",
    "02_Zhang_2023_DINO_ICLR.pdf": "https://arxiv.org/pdf/2203.03605",
    
    # Nature Scientific Data — open access PDF
    "03_Pesonen_2025_SciData.pdf": "https://www.nature.com/articles/s41597-025-05634-0.pdf",
    
    # MDPI Sensors — open access
    "04_Kim_Muminov_2023_Sensors.pdf": "https://mdpi.com/1424-8220/23/12/5702/pdf",
    "05_Mukhiddinov_2022_Sensors.pdf": "https://mdpi.com/1424-8220/22/23/9384/pdf",
    
    # MDPI Fire — open access
    "06_Chetoui_Akhloufi_2024_Fire.pdf": "https://mdpi.com/2571-6255/7/4/135/pdf",
    "07_Yang_2024_Fire.pdf": "https://mdpi.com/2571-6255/7/11/387/pdf",
    "08_Huang_2025_Fire.pdf": "https://mdpi.com/2571-6255/8/5/170/pdf",
    "09_Goncalves_2024_Fire.pdf": "https://mdpi.com/2571-6255/7/10/369/pdf",
    
    # ISPRS Archives — open access
    "10_Raita_Hakola_2023_ISPRS.pdf": "https://isprs-archives.copernicus.org/articles/XLVIII-1-W2-2023/1771/2023/isprs-archives-XLVIII-1-W2-2023-1771-2023.pdf",
    
    # PLOS ONE — open access
    "11_Shamta_Demir_2024_PLOSONE.pdf": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0299058.pdf",
    
    # Taylor & Francis / Geomatics — open access
    "12_Zhou_2025_Geomatics.pdf": "https://www.tandfonline.com/doi/pdf/10.1080/19475705.2025.2556144",
}

out_dir = r"paper\references"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

for filename, url in papers.items():
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"SKIP (exists)  {filename}")
        continue
    
    print(f"DOWNLOADING   {filename}")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
            data = response.read()
            if len(data) > 5000:
                with open(out_path, 'wb') as f:
                    f.write(data)
                size_kb = len(data) / 1024
                print(f"  OK — {size_kb:.0f} KB")
            else:
                print(f"  FAIL — only {len(data)} bytes (likely HTML redirect)")
    except Exception as e:
        print(f"  ERROR — {e}")
    time.sleep(2)

print("\nDone. Checking results:")
for filename in papers:
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        status = "OK" if size > 50000 else "SMALL"
        print(f"  {status:5}  {size/1024:8.0f} KB  {filename}")
    else:
        print(f"  MISSING  {filename}")
