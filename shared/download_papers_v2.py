import os
import urllib.request
import time
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

papers = {
    # MDPI Sensors — try alternative URL patterns
    "04_Kim_Muminov_2023_Sensors.pdf": [
        "https://www.mdpi.com/1424-8220/23/12/5702/pdf?version=1686912000",
        "https://mdpi-res.com/sensors/sensors-23-05702/article_deploy/sensors-23-05702.pdf",
    ],
    "05_Mukhiddinov_2022_Sensors.pdf": [
        "https://www.mdpi.com/1424-8220/22/23/9384/pdf?version=1669939200",
        "https://mdpi-res.com/sensors/sensors-22-09384/article_deploy/sensors-22-09384.pdf",
    ],
    # MDPI Fire
    "06_Chetoui_Akhloufi_2024_Fire.pdf": [
        "https://www.mdpi.com/2571-6255/7/4/135/pdf?version=1712534400",
        "https://mdpi-res.com/fire/fire-07-00135/article_deploy/fire-07-00135.pdf",
    ],
    "07_Yang_2024_Fire.pdf": [
        "https://www.mdpi.com/2571-6255/7/11/387/pdf?version=1730073600",
        "https://mdpi-res.com/fire/fire-07-00387/article_deploy/fire-07-00387.pdf",
    ],
    "08_Huang_2025_Fire.pdf": [
        "https://www.mdpi.com/2571-6255/8/5/170/pdf?version=1746144000",
        "https://mdpi-res.com/fire/fire-08-00170/article_deploy/fire-08-00170.pdf",
    ],
    "09_Goncalves_2024_Fire.pdf": [
        "https://www.mdpi.com/2571-6255/7/10/369/pdf?version=1727740800",
        "https://mdpi-res.com/fire/fire-07-00369/article_deploy/fire-07-00369.pdf",
    ],
    # PLOS ONE — try direct
    "11_Shamta_Demir_2024_PLOSONE.pdf": [
        "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0299058&type=printable",
        "https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0299058",
    ],
    # Taylor & Francis
    "12_Zhou_2025_Geomatics.pdf": [
        "https://www.tandfonline.com/doi/pdf/10.1080/19475705.2025.2556144?download=true",
    ],
}

out_dir = r"paper\references"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://scholar.google.com/",
}

for filename, urls in papers.items():
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"SKIP (exists)  {filename}")
        continue
    
    for url in urls:
        print(f"TRYING        {filename}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
                content_type = response.headers.get('Content-Type', '')
                data = response.read()
                if len(data) > 50000 and ('pdf' in content_type.lower() or data[:4] == b'%PDF'):
                    with open(out_path, 'wb') as f:
                        f.write(data)
                    size_kb = len(data) / 1024
                    print(f"  OK — {size_kb:.0f} KB")
                    break
                elif len(data) > 5000 and data[:4] == b'%PDF':
                    with open(out_path, 'wb') as f:
                        f.write(data)
                    size_kb = len(data) / 1024
                    print(f"  OK (small) — {size_kb:.0f} KB")
                    break
                else:
                    print(f"  FAIL — {len(data)} bytes, Content-Type: {content_type}")
        except Exception as e:
            print(f"  ERROR — {e}")
        time.sleep(2)
    else:
        print(f"  ALL URLS FAILED for {filename}")

print("\nFinal results:")
for filename in [f for f in os.listdir(out_dir) if f.endswith('.pdf')]:
    size = os.path.getsize(os.path.join(out_dir, filename))
    print(f"  {size/1024:8.0f} KB  {filename}")
missing = set(os.path.splitext(f)[0] for f in papers.keys()) - set(os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith('.pdf'))
if missing:
    print(f"\nStill missing ({len(missing)}):")
    for m in sorted(missing):
        print(f"  - {m}")
else:
    print("\nALL 12 PAPERS DOWNLOADED!")
