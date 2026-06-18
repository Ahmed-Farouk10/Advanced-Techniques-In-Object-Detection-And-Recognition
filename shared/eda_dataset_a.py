import os
from PIL import Image

path = r'd:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-a\raw\Forest Fire Dataset\Training'

stats = {'fire': {'count': 0, 'sizes': set()}, 'nofire': {'count': 0, 'sizes': set()}}

for c in ['fire', 'nofire']:
    c_path = os.path.join(path, c)
    if os.path.exists(c_path):
        for img_name in os.listdir(c_path):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                stats[c]['count'] += 1
                try:
                    with Image.open(os.path.join(c_path, img_name)) as img:
                        stats[c]['sizes'].add(img.size)
                except Exception:
                    pass

print('--- DATA UNDERSTANDING REPORT ---')
for c in ['fire', 'nofire']:
    print("Class: " + c)
    print("  Count: " + str(stats[c]['count']))
    print("  Unique Image Resolutions: " + str(len(stats[c]['sizes'])))
    print("  Sample Resolutions: " + str(list(stats[c]['sizes'])[:3]))
