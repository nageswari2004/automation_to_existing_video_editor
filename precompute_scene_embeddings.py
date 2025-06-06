import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('scenes_db.json', 'r', encoding='utf-8') as f:
    scenes = json.load(f)

for scene in scenes:
    scene['embedding'] = model.encode(scene['scene_text']).tolist()

with open('scenes_db_embedded.json', 'w', encoding='utf-8') as f:
    json.dump(scenes, f, indent=2) 