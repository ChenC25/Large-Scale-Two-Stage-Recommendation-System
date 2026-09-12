"""Read-only demo presentation; metadata does not enter model features."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def catalog(raw_dir, category):
    path = Path(raw_dir) / 'downloads/raw/meta_categories' / f'meta_{category}.jsonl'
    items = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                images = row.get('images') or []
                image = next((im.get('large') or im.get('thumb') for im in images
                              if isinstance(im, dict) and (im.get('large') or im.get('thumb'))), None)
                if image and not image.startswith('https://'):
                    image = None
                items[str(row['parent_asin'])] = {'title': row.get('title') or row['parent_asin'],
                    'image': image, 'category': row.get('main_category') or 'Video Games'}
    return items


def demo_result(engine, cfg, user):
    items = catalog(cfg['data']['raw_dir'], cfg['data']['dataset_name'])
    def product(iid):
        return {'item_id': iid, **items.get(iid, {'title': iid, 'image': None, 'category': 'Video Games'})}
    candidates, similarities = engine.candidates(user)
    scores = engine.ranker.predict(engine.features.build(user, candidates, similarities), num_threads=1)
    order = np.argsort(-scores, kind='stable')[:20]
    history = engine.user_histories.get(engine.vocab['user_id'][user], [])
    return {
        'history': [product(engine.id_to_raw_item[i]) for i in history[-8:][::-1] if i in engine.id_to_raw_item],
        'retrieval': [{**product(iid), 'score': float(similarities[i]), 'retrieval_rank': i+1}
                      for i, iid in enumerate(candidates[:20])],
        'ranking': [{**product(candidates[i]), 'score': float(scores[i]), 'retrieval_rank': int(i)+1}
                    for i in order],
        'candidate_count': len(candidates), 'history_count': len(history),
    }
