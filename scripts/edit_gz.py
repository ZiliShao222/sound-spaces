import gzip, json
p='data/datasets/semantic_audionav/mp3d/v1/test/content/yqstnuAEVhm.json.gz'
with gzip.open(p,'rt') as f:
    data=json.load(f)

data['episodes'][0]['sound_source_schedule'] = ['round_robin', 25]


with gzip.open(p,'wt') as f:
    json.dump(data,f)
print('done')