"""Find available uncensored models on OpenRouter"""
import os
import requests

api_key = os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY")
base_url = os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Get model list
r = requests.get(
    f"{base_url.replace('/v1', '')}/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

models = r.json().get('data', [])

# Keywords for uncensored/unaligned models
keywords = ['dolphin', 'hermes', 'mytho', 'uncensored', 'airoboros', 'wizard', 
            'samantha', 'synthia', 'noromaid', 'toppy', 'goliath', 'kimiko']

print("AVAILABLE UNCENSORED/UNALIGNED MODELS ON OPENROUTER:")
print("="*70)

found = []
for m in models:
    mid = m.get('id', '').lower()
    name = m.get('name', '').lower()
    
    for kw in keywords:
        if kw in mid or kw in name:
            found.append({
                'id': m.get('id'),
                'name': m.get('name'),
                'context': m.get('context_length', 0)
            })
            break

# Sort by name
found.sort(key=lambda x: x['id'])

for m in found:
    print(f"{m['id']:<55} ctx={m['context']}")

print(f"\nTotal found: {len(found)}")
