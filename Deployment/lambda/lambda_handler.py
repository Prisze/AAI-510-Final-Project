import json
import torch
from tab_transformer_pytorch import TabTransformer
import os

with open('config.json') as config_file:
    CONFIG = json.load(config_file)

# Load model
checkpoint = torch.load('tabtransformer_best.tb', map_location='cpu')
model = TabTransformer(
  categories=tuple(CONFIG['cardinalities']),
  num_continuous=len(CONFIG['num_cols']),
  dim=32,
  dim_out=1,
  depth=6,
  heads=8,
  attn_dropout=0.1,
  ff_dropout=0.1,
  mlp_hidden_mults=(4, 2),
  continuous_mean_std=torch.tensor(CONFIG['continuous_mean_std']),
)
model.load_state_dict(checkpoint)

JSON_HEADERS = {"Content-Type": "application/json"}
def handler(event, context):
    # 1) API key check via Authorization header
    headers = event.get('headers') or {}
    api_key = headers.get('authorization')
    if not api_key or api_key != os.environ.get('API_KEY'):
        return {
            'statusCode': 403,
            'headers': JSON_HEADERS,
            'body': json.dumps({'error': 'Forbidden'})
        }
    
    # 2) Only allow POST
    method = event.get('requestContext', {}).get('http', {}).get('method')
    if method != 'POST':
        return {
            'statusCode': 405,
            'headers': JSON_HEADERS,
            'body': json.dumps({'error': 'Method Not Allowed'})
        }

    # 3) Only path /inference
    path = event.get('rawPath', '')
    if path != '/inference':
        return {
            'statusCode': 404,
            'headers': JSON_HEADERS,
            'body': json.dumps({'error': 'Not Found'})
        }


    # 4) Parse JSON body
    try:
        payload = json.loads(event.get('body') or '{}')
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'headers': JSON_HEADERS,
            'body': json.dumps({'error': 'Bad Request'})
        }

    errors = []

    # 4a) Validate numeric columns
    for key in CONFIG['num_cols']:
        v = payload.get(key)
        if v is None:
            errors.append(f'Missing numeric field "{key}".')
        elif not isinstance(v, int):
            errors.append(f'"{key}" must be an integer.')

    # 4b) Validate categorical columns
    for key in CONFIG['cat_cols']:
        v = payload.get(key)
        allowed = CONFIG['cat_dtypes'].get(key, [])
        if v is None:
            errors.append(f'Missing categorical field "{key}".')
        elif not isinstance(v, str):
            errors.append(f'"{key}" must be a string.')
        elif v not in allowed:
            errors.append(f'"{key}" must be one of [{", ".join(allowed)}]; got "{v}".')

    # 5) If any validation errors, return 400 with details
    if errors:
        return {
            'statusCode': 400,
            'headers': JSON_HEADERS,
            'body': json.dumps({
                'error': 'Validation failed.',
                'details': errors
            })
        }

    # 6) All good - perform inference
    # Categorical features as LongTensor in CONFIG order
    cat_indices = [CONFIG['cat_dtypes'][col].index(payload[col]) for col in CONFIG['cat_cols']]
    x_categ = torch.tensor([cat_indices], dtype=torch.long)

    # Continuous features as FloatTensor in CONFIG order
    cont_values = [float(payload[col]) for col in CONFIG['num_cols']]
    x_cont = torch.tensor([cont_values], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_categ, x_cont)
        prediction = torch.sigmoid(logits.detach()).cpu().tolist()
    result = {
        'prediction': prediction[0][0]
    }

    return {
        'statusCode': 200,
        'headers': JSON_HEADERS,
        'body': json.dumps(result)
    }