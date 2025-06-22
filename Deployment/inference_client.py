import json
import urllib.request
import urllib.error

def invoke_inference(
    lambda_url: str,
    api_key: str,
    # numeric features
    age: int,
    tenure: int,
    usage_frequency: int,
    support_calls: int,
    payment_delay: int,
    total_spend: int,
    last_interaction: int,
    churn: int,
    # categorical features
    gender: str,
    subscription_type: str,
    contract_length: str
) -> dict:
    """
    Calls a POST /inference on your Lambda URL using only the standard library.

    Args:
        lambda_url: Full Function URL (…/inference)
        api_key:    API key to send in the Authorization header
        age:                "Age" (int)
        tenure:             "Tenure" (int)
        usage_frequency:    "Usage Frequency" (int)
        support_calls:      "Support Calls" (int)
        payment_delay:      "Payment Delay" (int)
        total_spend:        "Total Spend" (int)
        last_interaction:   "Last Interaction" (int)
        churn:              "Churn" (int)
        gender:             "Gender" (str: "Female"|"Male")
        subscription_type:  "Subscription Type" (str: "Basic"|"Premium"|"Standard")
        contract_length:    "Contract Length" (str: "Annual"|"Monthly"|"Quarterly")

    Returns:
        The parsed JSON response as a Python dict.

    Raises:
        urllib.error.HTTPError on non-2xx responses,
        urllib.error.URLError on network errors,
        ValueError if response is not valid JSON.
    """
    payload = {
        'Age': age,
        'Tenure': tenure,
        'Usage Frequency': usage_frequency,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction,
        'Churn': churn,
        'Gender': gender,
        'Subscription Type': subscription_type,
        'Contract Length': contract_length
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url=f'{lambda_url}/inference',
        data=data,
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'Authorization': api_key
        }
    )

    try:
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read().decode('utf-8')
            return json.loads(resp_body)
    except urllib.error.HTTPError as he:
        try:
            err_body = he.read().decode('utf-8')
        except Exception:
            err_body = '<could not read error body>'
        print(f'[HTTPError {he.code}] {he.reason}')
        print(f'[Error body] {err_body}')
        raise

    except urllib.error.URLError as ue:
        print(f'[URLError] {ue.reason}')
        raise

# Example usage:
if __name__ == '__main__':
    LAMBDA_URL = 'https://your-lambda-url/inference'
    API_KEY    = 'my-secret-key'

    try:
        result = invoke_inference(
            LAMBDA_URL,
            API_KEY,
            age=45,
            tenure=12,
            usage_frequency=20,
            support_calls=1,
            payment_delay=0,
            total_spend=500,
            last_interaction=5,
            churn=0,
            gender='Female',
            subscription_type='Premium',
            contract_length='Monthly'
        )
        print('Inference result:', result)
    except Exception as err:
        print('Error calling inference endpoint:', err)
