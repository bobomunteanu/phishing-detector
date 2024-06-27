import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from urllib.parse import urlparse
import re


class PhishingClassifier(nn.Module):
    def __init__(self, input_size):
        super(PhishingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def extract_url_features(url):
    # Initialize feature counts
    features = {
        'qty_dot_url': 0,
        'qty_hyphen_url': 0,
        'qty_underline_url': 0,
        'qty_slash_url': 0,
        'qty_questionmark_url': 0,
        'qty_equal_url': 0,
        'qty_at_url': 0,
        'qty_and_url': 0,
        'qty_exclamation_url': 0,
        'qty_space_url': 0,
        'qty_tilde_url': 0,
        'qty_comma_url': 0,
        'qty_plus_url': 0,
        'qty_asterisk_url': 0,
        'qty_hashtag_url': 0,
        'qty_dollar_url': 0,
        'qty_percent_url': 0,
        'qty_tld_url': 0,
        'length_url': 0,
        'qty_dot_domain': 0,
        'qty_hyphen_domain': 0,
        'qty_underline_domain': 0,
        'qty_slash_domain': 0,
        'qty_questionmark_domain': 0,
        'qty_equal_domain': 0,
        'qty_at_domain': 0,
        'qty_and_domain': 0,
        'qty_exclamation_domain': 0,
        'qty_space_domain': 0,
        'qty_tilde_domain': 0,
        'qty_comma_domain': 0,
        'qty_plus_domain': 0,
        'qty_asterisk_domain': 0,
        'qty_hashtag_domain': 0,
        'qty_dollar_domain': 0,
        'qty_percent_domain': 0,
        'domain_length': 0
    }

    # Extract domain and path from URL using urlparse
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    # Count occurrences of each character in the URL
    url_chars = re.findall(r'\w', url)  # Extract alphanumeric characters only

    # Update features dictionary with counts
    features['qty_dot_url'] = url.count('.')
    features['qty_hyphen_url'] = url.count('-')
    features['qty_underline_url'] = url.count('_')
    features['qty_slash_url'] = url.count('/')
    features['qty_questionmark_url'] = url.count('?')
    features['qty_equal_url'] = url.count('=')
    features['qty_at_url'] = url.count('@')
    features['qty_and_url'] = url.count('&')
    features['qty_exclamation_url'] = url.count('!')
    features['qty_space_url'] = url.count(' ')
    features['qty_tilde_url'] = url.count('~')
    features['qty_comma_url'] = url.count(',')
    features['qty_plus_url'] = url.count('+')
    features['qty_asterisk_url'] = url.count('*')
    features['qty_hashtag_url'] = url.count('#')
    features['qty_dollar_url'] = url.count('$')
    features['qty_percent_url'] = url.count('%')
    features['qty_tld_url'] = len(parsed_url.path.split('.')[-1]) if parsed_url.path else 0
    features['length_url'] = len(url_chars)

    # Count occurrences of each character in the domain
    domain_chars = re.findall(r'\w', domain)  # Extract alphanumeric characters only

    # Update features dictionary with domain counts
    features['qty_dot_domain'] = domain.count('.')
    features['qty_hyphen_domain'] = domain.count('-')
    features['qty_underline_domain'] = domain.count('_')
    features['qty_slash_domain'] = domain.count('/')
    features['qty_questionmark_domain'] = domain.count('?')
    features['qty_equal_domain'] = domain.count('=')
    features['qty_at_domain'] = domain.count('@')
    features['qty_and_domain'] = domain.count('&')
    features['qty_exclamation_domain'] = domain.count('!')
    features['qty_space_domain'] = domain.count(' ')
    features['qty_tilde_domain'] = domain.count('~')
    features['qty_comma_domain'] = domain.count(',')
    features['qty_plus_domain'] = domain.count('+')
    features['qty_asterisk_domain'] = domain.count('*')
    features['qty_hashtag_domain'] = domain.count('#')
    features['qty_dollar_domain'] = domain.count('$')
    features['qty_percent_domain'] = domain.count('%')
    features['domain_length'] = len(domain_chars)

    return features


def predict_phishing(url, model, scaler, columns_to_normalize):
    # Extract features from URL
    features = extract_url_features(url)
    features_values = [features[col] for col in columns_to_normalize]

    # Convert features to numpy array and reshape
    features_array = np.array(features_values).reshape(1, -1)

    # Normalize features using the scaler
    features_array = scaler.transform(features_array)

    # Convert to tensor
    features_tensor = torch.tensor(features_array, dtype=torch.float32)

    # Set the model to evaluation mode
    model.eval()

    # Predict using the model
    with torch.no_grad():
        output = model(features_tensor)
        print(output)
        prediction = torch.round(output.squeeze()).item()

    # Convert prediction to human-readable format
    is_phishing = bool(prediction)

    return is_phishing


# Load the model and scaler
input_size = 37  # Example input size
model_path = 'phishing_model.pth'
model = PhishingClassifier(input_size)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Load the scaler
scaler = StandardScaler()
scaler_path = 'scaler.pkl'  # Save the scaler using joblib.dump(scaler, scaler_path)
scaler = joblib.load(scaler_path)

# Columns to normalize
columns_to_normalize = [
    'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
    'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
    'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
    'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
    'qty_percent_url', 'qty_tld_url', 'length_url', 'qty_dot_domain',
    'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
    'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
    'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
    'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain', 'qty_asterisk_domain',
    'qty_hashtag_domain', 'qty_dollar_domain', 'qty_percent_domain', 'domain_length'
]

# Streamlit app
st.title('Phishing URL Detection')
url = st.text_input('Enter a URL to check if it is phishing:')
if url:
    is_phishing = predict_phishing(url, model, scaler, columns_to_normalize)
    result = 'phishing' if is_phishing else 'not phishing'
    st.write(f'The URL is {result}.')
