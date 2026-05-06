from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

hf_token = os.getenv('HF_TOKEN')
wandb_key = os.getenv('WANDB_API_KEY')