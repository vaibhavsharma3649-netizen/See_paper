import requests
from pathlib import Path

def fetch_arxiv_pdf(arxiv_id: str, save_dir: str = "./papers") -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_dir}/{arxiv_id}.pdf"
    
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        f.write(response.content)
    
    return save_path
