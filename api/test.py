import shutil
import os
arxiv_id ='1706.03762'
path = f"./vector_cache/{arxiv_id}"

if os.path.exists(path):
    shutil.rmtree(path)