import json
from pathlib import Path
from .models import RedditPost, Comment

def load_sample_posts():
    data=json.loads((Path(__file__).resolve().parent.parent/"data/sample_reddit.json").read_text(encoding="utf-8"))
    return [RedditPost(str(p["post_id"]),p["title"],p["body"],p["subreddit"],p["author"],p["created_utc"],p.get("url",""),[Comment(str(c["comment_id"]),str(c["author"]),str(c["body"]),str(c["created_utc"]),str(c.get("parent_id",""))) for c in p.get("comments",[])]) for p in data]
