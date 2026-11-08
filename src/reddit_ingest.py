import os
from datetime import datetime, timezone
import praw
from .models import RedditPost, Comment

def load_reddit_posts():
    if not os.getenv("REDDIT_CLIENT_ID") or not os.getenv("REDDIT_CLIENT_SECRET"):
        raise RuntimeError("Reddit API credentials are missing from .env")
    reddit=praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),client_secret=os.getenv("REDDIT_CLIENT_SECRET"),user_agent=os.getenv("REDDIT_USER_AGENT","HybridGraphRAG/1.0"))
    posts=[]
    for name in [x.strip() for x in os.getenv("SUBREDDITS","LocalLLaMA,MachineLearning,artificial").split(",") if x.strip()]:
        for s in reddit.subreddit(name).hot(limit=int(os.getenv("POST_LIMIT_PER_SUBREDDIT","50"))):
            s.comments.replace_more(limit=0); comments=[]
            for c in s.comments.list()[:int(os.getenv("COMMENTS_PER_POST","30"))]:
                comments.append(Comment(str(c.id),str(c.author) if c.author else "[deleted]",str(c.body),datetime.fromtimestamp(c.created_utc,tz=timezone.utc).isoformat().replace("+00:00","Z"),str(c.parent_id or "")))
            posts.append(RedditPost(str(s.id),str(s.title),str(s.selftext or ""),str(s.subreddit.display_name),str(s.author) if s.author else "[deleted]",datetime.fromtimestamp(s.created_utc,tz=timezone.utc).isoformat().replace("+00:00","Z"),f"https://www.reddit.com{s.permalink}",comments))
    return posts
