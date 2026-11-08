from dataclasses import dataclass, field
from typing import List

@dataclass
class Comment:
    comment_id: str
    author: str
    body: str
    created_utc: str
    parent_id: str = ""

@dataclass
class RedditPost:
    post_id: str
    title: str
    body: str
    subreddit: str
    author: str
    created_utc: str
    url: str = ""
    comments: List[Comment] = field(default_factory=list)
