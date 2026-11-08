import os
from datetime import datetime, timezone

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self):

        path = os.getenv(
            "CHROMA_PATH",
            "./chroma_db"
        )

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="reddit_posts"
            )
        )

    # ============================================================
    # DATE
    # ============================================================

    def _to_epoch(self, value):

        if isinstance(
            value,
            (int, float)
        ):
            return float(value)

        if not value:
            return 0.0

        try:

            value = str(value).replace(
                "Z",
                "+00:00"
            )

            dt = datetime.fromisoformat(
                value
            )

            if dt.tzinfo is None:

                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            return dt.timestamp()

        except Exception:

            return 0.0

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        try:

            self.client.delete_collection(
                "reddit_posts"
            )

        except Exception:

            pass

        self.collection = (
            self.client.get_or_create_collection(
                name="reddit_posts"
            )
        )

    # ============================================================
    # ADD DOCUMENT
    # ============================================================

    def add_document(
        self,
        document_id,
        text,
        metadata
    ):

        embedding = (
            self.embedder
            .encode(
                text,
                normalize_embeddings=True
            )
            .tolist()
        )

        created_utc = metadata.get(
            "created_utc",
            ""
        )

        # Chroma metadata values should be
        # scalar values, so lists are stored
        # as comma-separated strings.

        topics = metadata.get(
            "topics",
            []
        )

        entities = metadata.get(
            "entities",
            []
        )

        if isinstance(
            topics,
            list
        ):

            topics = ", ".join(
                str(x)
                for x in topics
            )

        if isinstance(
            entities,
            list
        ):

            entities = ", ".join(
                str(x)
                for x in entities
            )

        clean_metadata = {

            "subreddit": str(
                metadata.get(
                    "subreddit",
                    ""
                )
            ),

            "author": str(
                metadata.get(
                    "author",
                    ""
                )
            ),

            "created_utc": str(
                created_utc
            ),

            "created_utc_epoch": float(
                self._to_epoch(
                    created_utc
                )
            ),

            "source": str(
                metadata.get(
                    "source",
                    "reddit"
                )
            ),

            "url": str(
                metadata.get(
                    "url",
                    ""
                )
            ),

            "title": str(
                metadata.get(
                    "title",
                    ""
                )
            ),

            "topics": str(
                topics
            ),

            "entities": str(
                entities
            )
        }

        self.collection.add(

            ids=[
                str(document_id)
            ],

            documents=[
                text
            ],

            embeddings=[
                embedding
            ],

            metadatas=[
                clean_metadata
            ]
        )

    # ============================================================
    # ADD POST
    # ============================================================

    def add_post(
        self,
        post,
        extracted=None
    ):

        extracted = extracted or {}

        text = (
            f"{post.title}\n"
            f"{post.body}"
        ).strip()

        metadata = {

            "subreddit":
                post.subreddit,

            "author":
                post.author,

            "created_utc":
                post.created_utc,

            "source":
                "reddit"
                if post.url
                else "sample",

            "url":
                post.url,

            "title":
                post.title,

            "topics":
                extracted.get(
                    "topics",
                    []
                ),

            "entities":
                extracted.get(
                    "entities",
                    []
                )
        }

        self.add_document(
            post.post_id,
            text,
            metadata
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query,
        n=5,
        start_date=None,
        end_date=None,
        subreddit=None,
        author=None,
        topic=None,
        entity=None
    ):

        query_embedding = (
            self.embedder
            .encode(
                query,
                normalize_embeddings=True
            )
            .tolist()
        )

        filters = []

        # --------------------------------------------
        # DATE
        # --------------------------------------------

        if start_date:

            filters.append(
                {
                    "created_utc_epoch": {
                        "$gte": float(
                            self._to_epoch(
                                start_date
                            )
                        )
                    }
                }
            )

        if end_date:

            filters.append(
                {
                    "created_utc_epoch": {
                        "$lt": float(
                            self._to_epoch(
                                end_date
                            )
                        )
                    }
                }
            )

        # --------------------------------------------
        # SUBREDDIT
        # --------------------------------------------

        if subreddit:

            filters.append(
                {
                    "subreddit": str(
                        subreddit
                    )
                }
            )

        # --------------------------------------------
        # AUTHOR
        # --------------------------------------------

        if author:

            filters.append(
                {
                    "author": str(
                        author
                    )
                }
            )

        # --------------------------------------------
        # TOPIC
        # --------------------------------------------

        if topic:

            filters.append(
                {
                    "topics": {
                        "$contains": str(
                            topic
                        )
                    }
                }
            )

        # --------------------------------------------
        # ENTITY
        # --------------------------------------------

        if entity:

            filters.append(
                {
                    "entities": {
                        "$contains": str(
                            entity
                        )
                    }
                }
            )

        where = None

        if len(filters) == 1:

            where = filters[0]

        elif len(filters) > 1:

            where = {
                "$and": filters
            }

        kwargs = {

            "query_embeddings": [
                query_embedding
            ],

            "n_results": int(n)
        }

        if where:

            kwargs["where"] = where

        return self.collection.query(
            **kwargs
        )