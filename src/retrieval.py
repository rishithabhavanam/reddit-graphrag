from datetime import datetime, timezone

from .graph_store import GraphStore
from .vector_store import VectorStore


# ============================================================
# DATE HELPERS
# ============================================================

def _to_datetime(value):

    if not value:
        return None

    try:

        value = str(value)

        if value.endswith("Z"):
            value = (
                value[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            value
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt

    except Exception:

        return None


def _filter_by_date(
    results,
    start_date=None,
    end_date=None
):

    start = _to_datetime(
        start_date
    )

    end = _to_datetime(
        end_date
    )

    if not start and not end:

        return results

    filtered = []

    for item in results:

        created = _to_datetime(
            item.get(
                "created_utc"
            )
        )

        if not created:
            continue

        if start and created < start:
            continue

        if end and created >= end:
            continue

        filtered.append(item)

    return filtered


# ============================================================
# VECTOR NORMALIZATION
# ============================================================

def _normalize_vector_results(
    results
):

    if not isinstance(
        results,
        dict
    ):
        return []

    ids = results.get(
        "ids",
        [[]]
    )

    documents = results.get(
        "documents",
        [[]]
    )

    metadatas = results.get(
        "metadatas",
        [[]]
    )

    distances = results.get(
        "distances",
        [[]]
    )

    if ids and isinstance(
        ids[0],
        list
    ):
        ids = ids[0]

    if documents and isinstance(
        documents[0],
        list
    ):
        documents = documents[0]

    if metadatas and isinstance(
        metadatas[0],
        list
    ):
        metadatas = metadatas[0]

    if distances and isinstance(
        distances[0],
        list
    ):
        distances = distances[0]

    normalized = []

    for i, item_id in enumerate(
        ids
    ):

        metadata = {}

        if (
            i < len(metadatas)
            and metadatas[i]
        ):
            metadata = dict(
                metadatas[i]
            )

        # Convert comma-separated metadata
        # back into lists.

        topics = metadata.get(
            "topics",
            ""
        )

        if isinstance(
            topics,
            str
        ):

            topics = [
                x.strip()
                for x in topics.split(",")
                if x.strip()
            ]

        entities = metadata.get(
            "entities",
            ""
        )

        if isinstance(
            entities,
            str
        ):

            entities = [
                x.strip()
                for x in entities.split(",")
                if x.strip()
            ]

        item = {

            "id":
                str(item_id),

            "text":
                (
                    documents[i]
                    if i < len(documents)
                    else ""
                ),

            "topics":
                topics,

            "entities":
                entities
        }

        item.update(
            metadata
        )

        # Restore lists after metadata update.

        item["topics"] = topics
        item["entities"] = entities

        if (
            i < len(distances)
            and distances[i] is not None
        ):

            item["distance"] = (
                distances[i]
            )

        normalized.append(
            item
        )

    return normalized


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    query,
    n=5,
    start_date=None,
    end_date=None,
    subreddit=None,
    author=None,
    topic=None,
    entity=None
):

    vector_store = (
        VectorStore()
    )

    raw_results = (
        vector_store.search(
            query=query,
            n=n,
            start_date=start_date,
            end_date=end_date,
            subreddit=subreddit,
            author=author,
            topic=topic,
            entity=entity
        )
    )

    results = (
        _normalize_vector_results(
            raw_results
        )
    )

    results = _filter_by_date(
        results,
        start_date=start_date,
        end_date=end_date
    )

    return results[:n]


# ============================================================
# GRAPH SEARCH
# ============================================================

def graph_search(
    query,
    n=5,
    topic=None,
    entity=None,
    subreddit=None,
    author=None,
    start_date=None,
    end_date=None
):

    graph_store = (
        GraphStore()
    )

    try:

        if topic:

            results = (
                graph_store.topic_posts(
                    topic=topic,
                    start=start_date,
                    end=end_date,
                    limit=n
                )
            )

        elif entity:

            results = (
                graph_store.entity_posts(
                    entity=entity,
                    start=start_date,
                    end=end_date,
                    limit=n
                )
            )

        else:

            results = (
                graph_store.posts(
                    q=query,
                    start=start_date,
                    end=end_date,
                    subreddit=subreddit,
                    author=author,
                    limit=n
                )
            )

        results = _filter_by_date(
            results,
            start_date=start_date,
            end_date=end_date
        )

        return results[:n]

    finally:

        graph_store.close()


# ============================================================
# GRAPH TRAVERSAL
# ============================================================

def graph_traversal(
    entity,
    n=10
):

    if not entity:

        return [], []

    graph_store = (
        GraphStore()
    )

    try:

        rows = (
            graph_store.entity_context(
                entity=entity,
                limit=n
            )
        )

        relationships = []

        related_topics = set()

        for row in rows:

            source = row.get(
                "source"
            )

            target = row.get(
                "target"
            )

            relation = row.get(
                "relation"
            )

            if (
                source
                and target
                and relation
            ):

                relationships.append(
                    {
                        "source":
                            source,

                        "relationship":
                            relation,

                        "target":
                            target
                    }
                )

            for topic in (
                row.get(
                    "topics",
                    []
                )
                or []
            ):

                if topic:

                    related_topics.add(
                        str(topic)
                    )

        return (
            relationships,
            sorted(
                related_topics
            )
        )

    finally:

        graph_store.close()