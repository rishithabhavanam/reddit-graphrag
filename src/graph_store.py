import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


class GraphStore:

    def __init__(self):

        uri = os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687"
        )

        username = os.getenv(
            "NEO4J_USERNAME",
            "neo4j"
        )

        password = os.getenv(
            "NEO4J_PASSWORD",
            ""
        )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(
                username,
                password
            )
        )

        self.driver.verify_connectivity()

    # ============================================================
    # CONNECTION
    # ============================================================

    def close(self):

        self.driver.close()

    # ============================================================
    # DATABASE
    # ============================================================

    def clear_database(self):

        with self.driver.session() as session:

            session.run(
                "MATCH (n) DETACH DELETE n"
            )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _safe_list(value):

        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    @staticmethod
    def _normalize_relationships(
        relationships
    ):

        normalized = []

        for rel in GraphStore._safe_list(
            relationships
        ):

            # --------------------------------------------
            # Dictionary
            # --------------------------------------------

            if isinstance(rel, dict):

                source = str(
                    rel.get("source", "")
                ).strip()

                target = str(
                    rel.get("target", "")
                ).strip()

                relation = str(
                    rel.get("relation")
                    or rel.get("type")
                    or "related_to"
                ).strip()

                if source and target:

                    normalized.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": relation
                        }
                    )

                continue

            # --------------------------------------------
            # String
            # --------------------------------------------

            if isinstance(rel, str):

                text = rel.strip()

                if not text:
                    continue

                separators = [
                    "->",
                    "→",
                    " to ",
                    " - ",
                    ":"
                ]

                for separator in separators:

                    if separator in text:

                        parts = text.split(
                            separator,
                            1
                        )

                        source = (
                            parts[0]
                            .strip()
                        )

                        target = (
                            parts[1]
                            .strip()
                        )

                        if source and target:

                            normalized.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "relation": "related_to"
                                }
                            )

                        break

        # --------------------------------------------
        # Remove duplicates
        # --------------------------------------------

        unique = []

        seen = set()

        for rel in normalized:

            key = (
                rel["source"].lower(),
                rel["target"].lower(),
                rel["relation"].lower()
            )

            if key not in seen:

                seen.add(key)
                unique.append(rel)

        return unique

    # ============================================================
    # ADD POST
    # ============================================================

    def add_post(
        self,
        post,
        extracted=None
    ):

        extracted = extracted or {}

        with self.driver.session() as session:

            session.execute_write(
                self._create,
                post,
                extracted
            )

    @staticmethod
    def _create(
        tx,
        post,
        extracted
    ):

        post_id = str(
            post.post_id
        )

        # --------------------------------------------------------
        # POST
        # --------------------------------------------------------

        tx.run(
            """
            MERGE (p:Post {id: $id})
            SET
                p.title = $title,
                p.body = $body,
                p.created_utc = $created_utc,
                p.url = $url
            """,

            id=post_id,
            title=str(
                post.title or ""
            ),
            body=str(
                post.body or ""
            ),
            created_utc=str(
                post.created_utc or ""
            ),
            url=str(
                post.url or ""
            )
        )

        # --------------------------------------------------------
        # SUBREDDIT
        # --------------------------------------------------------

        subreddit = str(
            post.subreddit or ""
        ).strip()

        if subreddit:

            tx.run(
                """
                MERGE (s:Subreddit {name: $name})

                WITH s

                MATCH (p:Post {id: $post_id})

                MERGE (p)-[:POSTED_IN]->(s)
                """,

                name=subreddit,
                post_id=post_id
            )

        # --------------------------------------------------------
        # AUTHOR
        # --------------------------------------------------------

        author = str(
            post.author or ""
        ).strip()

        if author:

            tx.run(
                """
                MERGE (a:Author {name: $name})

                WITH a

                MATCH (p:Post {id: $post_id})

                MERGE (p)-[:AUTHORED_BY]->(a)
                """,

                name=author,
                post_id=post_id
            )

        # --------------------------------------------------------
        # TOPICS
        # --------------------------------------------------------

        topics = extracted.get(
            "topics",
            []
        )

        for topic in GraphStore._safe_list(
            topics
        ):

            if isinstance(topic, dict):

                topic_name = str(
                    topic.get("name")
                    or topic.get("topic")
                    or ""
                ).strip()

            else:

                topic_name = str(
                    topic
                ).strip()

            if not topic_name:
                continue

            tx.run(
                """
                MERGE (t:Topic {name: $name})

                WITH t

                MATCH (p:Post {id: $post_id})

                MERGE (p)-[:HAS_TOPIC]->(t)
                """,

                name=topic_name,
                post_id=post_id
            )

        # --------------------------------------------------------
        # ENTITIES
        # --------------------------------------------------------

        entities = extracted.get(
            "entities",
            []
        )

        entity_names = []

        for entity in GraphStore._safe_list(
            entities
        ):

            if isinstance(entity, dict):

                entity_name = str(
                    entity.get("name")
                    or entity.get("entity")
                    or ""
                ).strip()

            else:

                entity_name = str(
                    entity
                ).strip()

            if not entity_name:
                continue

            if entity_name not in entity_names:

                entity_names.append(
                    entity_name
                )

            tx.run(
                """
                MERGE (e:Entity {name: $name})

                WITH e

                MATCH (p:Post {id: $post_id})

                MERGE (p)-[:MENTIONS]->(e)
                """,

                name=entity_name,
                post_id=post_id
            )

        # --------------------------------------------------------
        # SENTIMENT
        # --------------------------------------------------------

        sentiment = extracted.get(
            "sentiment",
            ""
        )

        if isinstance(
            sentiment,
            dict
        ):

            sentiment = (
                sentiment.get("label")
                or sentiment.get("sentiment")
                or ""
            )

        sentiment = str(
            sentiment
        ).strip()

        if sentiment:

            tx.run(
                """
                MERGE (s:Sentiment {label: $label})

                WITH s

                MATCH (p:Post {id: $post_id})

                MERGE (p)-[:HAS_SENTIMENT]->(s)
                """,

                label=sentiment,
                post_id=post_id
            )

        # --------------------------------------------------------
        # ENTITY RELATIONSHIPS
        # --------------------------------------------------------

        relationships = (
            GraphStore
            ._normalize_relationships(
                extracted.get(
                    "relationships",
                    []
                )
            )
        )

        for rel in relationships:

            source = rel["source"]
            target = rel["target"]
            relation = rel["relation"]

            if (
                source.lower()
                == target.lower()
            ):
                continue

            tx.run(
                """
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})

                MERGE (a)-[
                    r:RELATED_TO
                ]->(b)

                SET r.relation = $relation
                """,

                source=source,
                target=target,
                relation=relation
            )

            # Ensure the post is connected to both entities

            tx.run(
                """
                MATCH (p:Post {id: $post_id})

                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})

                MERGE (p)-[:MENTIONS]->(a)
                MERGE (p)-[:MENTIONS]->(b)
                """,

                post_id=post_id,
                source=source,
                target=target
            )

        # --------------------------------------------------------
        # CO-OCCURRENCE FALLBACK
        # --------------------------------------------------------
        #
        # If multiple entities occur in the same post but the LLM
        # did not explicitly create a relationship, connect them
        # with a RELATED_TO relationship marked as CO_OCCURS_WITH.
        #
        # This makes graph traversal useful while remaining honest:
        # it represents co-occurrence, not a claimed semantic fact.
        # --------------------------------------------------------

        if len(entity_names) >= 2:

            for i in range(
                len(entity_names)
            ):

                for j in range(
                    i + 1,
                    len(entity_names)
                ):

                    source = entity_names[i]
                    target = entity_names[j]

                    tx.run(
                        """
                        MERGE (a:Entity {
                            name: $source
                        })

                        MERGE (b:Entity {
                            name: $target
                        })

                        MERGE (a)-[
                            r:RELATED_TO
                        ]-(b)

                        ON CREATE SET
                            r.relation =
                            'co_occurs_with'
                        """,

                        source=source,
                        target=target
                    )

        # --------------------------------------------------------
        # COMMENTS
        # --------------------------------------------------------

        comments = getattr(
            post,
            "comments",
            []
        ) or []

        for comment in comments:

            comment_id = str(
                getattr(
                    comment,
                    "comment_id",
                    ""
                )
            ).strip()

            if not comment_id:
                continue

            comment_body = str(
                getattr(
                    comment,
                    "body",
                    ""
                )
            )

            comment_author = str(
                getattr(
                    comment,
                    "author",
                    ""
                )
            )

            comment_created = str(
                getattr(
                    comment,
                    "created_utc",
                    ""
                )
            )

            tx.run(
                """
                MERGE (c:Comment {
                    id: $comment_id
                })

                SET
                    c.body = $body,
                    c.author = $author,
                    c.created_utc = $created_utc

                WITH c

                MATCH (p:Post {
                    id: $post_id
                })

                MERGE (p)-[:HAS_COMMENT]->(c)
                """,

                comment_id=comment_id,
                body=comment_body,
                author=comment_author,
                created_utc=comment_created,
                post_id=post_id
            )

    # ============================================================
    # GRAPH POST SEARCH
    # ============================================================

    def posts(
        self,
        q=None,
        start=None,
        end=None,
        subreddit=None,
        author=None,
        limit=10
    ):

        conditions = []

        params = {
            "limit": int(limit)
        }

        if q:

            conditions.append(
                """
                (
                    toLower(p.title)
                    CONTAINS toLower($q)

                    OR

                    toLower(p.body)
                    CONTAINS toLower($q)
                )
                """
            )

            params["q"] = q

        if subreddit:

            conditions.append(
                """
                toLower(s.name)
                =
                toLower($subreddit)
                """
            )

            params["subreddit"] = subreddit

        if author:

            conditions.append(
                """
                toLower(a.name)
                =
                toLower($author)
                """
            )

            params["author"] = author

        if start:

            conditions.append(
                "p.created_utc >= $start"
            )

            params["start"] = start

        if end:

            conditions.append(
                "p.created_utc < $end"
            )

            params["end"] = end

        where = ""

        if conditions:

            where = (
                "WHERE "
                + " AND ".join(
                    conditions
                )
            )

        query = f"""
        MATCH (p:Post)

        OPTIONAL MATCH
            (p)-[:POSTED_IN]->(s:Subreddit)

        OPTIONAL MATCH
            (p)-[:AUTHORED_BY]->(a:Author)

        OPTIONAL MATCH
            (p)-[:HAS_TOPIC]->(t:Topic)

        OPTIONAL MATCH
            (p)-[:MENTIONS]->(e:Entity)

        {where}

        RETURN
            p.id AS id,
            p.title AS title,
            p.body AS body,
            p.created_utc AS created_utc,
            p.url AS url,
            s.name AS subreddit,
            a.name AS author,
            collect(DISTINCT t.name) AS topics,
            collect(DISTINCT e.name) AS entities

        ORDER BY p.created_utc DESC

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                params
            )

            return [
                dict(record)
                for record in result
            ]

    # ============================================================
    # TOPIC SEARCH
    # ============================================================

    def topic_posts(
        self,
        topic,
        start=None,
        end=None,
        limit=10
    ):

        query = """
        MATCH
            (p:Post)-[:HAS_TOPIC]->(t:Topic)

        WHERE
            toLower(t.name)
            =
            toLower($topic)
        """

        params = {
            "topic": topic,
            "limit": int(limit)
        }

        if start:

            query += """
            AND p.created_utc >= $start
            """

            params["start"] = start

        if end:

            query += """
            AND p.created_utc < $end
            """

            params["end"] = end

        query += """

        OPTIONAL MATCH
            (p)-[:MENTIONS]->(e:Entity)

        RETURN
            p.id AS id,
            p.title AS title,
            p.body AS body,
            p.created_utc AS created_utc,
            p.url AS url,
            t.name AS topic,
            collect(DISTINCT e.name) AS entities

        ORDER BY p.created_utc DESC

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                params
            )

            return [
                dict(record)
                for record in result
            ]

    # ============================================================
    # ENTITY SEARCH
    # ============================================================

    def entity_posts(
        self,
        entity,
        start=None,
        end=None,
        limit=10
    ):

        query = """
        MATCH
            (p:Post)-[:MENTIONS]->(e:Entity)

        WHERE
            toLower(e.name)
            =
            toLower($entity)
        """

        params = {
            "entity": entity,
            "limit": int(limit)
        }

        if start:

            query += """
            AND p.created_utc >= $start
            """

            params["start"] = start

        if end:

            query += """
            AND p.created_utc < $end
            """

            params["end"] = end

        query += """

        OPTIONAL MATCH
            (p)-[:HAS_TOPIC]->(t:Topic)

        RETURN
            p.id AS id,
            p.title AS title,
            p.body AS body,
            p.created_utc AS created_utc,
            p.url AS url,
            e.name AS entity,
            collect(DISTINCT t.name) AS topics

        ORDER BY p.created_utc DESC

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                params
            )

            return [
                dict(record)
                for record in result
            ]

    # ============================================================
    # GRAPH TRAVERSAL
    # ============================================================

    def related_entities(
        self,
        entity,
        limit=10
    ):

        query = """
        MATCH (e:Entity)

        WHERE
            toLower(e.name)
            =
            toLower($entity)

        MATCH
            (e)-[r:RELATED_TO]-(related:Entity)

        RETURN DISTINCT
            e.name AS source,
            related.name AS target,
            coalesce(
                r.relation,
                'related_to'
            ) AS relation

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                {
                    "entity": entity,
                    "limit": int(limit)
                }
            )

            return [
                dict(record)
                for record in result
            ]

    # ============================================================
    # GRAPH TRAVERSAL WITH TOPICS
    # ============================================================

    def entity_context(
        self,
        entity,
        limit=10
    ):

        query = """
        MATCH (e:Entity)

        WHERE
            toLower(e.name)
            =
            toLower($entity)

        OPTIONAL MATCH
            (e)-[r:RELATED_TO]-(related:Entity)

        OPTIONAL MATCH
            (p:Post)-[:MENTIONS]->(related)

        OPTIONAL MATCH
            (p)-[:HAS_TOPIC]->(t:Topic)

        RETURN
            e.name AS source,
            related.name AS target,
            coalesce(
                r.relation,
                'related_to'
            ) AS relation,
            collect(
                DISTINCT t.name
            ) AS topics,
            collect(
                DISTINCT p.id
            ) AS posts

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                {
                    "entity": entity,
                    "limit": int(limit)
                }
            )

            return [
                dict(record)
                for record in result
            ]

    # ============================================================
    # GRAPH STATS
    # ============================================================

    def graph_stats(self):

        with self.driver.session() as session:

            nodes = session.run(
                """
                MATCH (n)
                RETURN count(n) AS count
                """
            ).single()["count"]

            relationships = session.run(
                """
                MATCH ()-[r]->()
                RETURN count(r) AS count
                """
            ).single()["count"]

            posts = session.run(
                """
                MATCH (p:Post)
                RETURN count(p) AS count
                """
            ).single()["count"]

            entities = session.run(
                """
                MATCH (e:Entity)
                RETURN count(e) AS count
                """
            ).single()["count"]

            topics = session.run(
                """
                MATCH (t:Topic)
                RETURN count(t) AS count
                """
            ).single()["count"]

            return {
                "nodes": nodes,
                "relationships": relationships,
                "posts": posts,
                "entities": entities,
                "topics": topics
            }