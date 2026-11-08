import json
import os
import re
from collections import Counter

from dotenv import load_dotenv
from ollama import Client

from .temporal import parse_time_expression

load_dotenv()


class LLMService:
    """
    Handles:
    1. LLM-based information extraction
    2. Query understanding
    3. Grounded answer generation

    The code deliberately contains deterministic fallbacks because
    small local LLMs can sometimes return incomplete JSON.
    """

    def __init__(self):
        self.client = Client(
            host=os.getenv(
                "OLLAMA_HOST",
                "http://localhost:11434"
            )
        )

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:3b"
        )

    # ============================================================
    # BASIC LLM HELPERS
    # ============================================================

    def _chat(self, prompt):
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        return response["message"]["content"]

    def _json(self, text):
        """
        Extract JSON from:
        {...}
        or
        ```json
        {...}
        ```
        """

        if not text:
            return ""

        text = text.strip()

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"```$",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return text[start:end + 1]

        return text

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _clean_text(value):
        return str(value or "").strip()

    @staticmethod
    def _normalize_list(value):
        """
        Converts:
        None
        string
        list
        dict

        into a clean list of strings.
        """

        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if isinstance(value, dict):
            value = [
                value.get("name")
                or value.get("topic")
                or value.get("entity")
                or ""
            ]

        if not isinstance(value, list):
            return []

        output = []

        for item in value:

            if isinstance(item, dict):
                item = (
                    item.get("name")
                    or item.get("topic")
                    or item.get("entity")
                    or ""
                )

            item = str(item).strip()

            if item and item not in output:
                output.append(item)

        return output

    @staticmethod
    def _normalize_relationships(value):

        if value is None:
            return []

        if isinstance(value, dict):
            value = [value]

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            return []

        output = []

        for item in value:

            if isinstance(item, dict):

                source = str(
                    item.get("source", "")
                ).strip()

                target = str(
                    item.get("target", "")
                ).strip()

                relation = str(
                    item.get("relation")
                    or item.get("type")
                    or "related_to"
                ).strip()

                if source and target:
                    output.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": relation
                        }
                    )

                continue

            if isinstance(item, str):

                text = item.strip()

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

                        source = parts[0].strip()
                        target = parts[1].strip()

                        if source and target:

                            output.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "relation": "related_to"
                                }
                            )

                        break

        # Remove duplicates
        unique = []

        seen = set()

        for rel in output:

            key = (
                rel["source"].lower(),
                rel["target"].lower(),
                rel["relation"].lower()
            )

            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    def _normalize_extraction(self, data):

        if not isinstance(data, dict):
            return {}

        topics = self._normalize_list(
            data.get("topics")
        )

        entities = self._normalize_list(
            data.get("entities")
        )

        relationships = self._normalize_relationships(
            data.get("relationships")
        )

        sentiment = data.get(
            "sentiment",
            "neutral"
        )

        if isinstance(sentiment, dict):
            sentiment = (
                sentiment.get("label")
                or sentiment.get("sentiment")
                or "neutral"
            )

        sentiment = str(
            sentiment
        ).strip().lower()

        if sentiment not in {
            "positive",
            "negative",
            "neutral",
            "mixed"
        }:
            sentiment = "neutral"

        return {
            "topics": topics,
            "entities": entities,
            "sentiment": sentiment,
            "relationships": relationships
        }

    # ============================================================
    # DETERMINISTIC EXTRACTION
    # ============================================================

    def _fallback_extract(self, title, body):

        text = (
            f"{title} {body}"
        ).lower()

        # --------------------------------------------
        # TOPIC RULES
        # --------------------------------------------

        topic_rules = {

            "local llms": [
                "local llm",
                "local llama",
                "local model",
                "on-device",
                "on device"
            ],

            "quantization": [
                "quantization",
                "quantized",
                "gguf",
                "4-bit",
                "4 bit",
                "8-bit",
                "8 bit"
            ],

            "rag": [
                "rag",
                "retrieval augmented",
                "retrieval-augmented"
            ],

            "graph retrieval": [
                "graph retrieval",
                "graph-based retrieval",
                "graph based retrieval",
                "knowledge graph",
                "graphrag",
                "graph rag"
            ],

            "ai safety": [
                "ai safety",
                "ai-safety",
                "alignment",
                "safety evaluation",
                "model safety"
            ],

            "ai governance": [
                "governance",
                "regulation",
                "regulatory",
                "policy",
                "policies"
            ],

            "efficient inference": [
                "inference",
                "latency",
                "memory efficient",
                "memory-efficient",
                "efficient inference"
            ],

            "open source models": [
                "open source",
                "open-source",
                "open models"
            ],

            "smaller models": [
                "small model",
                "smaller model",
                "small models",
                "smaller models",
                "tiny model",
                "tiny models"
            ],

            "coding assistants": [
                "coding assistant",
                "coding assistants",
                "code generation",
                "code assistant",
                "coding"
            ],

            "llm evaluation": [
                "evaluation",
                "benchmark",
                "benchmarks",
                "evals"
            ],

            "model comparison": [
                "versus",
                " vs ",
                "comparison",
                "compare"
            ]
        }

        topics = []

        for topic, terms in topic_rules.items():

            if any(term in text for term in terms):
                topics.append(topic)

        # --------------------------------------------
        # ENTITY RULES
        # --------------------------------------------

        entity_rules = {

            "RAG": [
                "rag",
                "retrieval augmented"
            ],

            "GraphRAG": [
                "graphrag",
                "graph rag"
            ],

            "Graph Retrieval": [
                "graph retrieval",
                "graph-based retrieval",
                "graph based retrieval"
            ],

            "Knowledge Graph": [
                "knowledge graph"
            ],

            "Llama": [
                "llama"
            ],

            "Gemma": [
                "gemma"
            ],

            "GGUF": [
                "gguf"
            ],

            "Ollama": [
                "ollama"
            ],

            "Neo4j": [
                "neo4j"
            ],

            "OpenAI": [
                "openai"
            ],

            "AI": [
                "artificial intelligence",
                " ai ",
                "ai safety",
                "ai governance"
            ],

            "Python": [
                "python"
            ],

            "Hugging Face": [
                "hugging face",
                "huggingface"
            ]
        }

        entities = []

        padded_text = f" {text} "

        for entity, terms in entity_rules.items():

            if any(
                term in padded_text
                for term in terms
            ):
                entities.append(entity)

        # --------------------------------------------
        # RELATIONSHIPS
        # --------------------------------------------

        relationships = []

        entity_set = {
            x.lower(): x
            for x in entities
        }

        def add_relation(source, target, relation):

            if (
                source.lower() in entity_set
                and target.lower() in entity_set
            ):

                relationships.append(
                    {
                        "source": entity_set[source.lower()],
                        "target": entity_set[target.lower()],
                        "relation": relation
                    }
                )

        if "rag" in entity_set and "graphrag" in entity_set:
            add_relation(
                "RAG",
                "GraphRAG",
                "related_to"
            )

        if (
            "rag" in entity_set
            and "graph retrieval" in entity_set
        ):
            add_relation(
                "RAG",
                "Graph Retrieval",
                "related_to"
            )

        if (
            "graph retrieval" in entity_set
            and "knowledge graph" in entity_set
        ):
            add_relation(
                "Graph Retrieval",
                "Knowledge Graph",
                "uses"
            )

        if (
            "llama" in entity_set
            and "gemma" in entity_set
        ):
            add_relation(
                "Llama",
                "Gemma",
                "compared_with"
            )

        if (
            "ollama" in entity_set
            and "llama" in entity_set
        ):
            add_relation(
                "Ollama",
                "Llama",
                "runs"
            )

        if (
            "neo4j" in entity_set
            and "graphrag" in entity_set
        ):
            add_relation(
                "GraphRAG",
                "Neo4j",
                "uses"
            )

        # --------------------------------------------
        # SENTIMENT
        # --------------------------------------------

        positive_words = [
            "excellent",
            "great",
            "good",
            "useful",
            "promising",
            "improving",
            "better",
            "success"
        ]

        negative_words = [
            "bad",
            "poor",
            "problem",
            "problems",
            "risk",
            "risks",
            "concern",
            "concerns",
            "difficult",
            "issue",
            "issues"
        ]

        positive_count = sum(
            text.count(word)
            for word in positive_words
        )

        negative_count = sum(
            text.count(word)
            for word in negative_words
        )

        if (
            positive_count > 0
            and negative_count > 0
        ):
            sentiment = "mixed"

        elif positive_count > negative_count:
            sentiment = "positive"

        elif negative_count > positive_count:
            sentiment = "negative"

        else:
            sentiment = "neutral"

        return {
            "topics": topics,
            "entities": entities,
            "sentiment": sentiment,
            "relationships": relationships
        }

    # ============================================================
    # LLM EXTRACTION
    # ============================================================

    def extract_information(
        self,
        title,
        body
    ):

        fallback = self._fallback_extract(
            title,
            body
        )

        prompt = f"""
You are extracting structured information from a Reddit post.

Return ONLY valid JSON.

Required schema:

{{
  "topics": ["topic 1", "topic 2"],
  "entities": ["entity 1", "entity 2"],
  "sentiment": "positive|negative|neutral|mixed",
  "relationships": [
    {{
      "source": "entity",
      "relation": "relationship type",
      "target": "entity"
    }}
  ]
}}

Rules:

1. Topics must describe actual discussion themes.
2. Entities must be actual named technologies, models,
   organizations, concepts or systems mentioned in the text.
3. Do not invent entities.
4. Relationships must connect entities that are actually
   related in the post.
5. If no relationship is explicit, return [].
6. Keep topic/entity names short and consistent.
7. Do not write explanations outside JSON.

TITLE:
{title}

BODY:
{body}
"""

        try:

            raw = self._chat(prompt)

            parsed = json.loads(
                self._json(raw)
            )

            llm_data = self._normalize_extraction(
                parsed
            )

            # ------------------------------------------------
            # Merge deterministic extraction with LLM output.
            # This prevents a valid-but-empty LLM response
            # from destroying graph information.
            # ------------------------------------------------

            topics = list(
                dict.fromkeys(
                    llm_data["topics"]
                    + fallback["topics"]
                )
            )

            entities = list(
                dict.fromkeys(
                    llm_data["entities"]
                    + fallback["entities"]
                )
            )

            relationships = (
                llm_data["relationships"]
                + fallback["relationships"]
            )

            # Remove duplicate relationships
            unique_relationships = []

            seen = set()

            for rel in relationships:

                key = (
                    rel["source"].lower(),
                    rel["target"].lower(),
                    rel["relation"].lower()
                )

                if key not in seen:

                    seen.add(key)

                    unique_relationships.append(
                        rel
                    )

            sentiment = llm_data["sentiment"]

            if (
                sentiment == "neutral"
                and fallback["sentiment"] != "neutral"
            ):
                sentiment = fallback["sentiment"]

            return {
                "topics": topics,
                "entities": entities,
                "sentiment": sentiment,
                "relationships": unique_relationships
            }

        except Exception:

            return fallback

    # ============================================================
    # QUERY UNDERSTANDING
    # ============================================================

    def understand_query(self, question):

        temporal = parse_time_expression(
            question
        )

        q = question.lower().strip()

        # --------------------------------------------------------
        # Detect comparison independently of temporal expressions.
        # --------------------------------------------------------

        comparison_words = [
            "difference",
            "differences",
            "compare",
            "comparison",
            "versus",
            "vs",
            "against",
            "change",
            "changed",
            "how has",
            "how have"
        ]

        is_comparison = any(
            word in q
            for word in comparison_words
        )

        # --------------------------------------------------------
        # Routing priority
        #
        # Temporal only when actual periods exist.
        # Graph when relationship/traversal is explicitly requested.
        # Comparison without dates -> hybrid.
        # --------------------------------------------------------

        if (
            len(temporal["periods"]) >= 2
            or (
                temporal["comparison"]
                and len(temporal["periods"]) >= 1
            )
        ):
            query_type = "temporal"

        elif any(
            phrase in q
            for phrase in [
                "connected to",
                "connected with",
                "related to",
                "relationship between",
                "relationships between",
                "linked to",
                "which topics are connected",
                "what is connected to",
                "what entities are connected"
            ]
        ):
            query_type = "graph"

        elif is_comparison:
            query_type = "hybrid"

        elif any(
            phrase in q
            for phrase in [
                "what are people discussing",
                "what do people discuss",
                "what are people talking about",
                "main topics",
                "what is being discussed"
            ]
        ):
            query_type = "hybrid"

        else:
            query_type = "semantic"

        # --------------------------------------------------------
        # Topic detection
        # --------------------------------------------------------

        topic_aliases = [
            ("local llms", [
                "local llm",
                "local llms"
            ]),
            ("quantization", [
                "quantization",
                "quantized",
                "gguf"
            ]),
            ("rag", [
                "rag",
                "retrieval augmented"
            ]),
            ("graph retrieval", [
                "graph retrieval",
                "graph-based retrieval",
                "graph based retrieval"
            ]),
            ("ai safety", [
                "ai safety",
                "alignment"
            ]),
            ("ai governance", [
                "ai governance",
                "regulation",
                "regulatory"
            ]),
            ("open source models", [
                "open source",
                "open-source"
            ]),
            ("smaller models", [
                "smaller models",
                "small models"
            ]),
            ("efficient inference", [
                "efficient inference",
                "inference"
            ])
        ]

        topic = ""

        for canonical, aliases in topic_aliases:

            if any(
                alias in q
                for alias in aliases
            ):
                topic = canonical
                break

        # --------------------------------------------------------
        # Entity detection
        # --------------------------------------------------------

        entity_aliases = [
            ("rag", ["rag"]),
            ("llama", ["llama"]),
            ("gemma", ["gemma"]),
            ("ollama", ["ollama"]),
            ("neo4j", ["neo4j"]),
            ("graphrag", ["graphrag", "graph rag"]),
            ("gguf", ["gguf"]),
            ("graph retrieval", ["graph retrieval"])
        ]

        entity = ""

        for canonical, aliases in entity_aliases:

            if any(
                alias in q
                for alias in aliases
            ):
                entity = canonical
                break

        return {
            "query_type": query_type,
            "keywords": re.findall(
                r"[a-zA-Z][a-zA-Z0-9_-]+",
                q
            )[:12],
            "topic": topic,
            "entity": entity,
            "time_expression": (
                question
                if temporal["periods"]
                else ""
            ),
            "comparison": (
                is_comparison
                or temporal["comparison"]
            ),
            "periods": temporal["periods"]
        }

    # ============================================================
    # EVIDENCE
    # ============================================================

    def _evidence(self, evidence):

        lines = []

        for item in evidence:

            lines.append(
                "[SOURCE "
                f"{item.get('id')}] "
                f"title={item.get('title', '')} | "
                f"date={item.get('created_utc', '')} | "
                f"subreddit={item.get('subreddit', '')} | "
                f"topics={item.get('topics', [])} | "
                f"entities={item.get('entities', [])} | "
                f"text={str(item.get('text', item.get('body', '')))[:500]}"
            )

        return "\n".join(lines)

    def _valid_sources(
        self,
        answer,
        evidence
    ):

        found = set(
            re.findall(
                r"\[\s*SOURCE\s+([^\]\s]+)\s*\]",
                answer,
                flags=re.IGNORECASE
            )
        )

        valid = {
            str(
                item.get("id")
                or item.get("post_id")
            )
            for item in evidence
        }

        return found.issubset(valid)

    # ============================================================
    # FALLBACK ANSWERS
    # ============================================================

    def _fallback(self, evidence):

        if not evidence:

            return (
                "There is not enough retrieved evidence "
                "to answer this question."
            )

        counts = Counter(
            topic
            for item in evidence
            for topic in item.get(
                "topics",
                []
            )
        )

        topics = [
            topic
            for topic, _ in counts.most_common(5)
        ]

        source_ids = [
            str(item.get("id"))
            for item in evidence[:5]
        ]

        if topics:

            return (
                "The main retrieved topics are "
                + ", ".join(topics)
                + ". "
                + "Relevant sources: "
                + ", ".join(source_ids)
                + "."
            )

        return (
            "The retrieved evidence does not contain "
            "enough extracted topic information. "
            "Relevant sources: "
            + ", ".join(source_ids)
            + "."
        )

    def _temporal_fallback(
        self,
        period_results
    ):

        lines = []

        for label, results in period_results:

            counts = Counter(
                topic
                for item in results
                for topic in item.get(
                    "topics",
                    []
                )
            )

            topics = [
                topic
                for topic, _ in counts.most_common(5)
            ]

            source_ids = [
                str(item.get("id"))
                for item in results[:5]
            ]

            lines.append(
                f"{label}: "
                + (
                    ", ".join(topics)
                    if topics
                    else "no extracted topics"
                )
                + " | sources: "
                + (
                    ", ".join(source_ids)
                    if source_ids
                    else "none"
                )
            )

        return (
            "Side-by-side period comparison:\n"
            + "\n".join(lines)
        )

    def _graph_fallback(
        self,
        graph_info
    ):

        if not graph_info:

            return (
                "No graph relationships were found "
                "for the requested entity."
            )

        lines = []

        for item in graph_info:

            lines.append(
                "- "
                f"{item.get('source')} "
                f"--{item.get('relation', 'RELATED_TO')}--> "
                f"{item.get('target')}"
            )

        return (
            "Graph relationships found:\n"
            + "\n".join(lines)
        )

    # ============================================================
    # ANSWER GENERATION
    # ============================================================

    def generate_answer(
        self,
        question,
        route,
        evidence,
        graph_info=None,
        period_results=None
    ):

        # --------------------------------------------------------
        # Build complete evidence set.
        # For temporal queries this includes both periods.
        # --------------------------------------------------------

        all_evidence = list(
            evidence or []
        )

        if period_results:

            for _, results in period_results:

                for item in results:

                    item_id = str(
                        item.get("id")
                    )

                    if not any(
                        str(x.get("id"))
                        == item_id
                        for x in all_evidence
                    ):
                        all_evidence.append(
                            item
                        )

        # --------------------------------------------------------
        # Immediate deterministic fallbacks
        # --------------------------------------------------------

        if route == "graph":

            if not graph_info:
                return self._graph_fallback(
                    graph_info
                )

        if (
            route == "temporal"
            and period_results
        ):

            fallback = self._temporal_fallback(
                period_results
            )

        else:

            fallback = self._fallback(
                all_evidence
            )

        if (
            not all_evidence
            and not graph_info
        ):

            return fallback

        # --------------------------------------------------------
        # Prepare period evidence separately.
        # This is critical for temporal comparisons.
        # --------------------------------------------------------

        period_text = ""

        if period_results:

            sections = []

            for label, results in period_results:

                sections.append(
                    f"PERIOD: {label}\n"
                    + self._evidence(results)
                )

            period_text = "\n\n".join(
                sections
            )

        prompt = f"""
You are answering a Reddit intelligence question.

Use ONLY the supplied evidence.

STRICT RULES:

1. Never invent facts.
2. Never invent relationships.
3. Never invent dates.
4. Never invent source IDs.
5. Never introduce concepts that are not supported by evidence.
6. Do not expand acronyms unless the evidence explicitly defines them.
7. Every factual claim about a specific post must have
   a [SOURCE post_id] citation.
8. For temporal comparisons, keep each period separate.
9. Do not move a source from one period into another period.
10. If the evidence is insufficient, explicitly say so.
11. Prefer a concise evidence-based answer over speculation.

QUESTION:
{question}

ROUTE:
{route}

GENERAL EVIDENCE:
{self._evidence(all_evidence)}

TEMPORAL PERIOD EVIDENCE:
{period_text}

GRAPH RELATIONSHIPS:
{json.dumps(graph_info or [], ensure_ascii=False)}
"""

        try:

            answer = self._chat(
                prompt
            ).strip()

            if self._valid_sources(
                answer,
                all_evidence
            ):
                return answer

        except Exception:
            pass

        return fallback