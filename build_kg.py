"""Build the assignment KG from SQLite into Neo4j."""

import os
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _guess_rule_type(sentence: str) -> str:
    lower = sentence.lower()
    if any(k in lower for k in ["must", "shall", "required", "should"]):
        return "requirement"
    if any(k in lower for k in ["cannot", "must not", "prohibit", "not allowed", "no "]):
        return "prohibition"
    if any(k in lower for k in ["penalty", "deduct", "zero score", "disciplinary", "fee"]):
        return "penalty"
    if any(k in lower for k in ["may", "can", "allowed", "eligible"]):
        return "permission"
    return "general"


def _split_sentences(content: str) -> list[str]:
    chunks = re.split(r"(?<=[\.\?!;])\s+|\n+", content)
    sentences = [_normalize(c) for c in chunks if _normalize(c)]
    return sentences


def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """Extract candidate rule facts from one article."""
    rules: list[dict[str, str]] = []
    sentences = _split_sentences(content)
    if not sentences:
        return {"rules": rules}

    for sent in sentences:
        sent_norm = _normalize(sent)
        if len(sent_norm) < 12:
            continue
        rtype = _guess_rule_type(sent_norm)
        # Keep explicit sanction fragments as result; otherwise mirror the sentence.
        sanction_match = re.search(
            r"(deduct(?:ion)?\s*\d+\s*points?|zero score|disciplinary action|"
            r"\d+\s*(?:minutes?|hours?|days?|years?|credits?|points?|NTD))",
            sent_norm,
            flags=re.IGNORECASE,
        )
        result = sanction_match.group(1) if sanction_match else sent_norm
        rules.append(
            {
                "type": rtype,
                "action": sent_norm,
                "result": _normalize(result),
                "art_ref": article_number,
                "reg_name": reg_name,
            }
        )

    return {"rules": rules}


def build_fallback_rules(article_number: str, content: str) -> list[dict[str, str]]:
    """Add fallback rule if extraction returns empty."""
    content_norm = _normalize(content)
    if not content_norm:
        return []
    return [
        {
            "type": "general",
            "action": content_norm[:500],
            "result": content_norm[:220],
            "art_ref": article_number,
            "reg_name": "",
        }
    ]


# SQLite tables used:
# - regulations(reg_id, name, category)
# - articles(reg_id, article_number, content)


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    with driver.session() as session:
        # Fixed strategy: clear existing graph data before rebuilding.
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Read regulations and create Regulation nodes.
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        # 2) Read articles and create Article + HAS_ARTICLE.
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        # 3) Create full-text index on Article content.
        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        rule_counter = 0
        dedup_keys: set[str] = set()

        for reg_id, article_number, content in articles:
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))
            parsed = extract_entities(article_number, reg_name, content)
            rules = parsed.get("rules", [])
            if not rules:
                rules = build_fallback_rules(article_number, content)
                for r in rules:
                    r["reg_name"] = reg_name

            for rule in rules:
                action = _normalize(str(rule.get("action", "")))
                result = _normalize(str(rule.get("result", "")))
                if not action or not result:
                    continue

                dedup_key = f"{reg_name}|{article_number}|{action.lower()}|{result.lower()}"
                if dedup_key in dedup_keys:
                    continue
                dedup_keys.add(dedup_key)

                rule_counter += 1
                rule_id = f"R{rule_counter:05d}"
                session.run(
                    """
                    MATCH (a:Article {number:$num, reg_name:$reg_name})
                    CREATE (r:Rule {
                        rule_id: $rule_id,
                        type: $rtype,
                        action: $action,
                        result: $result,
                        art_ref: $art_ref,
                        reg_name: $reg_name
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    num=article_number,
                    reg_name=reg_name,
                    rule_id=rule_id,
                    rtype=str(rule.get("type", "general")),
                    action=action,
                    result=result,
                    art_ref=article_number,
                )

        # 4) Create full-text index on Rule fields.
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result]
            """
        )

        # 5) Coverage audit (provided scaffold).
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )
        print(f"[Rules] total={rule_counter}")

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()
