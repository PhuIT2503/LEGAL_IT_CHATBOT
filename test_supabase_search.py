import argparse
import os
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from supabase import create_client


def _get_env_or_prompt(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    return input(f"Enter {name}: ").strip()


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _fetch_parents(supabase: Any, parent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not parent_ids:
        return {}

    response = (
        supabase.table("legal_parent_chunks")
        .select("id,dieu_id,van_ban_id,content,metadata")
        .in_("id", parent_ids)
        .execute()
    )
    return {row["id"]: row for row in response.data or []}


def _search(
    supabase: Any,
    model: SentenceTransformer,
    query: str,
    match_count: int,
    include_parent: bool,
    max_chars: int,
) -> None:
    embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    response = supabase.rpc(
        "match_legal_child_chunks",
        {
            "query_embedding": embedding,
            "match_count": match_count,
        },
    ).execute()

    rows = response.data or []
    if not rows:
        print("No results.")
        return

    parents = {}
    if include_parent:
        parent_ids = []
        seen = set()
        for row in rows:
            parent_id = row.get("parent_id")
            if parent_id and parent_id not in seen:
                parent_ids.append(parent_id)
                seen.add(parent_id)
        parents = _fetch_parents(supabase, parent_ids)

    print(f"\nQuery: {query}")
    print(f"Results: {len(rows)}\n")

    for idx, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or {}
        source = metadata.get("source", row.get("van_ban_id", ""))
        similarity = row.get("similarity", 0.0)

        print(f"[{idx}] similarity={similarity:.4f}")
        print(f"source: {source}")
        print(f"chunk: {row.get('id')}")
        print(f"dieu_id: {row.get('dieu_id')}")
        print(f"type: {row.get('chunk_type')}")
        print(_shorten(row.get("content", ""), max_chars))

        if include_parent:
            parent = parents.get(row.get("parent_id"))
            if parent:
                print("\nParent context:")
                print(_shorten(parent.get("content", ""), max_chars))

        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Supabase vector search for VBPL chunks")
    parser.add_argument("--query", help="Question to search. If omitted, enter interactive mode.")
    parser.add_argument("--match-count", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--include-parent", action="store_true")
    parser.add_argument("--model", default="AITeamVN/Vietnamese_Embedding_v2")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    supabase_url = _get_env_or_prompt("SUPABASE_URL")
    supabase_key = _get_env_or_prompt("SUPABASE_SERVICE_KEY")

    supabase = create_client(supabase_url, supabase_key)
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length

    if args.query:
        _search(
            supabase=supabase,
            model=model,
            query=args.query,
            match_count=args.match_count,
            include_parent=args.include_parent,
            max_chars=args.max_chars,
        )
        return

    print("Enter a question, or press Ctrl+C to exit.")
    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query:
            continue

        _search(
            supabase=supabase,
            model=model,
            query=query,
            match_count=args.match_count,
            include_parent=args.include_parent,
            max_chars=args.max_chars,
        )


if __name__ == "__main__":
    main()
