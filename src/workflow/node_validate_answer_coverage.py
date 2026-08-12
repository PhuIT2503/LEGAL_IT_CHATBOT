from src.agents.agent_generation.citation_coverage import find_missing_required_citations


def validate_answer_coverage_node(state: dict) -> dict:
    """Deterministically check whether the final answer names mandatory articles."""
    missing = find_missing_required_citations(
        state.get("final_response", ""),
        state.get("required_citations", []),
    )
    return {"missing_required_citations": missing}
