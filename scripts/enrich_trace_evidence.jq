.evidence_summary = {
  testset_id: .case.id,
  category: .case.category,
  question: .case.question,
  required_facts: .case.required_facts,
  gold_dieu_ids: .case.gold_article_ids,
  mode: .run.mode,
  top_k: .run.top_k,
  retrieved_child_ids: .retrieval.retrieved_child_ids,
  retrieved_dieu_ids: .retrieval.retrieved_article_ids,
  draft_answer: .critic.draft_answer,
  detected_gap_types: ([
    .critic.detected_gaps
    | to_entries[]
    | select(
        ((.value | type) == "array" and (.value | length) > 0)
        or ((.value | type) != "array" and .value != null and .value != false)
      )
    | .key
  ]),
  candidate_dieu_ids: .critic.candidate_article_ids,
  graph_fetched_dieu_ids: .critic.graph_fetched_article_ids,
  gate_decisions: [.critic.gate_observations[].decision],
  accepted_dieu_ids: .critic.gate_decision_by_article.accepted_article_ids,
  rejected_dieu_ids: .critic.gate_decision_by_article.rejected_article_ids,
  final_context_dieu_ids: (
    [.retrieval.retrieved_article_ids[], .critic.graph_fetched_article_ids[]]
    | unique
  ),
  final_answer: .generation.final_answer,
  required_facts_coverage: "not_logged_by_pipeline",
  token_usage: .usage.total,
  llm_call_count: .usage.total.call_count,
  errors: .errors,
  warnings: .warnings
}
