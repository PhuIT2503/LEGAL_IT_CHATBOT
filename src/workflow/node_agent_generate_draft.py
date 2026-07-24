def make_node_agent_generate_draft(generation_agent):
    def node(state: dict) -> dict:
        return generation_agent.run_draft(state)
    return node
