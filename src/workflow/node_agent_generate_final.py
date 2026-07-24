def make_node_agent_generate_final(generation_agent):
    def node(state: dict) -> dict:
        return generation_agent.run_final(state)
    return node
