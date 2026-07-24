def make_node_agent_retrieval(retrieval_agent):
    def node(state: dict) -> dict:
        return retrieval_agent.run(state)
    return node
