def make_node_agent_router(router_agent):
    def node(state: dict) -> dict:
        return router_agent.run(state)
    return node
