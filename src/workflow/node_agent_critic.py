def make_node_agent_critic(critic_agent):
    def node(state: dict) -> dict:
        return critic_agent.run(state)
    return node
