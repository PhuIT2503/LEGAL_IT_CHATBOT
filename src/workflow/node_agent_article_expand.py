def make_node_agent_article_expand(article_expand_agent):
    def node(state: dict) -> dict:
        return article_expand_agent.run(state)
    return node
