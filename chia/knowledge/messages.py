from chia import instrumentation


class ConceptChangeMessage(instrumentation.Message):
    def __init__(self, sender: str):
        super().__init__(sender=sender)


class RelationChangeMessage(instrumentation.Message):
    def __init__(self, sender: str):
        super().__init__(sender=sender)
