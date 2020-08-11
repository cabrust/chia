class Sample:
    def __init__(self, source=None, data=None, history=None, uid=None):
        if history is not None:
            self.history = history
        elif source is not None:
            self.history = [("init", "", source)]
        else:
            self.history = []

        if data is not None:
            self.data = data
            if "_lazy_resources" not in self.data.keys():
                self.data["_lazy_resources"] = {}
        else:
            if uid is not None:
                self.data = {"uid": uid, "_lazy_resources": {}}
            else:
                raise ValueError("Need UID for sample!")

    def add_resource(self, source, resource_id, datum):
        assert resource_id not in self.data.keys()

        new_history = self.history + [("add", resource_id, source)]
        new_data = {resource_id: datum, **self.data}

        return Sample(data=new_data, history=new_history)

    def add_lazy_resource(self, source, resource_id, fn):
        new_history = self.history + [("add_lazy", resource_id, source)]

        new_lazy_resources = {resource_id: fn, **self.data["_lazy_resources"]}
        new_data = {
            "_lazy_resources": new_lazy_resources,
            **{k: v for k, v in self.data.items() if k != "_lazy_resources"},
        }

        return Sample(data=new_data, history=new_history)

    def apply_on_resource(self, source, resource_id, fn):
        assert resource_id in self.data.keys()
        new_history = self.history + [("apply", resource_id, source)]
        new_data = {k: v if k != resource_id else fn(v) for k, v in self.data.items()}

        return Sample(data=new_data, history=new_history)

    def get_resource(self, resource_id):
        if resource_id in self.data.keys():
            return self.data[resource_id]
        elif resource_id in self.data["_lazy_resources"].keys():
            return self.data["_lazy_resources"][resource_id](self)
        else:
            raise ValueError(f"Unknown resource id {resource_id}")

    def get_resource_ids(self):
        return [k for k in self.data.keys() if not k.startswith("_")] + list(
            self.data["_lazy_resources"].keys()
        )

    def has_resource(self, resource_id):
        return (
            resource_id in self.data.keys()
            or resource_id in self.data["_lazy_resources"].keys()
        )

    def __eq__(self, other):
        return self.data["uid"] == other.data["uid"]

    def __hash__(self):
        return hash(self.data["uid"])
