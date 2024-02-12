import kappaconfig as kc


class MinModelPreProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            if "model_key" in parent_accessor:
                # replace the value before the first _ with "debug"
                # e.g. model_key: small --> model_key: debug
                # e.g. model_key: small_uneven --> model_key: debug_uneven
                actual = parent[parent_accessor].value
                postfixes = actual.split("_")[1:]
                new_key = "_".join(["debug"] + postfixes)
                parent[parent_accessor] = kc.from_primitive(new_key)
